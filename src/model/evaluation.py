import os
import logging
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.utils import save_dict_to_json
from model.model_structure import RunningAverage


def evaluate_session(model_spec, pipeline, writer, params):
  """
  Validate or test the model on batches given by the input pipeline

  :param model_spec: (Dictionary), structure that contains the graph operations or nodes needed for validation or testing
  :param pipeline: (DataLoader), Validation or Testing input pipeline
  :param writer: (SummaryWriter) used to stored validation/testing summary results in TensorBoard
  :param params: (Params), contains hyper-parameters of the model. Must define: num_epochs, batch_size, save_summary_steps, ... etc
  :return: (dict) the batch mean metrics - loss and accuracy etc
  """
  model_G = model_spec['models']['model_G']
  model_D = model_spec['models']['model_D']

  criterion_D = model_spec['losses']['criterion_D']
  criterion_G = model_spec['losses']['criterion_G']
  L1_criterion_G = model_spec['losses']['L1_criterion_G']

  criterion_mse = model_spec['metrics']['mse']
  criterion_ssim = model_spec['metrics']['ssim']
  criterion_pp_acc = model_spec['metrics']['per_pixel_accuracy']

  # set model to evaluation mode (useful for dropout and batch normalisation layers)
  model_G.eval()
  model_D.eval()

  # summary for current evaluation loop and a running average object for loss
  summ = []
  average_loss_D = RunningAverage()
  average_loss_G = RunningAverage()
  average_per_pixel_acc = RunningAverage()
  average_mse = RunningAverage()
  average_ssim = RunningAverage()
  logging.info("Evaluation Session Running...")
  # torch.no_grad() to remove the training effect of BatchNorm in this case as it evaluates the model
  with torch.no_grad():
    with tqdm(total=len(pipeline)) as t:
      for i, (image_real, image_masked) in enumerate(pipeline):
        image_real = image_real.to(params.device)
        image_masked = image_masked.to(params.device)
        batch_size = image_real.shape[0]

        # Discriminator ################################################################################################
        # real image
        output_D_real = model_D(image_real, image_masked).reshape(-1)  # output of the discriminator for real images
        label_D_real = (torch.ones(batch_size) * 0.9).to(params.device)  # labels for real images, multiplied by 0.9, training hack
        loss_D_real = criterion_D(output_D_real, label_D_real)  # first half of discriminator's loss

        confidence_D = output_D_real.mean().item()  # confidence of the discriminator (probability [0, 1])

        # fake image
        fake = model_G(image_masked)  # generate fakes, given masked images

        # detached so that the generator does not update it's weights while discriminating
        output_D_fake = model_D(fake.detach(), image_masked).reshape(-1)  # output of the discriminator for fake images
        label_D_fake = (torch.ones(batch_size) * 0.1).to(params.device)  # labels for fake images, multiplied by 0.1, training hack
        loss_D_fake = criterion_D(output_D_fake, label_D_fake)  # second half of discriminator's loss

        # aggregate discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * params.loss_D_factor  # multiplied by 0.5 to slow down discriminator's learning

        # Generator ##################################################################################################
        output_D_G_fake = model_D(fake, image_masked).reshape(-1)  # fake, D(G(x)), this time weights are updated
        label_D_G_fake = torch.ones(batch_size).to(params.device)  # labels for fake G(x)

        loss_G_only = criterion_G(output_D_G_fake, label_D_G_fake)  # raw generator loss
        loss_G_L1 = L1_criterion_G(fake, image_real) * params.L1_lambda  # L1 loss beterrn fake and real images
        loss_G = loss_G_only + loss_G_L1  # aggregated generator loss

        # Metrics ####################################################################################################
        # extract data from torch Tensors, move to cpu
        batch_real = image_real.data.cpu()
        batch_fake = fake.data.cpu()

        # Per Pixel Accuracy
        pp_accuracy = criterion_pp_acc(batch_real, batch_fake)

        # Mean Square Error (MSE)
        mse = criterion_mse(batch_real, batch_fake)

        # Structural Similarity Index Measure (SSIM)
        ssim = criterion_ssim(batch_real, batch_fake)

        # Evaluate summaries only once in a while
        # if i % params.save_summary_steps == 0:
        # store per batch metrics for the epoch results
        summary_batch = {'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'pp_acc': pp_accuracy, 'mse': mse.item(), 'ssim': ssim.item()}
        summ.append(summary_batch)

        # update the average losses for both discriminator ang generator
        average_loss_D.update(loss_D.item())
        average_loss_G.update(loss_G.item())
        average_per_pixel_acc.update(pp_accuracy.item())
        average_mse.update(mse.item())
        average_ssim.update(ssim.item())

        # Log the batch loss and accuracy in the tqdm progress bar
        t.set_postfix(confidence_D='{:05.3f}'.format(confidence_D), loss_D='{:05.3f}'.format(average_loss_D()),
                      loss_G='{:05.3f}'.format(average_loss_G()), pp_acc='{:05.3f}'.format(average_per_pixel_acc()),
                      mse='{:05.3f}'.format(average_mse()), ssim='{:05.3f}'.format(average_ssim()))

        # 3 image grids
        if i % params.save_generated_img_steps == 0:
          with torch.no_grad():
            # generate samples to display in evaluation mode
            model_G.eval()
            fake = model_G(image_masked)
            model_G.train()

            # create image grids for visualization
            img_grid_real = torchvision.utils.make_grid(image_real[:32], normalize=True, range=(0, 1))
            img_grid_masked = torchvision.utils.make_grid(image_masked[:32], normalize=True, range=(0, 1))
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True, range=(0, 1))

            # combine the grids
            img_grid_combined = torch.stack((img_grid_real, img_grid_masked, img_grid_fake))
            torchvision.utils.save_image(img_grid_combined, f'output\\validation_{i}.jpg')

            # write to tensorboard
            writer.add_image('Real Images', img_grid_real)
            writer.add_image('Masked Images', img_grid_masked)
            writer.add_image('Fake Images', img_grid_fake)

        t.update()

  metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
  metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
  logging.info("- Evaluation metrics: " + metrics_string)

  logging.info("Evaluation Session Finished!")

  return metrics_mean


def evaluate(model_spec, pipeline, model_dir, params, restore_from):
  """
  Evaluate the model on the test dataset

  :param model_spec: (Dictionary), structure that contains the graph operations or nodes needed for testing
  :param pipeline: (DataLoader), Testing input pipeline
  :param model_dir: (String), directory containing config, weights and logs
  :param params: (Params), contains hyper-parameters of the model. Must define: num_epochs, batch_size, save_summary_steps, ... etc
  :param restore_from: (String), Directory of file containing weights to restore the graph
  """
  # Reload weights from the saved file
  best_weights_dir = os.path.join(model_dir, 'best_weights')
  checkpoints = os.listdir(best_weights_dir)
  best_epochs = []
  test_writer = SummaryWriter(os.path.join(model_dir, 'test_summaries'))

  for c in checkpoints:
    best_epochs.append(c.split('.')[0].split('_')[-1])

  checkpoint = os.path.join(model_dir, 'best_weights', restore_from + '{}.pth.tar'.format(best_epochs[-1]))
  if not os.path.exists(checkpoint):
    raise ("File doesn't exist {}".format(checkpoint))

  checkpoint = torch.load(checkpoint)
  model_spec['models']['model_G'].load_state_dict(checkpoint['G_state_dict'])
  model_spec['models']['model_D'].load_state_dict(checkpoint['D_state_dict'])

  # Inference
  test_metrics = evaluate_session(model_spec, pipeline, test_writer, params)
  test_writer.flush()
  test_writer.close()

  save_path = os.path.join(model_dir, "metrics_test_{}.json".format(restore_from))
  save_dict_to_json(test_metrics, save_path)
