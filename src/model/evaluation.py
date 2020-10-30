import os
import logging
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.utils import save_image_batch
from fid import get_inception_features
import pickle

from model.utils import save_dict_to_json, get_random_noise_tensor, get_discriminator_loss_strided, \
  get_discriminator_loss_conv
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

  criterion_mse = model_spec['metrics']['MSE']
  criterion_ssim = model_spec['metrics']['SSIM']
  criterion_pp_acc = model_spec['metrics']['per_pixel_accuracy']

  # summary for current evaluation loop and a running average object for loss
  summ = []
  average_loss_D = RunningAverage()
  average_loss_G = RunningAverage()
  average_per_pixel_acc = RunningAverage()
  average_mse = RunningAverage()
  average_ssim = RunningAverage()
  logging.info("Evaluation Session Running...")

  frechet_real = []
  frechet_fake = []
  # torch.no_grad() to remove the training effect of BatchNorm in this case as it evaluates the model
  with torch.no_grad():
    with tqdm(total=len(pipeline)) as t:
      for i, (image_real, image_masked) in enumerate(pipeline):
        image_real = image_real.to(params.device)
        image_masked = image_masked.to(params.device)
        batch_size = image_real.shape[0]

        # Discriminator ################################################################################################
        # real image
        loss_D_real, confidence_D = get_discriminator_loss_conv(image_real, image_masked, params.patch_size, 'real_D',
                                                                model_D,
                                                                criterion_D,
                                                                params.image_size, params.device)

        # fake image
        noise_tensor = get_random_noise_tensor(batch_size, params.num_channels, params.image_size, params)
        fake = model_G(image_masked, noise_tensor)  # generate fakes, given masked images

        loss_D_fake, _ = get_discriminator_loss_conv(fake.detach(), image_masked, params.patch_size, 'fake_D', model_D,
                                                     criterion_D, params.image_size,
                                                     params.device)

        # aggregate discriminator loss
        loss_D = (loss_D_real + (
          loss_D_fake)) * params.loss_D_factor  # multiplied by 0.5 to slow down discriminator's learning

        # Generator ##################################################################################################
        loss_G_only, _ = get_discriminator_loss_conv(fake, image_masked, params.patch_size, 'fake_G', model_D,
                                                     criterion_G,
                                                     params.image_size,
                                                     params.device)

        loss_G_L1 = L1_criterion_G(fake, image_real) * params.L1_lambda  # L1 loss between fake and real images
        loss_G = loss_G_only + loss_G_L1  # aggregated generator loss

        # Metrics ####################################################################################################
        # extract data from torch Tensors, move to cpu
        batch_real = image_real.data.cpu()
        batch_fake = fake.data.cpu()

        # Per Pixel Accuracy
        pp_accuracy = criterion_pp_acc(batch_real, batch_fake, params)

        # Mean Square Error (MSE)
        mse = criterion_mse(batch_real, batch_fake)

        # Structural Similarity Index Measure (SSIM)
        ssim = criterion_ssim(batch_real, batch_fake)

        # Evaluate summaries only once in a while
        # if i % params.save_summary_steps == 0:
        # store per batch metrics for the epoch results
        summary_batch = {'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'pp_acc': pp_accuracy, 'mse': mse.item(),
                         'ssim': ssim.item()}
        summ.append(summary_batch)

        # update the average losses for both discriminator ang generator
        average_loss_D.update(loss_D.item())
        average_loss_G.update(loss_G.item())
        average_per_pixel_acc.update(pp_accuracy)
        average_mse.update(mse.item())
        average_ssim.update(ssim.item())

        # Log the batch loss and accuracy in the tqdm progress bar
        t.set_postfix(confidence_D='{:05.3f}'.format(confidence_D), loss_D='{:05.3f}'.format(average_loss_D()),
                      loss_G='{:05.3f}'.format(average_loss_G()), pp_acc='{:05.3f}'.format(average_per_pixel_acc()),
                      mse='{:05.3f}'.format(average_mse()), ssim='{:05.3f}'.format(average_ssim()))

        # 3 image grids
        if i % 1 == 0:
          with torch.no_grad():
            # generate samples to display in evaluation mode
            noise_tensor = get_random_noise_tensor(batch_size, params.num_channels, params.image_size, params)
            fake = model_G(image_masked, noise_tensor)

            reals, fakes = get_inception_features(image_real, fake)
            frechet_real.append(reals.numpy())
            frechet_fake.append(fakes.numpy())
            # print(f'frechet_distance: {frechet_distance}')

            image_real = (image_real * 0.5 + 0.5)
            image_masked = (image_masked * 0.5 + 0.5)
            fake = (fake * 0.5 + 0.5)

            # create image grids for visualization
            img_grid_real = torchvision.utils.make_grid(image_real[:32], normalize=True, range=(0, 1))
            img_grid_masked = torchvision.utils.make_grid(image_masked[:32], normalize=True, range=(0, 1))
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True, range=(0, 1))
            save_image_batch(image_real, i, 'real')
            save_image_batch(image_masked, i, 'masked')
            save_image_batch(fake, i, 'fake')

            # combine the grids
            img_grid_combined = torch.stack((img_grid_real, img_grid_masked, img_grid_fake))
            torchvision.utils.save_image(img_grid_combined, os.path.join('outputs', f'validation_{i}.jpg'))

            # write to tensorboard
            writer.add_image('Real_Images', img_grid_real)
            writer.add_image('Masked_Images', img_grid_masked)
            writer.add_image('Fake_Images', img_grid_fake)

        t.update()

  with open('frehcet.pkl', 'wb') as f:
    pickle.dump([frechet_real, frechet_fake], f)

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
  test_writer = SummaryWriter(os.path.join(model_dir, 'test_summaries'))

  if not os.path.exists(restore_from):
    raise FileNotFoundError("File doesn't exist {}".format(restore_from))

  checkpoint = torch.load(restore_from, map_location='cpu')
  model_spec['models']['model_G'].load_state_dict(checkpoint['G_state_dict'])
  model_spec['models']['model_D'].load_state_dict(checkpoint['D_state_dict'])

  # Inference
  test_metrics = evaluate_session(model_spec, pipeline, test_writer, params)
  test_writer.flush()
  test_writer.close()

  save_path = os.path.join(model_dir, "metrics_test_best.json")
  save_dict_to_json(test_metrics, save_path)
