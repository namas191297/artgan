import os
import logging
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.utils import save_dict_to_json, get_random_noise_tensor, get_discriminator_loss_strided, \
  get_discriminator_loss_conv, get_discriminator_loss
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
  model_G_c = model_spec['models']['model_G_c']
  model_G_r = model_spec['models']['model_G_r']
  model_D_c = model_spec['models']['model_D_c']
  model_D_r = model_spec['models']['model_D_r']

  criterion_D = model_spec['losses']['criterion_D']
  criterion_G = model_spec['losses']['criterion_G']
  L1_criterion_G = model_spec['losses']['L1_criterion_G']

  optimizer_D_c = model_spec['optimizers']['optimizer_D_c']
  optimizer_D_r = model_spec['optimizers']['optimizer_D_r']
  optimizer_G_c = model_spec['optimizers']['optimizer_G_c']
  optimizer_G_r = model_spec['optimizers']['optimizer_G_r']

  criterion_mse = model_spec['metrics']['MSE']
  criterion_ssim = model_spec['metrics']['SSIM']
  criterion_pp_acc = model_spec['metrics']['per_pixel_accuracy']

  # set model to training mode
  model_G_c.train()
  model_G_r.train()
  model_D_c.train()
  model_D_r.train()

  # summary for current training loop and a running average object for loss
  summ = []
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
        model_D_c.zero_grad()
        model_D_r.zero_grad()

        # real image coarse
        fake_coarse, loss_D_c, model_D_c, model_G_c, confidence_D_c = get_discriminator_loss(image_real,
                                                                                             image_masked,
                                                                                             model_D_c,
                                                                                             model_G_c,
                                                                                             criterion_D,
                                                                                             batch_size,
                                                                                             params)

        coarse_image = fake_coarse.detach()
        # real image refined
        fake_refined, loss_D_r, model_D_r, model_G_r, confidence_D_r = get_discriminator_loss(image_real,
                                                                                              image_masked,
                                                                                              model_D_r,
                                                                                              model_G_r,
                                                                                              criterion_D,
                                                                                              batch_size,
                                                                                              params,
                                                                                              coarse_image)

        # Generator ##################################################################################################
        loss_G_c_only, _ = get_discriminator_loss_conv(fake_coarse, image_masked, params.patch_size, 'fake_G',
                                                       model_D_c,
                                                       criterion_G,
                                                       params.image_size,
                                                       params.device)

        loss_G_r_only, _ = get_discriminator_loss_conv(fake_refined, image_masked, params.patch_size, 'fake_G',
                                                       model_D_r, criterion_G,
                                                       params.image_size,
                                                       params.device)

        loss_G_c_L1 = L1_criterion_G(fake_coarse, image_real)  # L1 loss beterrn fake and real images
        loss_G_c = loss_G_c_only * params.L1_lambda + loss_G_c_L1  # aggregated generator loss

        loss_G_r_L1 = L1_criterion_G(fake_refined,
                                     image_real) * params.L1_lambda  # L1 loss beterrn fake and real images
        loss_G_r = loss_G_r_only + loss_G_r_L1  # aggregated generator loss

        # Metrics ####################################################################################################
        # extract data from torch Tensors, move to cpu
        batch_real = image_real.detach().data.cpu()
        batch_fake = fake_refined.detach().data.cpu()

        # Per Pixel Accuracy
        pp_accuracy = criterion_pp_acc(batch_real, batch_fake, params)

        # Mean Square Error (MSE)
        mse = criterion_mse(batch_real, batch_fake)

        # Structural Similarity Index Measure (SSIM)
        ssim = criterion_ssim(batch_real, batch_fake)

        # Evaluate summaries only once in a while
        # if i % params.save_summary_steps == 0:
        # store per batch metrics for the epoch results
        summary_batch = {'loss_D_c': loss_D_c.item(), 'loss_D_r': loss_D_r.item(), 'loss_G_c': loss_G_c.item(),
                         'loss_G_r': loss_G_r.item(), 'pp_acc': pp_accuracy, 'mse': mse.item(), 'ssim': ssim.item()}
        summ.append(summary_batch)

        # update the average losses for both discriminator and generator
        # also update the metrics
        # this averages is only for visualisation in the progress bar

        average_per_pixel_acc.update(pp_accuracy)
        average_mse.update(mse.item())
        average_ssim.update(ssim.item())

        # 3 image grids
        if i % params.save_generated_img_steps == 0:
          with torch.no_grad():
            # generate samples to display in evaluation mode
            noise_tensor = get_random_noise_tensor(batch_size, params.num_channels, params.image_size, params)
            fake_coarse = model_G_c(image_masked, noise_tensor)
            fake_refined = model_G_r(image_masked, fake_coarse)

            # unnormalize
            image_real = (image_real * 0.5 + 0.5)
            image_masked = (image_masked * 0.5 + 0.5)
            fake_coarse = (fake_coarse * 0.5 + 0.5)
            fake_refined = (fake_refined * 0.5 + 0.5)

            # create image grids for visualization
            img_grid_real = torchvision.utils.make_grid(image_real[:32], normalize=True, range=(0, 1))
            img_grid_masked = torchvision.utils.make_grid(image_masked[:32], normalize=True, range=(0, 1))
            img_grid_fake_c = torchvision.utils.make_grid(fake_coarse[:32], normalize=True, range=(0, 1))
            img_grid_fake_r = torchvision.utils.make_grid(fake_refined[:32], normalize=True, range=(0, 1))

            # combine the grids
            img_grid_combined = torch.stack((img_grid_real, img_grid_masked, img_grid_fake_c, img_grid_fake_r))
            torchvision.utils.save_image(img_grid_combined, os.path.join('outputs', f'validation_{i}.jpg'))

            # write to tensorboard
            writer.add_image('Real_Images', img_grid_real)
            writer.add_image('Masked_Images', img_grid_masked)
            writer.add_image('Fake_Images', img_grid_fake_c)
            writer.add_image('Fake_Images', img_grid_fake_r)

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
  test_writer = SummaryWriter(os.path.join(model_dir, 'test_summaries'))

  if not os.path.exists(restore_from):
    raise FileNotFoundError("File {} doesn't exist".format(restore_from))

  checkpoint = torch.load(restore_from)
  model_spec['models']['model_G_c'].load_state_dict(checkpoint['G_c_state_dict'])
  model_spec['models']['model_G_r'].load_state_dict(checkpoint['G_r_state_dict'])
  model_spec['optimizers']['optimizer_G_c'].load_state_dict(checkpoint['G_c_optim_dict'])
  model_spec['optimizers']['optimizer_G_r'].load_state_dict(checkpoint['G_r_optim_dict'])
  model_spec['models']['model_D_c'].load_state_dict(checkpoint['D_c_state_dict'])
  model_spec['models']['model_D_r'].load_state_dict(checkpoint['D_r_state_dict'])
  model_spec['optimizers']['optimizer_D_c'].load_state_dict(checkpoint['D_c_optim_dict'])
  model_spec['optimizers']['optimizer_D_c'].load_state_dict(checkpoint['D_r_optim_dict'])

  # Inference
  test_metrics = evaluate_session(model_spec, pipeline, test_writer, params)
  test_writer.flush()
  test_writer.close()

  save_path = os.path.join(model_dir, "metrics_test_{}.json".format(restore_from))
  save_dict_to_json(test_metrics, save_path)
