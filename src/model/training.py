import os
import logging
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.model_structure import RunningAverage
from model.evaluation import evaluate_session
from model.utils import save_checkpoint_and_weights, get_random_noise_tensor, get_discriminator_loss_strided, \
  get_discriminator_loss_conv


def train_session(model_spec, pipeline, epoch, writer, params):
  """
  Train the model on batches given by the input pipeline

  :param model_spec: (torch.nn.Module) the neural network
  :param pipeline: (DataLoader) the training input pipeline
  :param epoch: (Integer) the epoch that we are currently on
  :param writer: (SummaryWriter) used to stored training summary results in TensorBoard
  :param params: (Params) hyper-parameters
  :return: (dict) the batch mean metrics - loss and accuracy etc
  """
  model_G = model_spec['models']['model_G']
  model_D = model_spec['models']['model_D']

  criterion_D = model_spec['losses']['criterion_D']
  criterion_G = model_spec['losses']['criterion_G']
  L1_criterion_G = model_spec['losses']['L1_criterion_G']

  optimizer_D = model_spec['optimizers']['optimizer_D']
  optimizer_G = model_spec['optimizers']['optimizer_G']

  criterion_mse = model_spec['metrics']['MSE']
  criterion_ssim = model_spec['metrics']['SSIM']
  criterion_pp_acc = model_spec['metrics']['per_pixel_accuracy']

  # set model to training mode
  model_G.train()
  model_D.train()

  # summary for current training loop and a running average object for loss
  summ = []
  average_loss_D = RunningAverage()
  average_loss_G = RunningAverage()
  average_per_pixel_acc = RunningAverage()
  average_mse = RunningAverage()
  average_ssim = RunningAverage()
  logging.info("Training Session Running...")
  with tqdm(total=len(pipeline)) as t:
    for i, (image_real, image_masked) in enumerate(pipeline):
      image_real = image_real.to(params.device)
      image_masked = image_masked.to(params.device)
      batch_size = image_real.shape[0]

      # Discriminator ################################################################################################
      model_D.zero_grad()

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

      # update discriminator weights
      loss_D.backward()
      optimizer_D.step()

      # Generator ##################################################################################################
      model_G.zero_grad()
      loss_G_only, _ = get_discriminator_loss_conv(fake, image_masked, params.patch_size, 'fake_G', model_D,
                                                   criterion_G,
                                                   params.image_size,
                                                   params.device)

      loss_G_L1 = L1_criterion_G(fake, image_real) * params.L1_lambda  # L1 loss beterrn fake and real images
      loss_G = loss_G_only + loss_G_L1  # aggregated generator loss

      # update generator weights
      loss_G.backward()
      optimizer_G.step()

      # Metrics ####################################################################################################
      # extract data from torch Tensors, move to cpu
      batch_real = image_real.detach().data.cpu()
      batch_fake = fake.detach().data.cpu()

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

      # update the average losses for both discriminator and generator
      # also update the metrics
      # this averages is only for visualisation in the progress bar
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
      if i % params.save_generated_img_steps == 0:
        with torch.no_grad():
          # generate samples to display in evaluation mode
          noise_tensor = get_random_noise_tensor(batch_size, params.num_channels, params.image_size, params)
          fake = model_G(image_masked, noise_tensor)

          # unnormalize
          image_real = (image_real * 0.5 + 0.5)
          image_masked = (image_masked * 0.5 + 0.5)
          fake = (fake * 0.5 + 0.5)

          # create image grids for visualization
          img_grid_real = torchvision.utils.make_grid(image_real[:32], normalize=True, range=(0, 1))
          img_grid_masked = torchvision.utils.make_grid(image_masked[:32], normalize=True, range=(0, 1))
          img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True, range=(0, 1))

          # combine the grids
          img_grid_combined = torch.stack((img_grid_real, img_grid_masked, img_grid_fake))
          torchvision.utils.save_image(img_grid_combined, os.path.join('outputs', f'{epoch}_{i}.jpg'))

          # write to tensorboard
          writer.add_image('Real_Images', img_grid_real)
          writer.add_image('Masked_Images', img_grid_masked)
          writer.add_image('Fake_Images', img_grid_fake)

      t.update()

  # compute mean of all metrics in summary
  metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
  metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
  logging.info("- Train metrics: " + metrics_string)

  logging.info("Training Session Finished")

  model_spec['models']['model_G'] = model_G
  model_spec['models']['model_D'] = model_D
  model_spec['optimizers']['optimizer_G'] = optimizer_G
  model_spec['optimizers']['optimizer_D'] = optimizer_D

  return metrics_mean, model_spec


def train_and_validate(model_spec, train_pipeline, valid_pipeline, model_dir, params, restore_from):
  """
  Train the model and validate every epoch

  :param model_spec: (Dictionary), structure that contains the graph operations or nodes needed for training and validation
  :param train_pipeline: (DataLoader), Training input pipeline
  :param valid_pipeline: (DataLoader), Validation input pipeline
  :param model_dir: (String), directory containing config, weights and logs
  :param params: (Params), contains hyper-parameters of the model. Must define: num_epochs, batch_size, save_summary_steps, ... etc
  """
  if restore_from is not None:
    if not os.path.exists(restore_from):
      raise FileNotFoundError("File {} doesn't exist".format(restore_from))

    checkpoint = torch.load(restore_from)

    model_spec['models']['model_G'].load_state_dict(checkpoint['G_state_dict'])
    model_spec['models']['model_D'].load_state_dict(checkpoint['D_state_dict'])
    model_spec['optimizers']['optimizer_G'].load_state_dict(checkpoint['G_optim_dict'])
    model_spec['optimizers']['optimizer_D'].load_state_dict(checkpoint['D_optim_dict'])

    begin_at_epoch = checkpoint['epoch']
  else:
    begin_at_epoch = 0

  best_valid_loss_G = np.Infinity
  # For tensorBoard (takes care of writing summaries to files)
  train_writer = SummaryWriter(os.path.join(model_dir, 'train_summaries'))
  eval_writer = SummaryWriter(os.path.join(model_dir, 'eval_summaries'))

  for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
    # Run one epoch
    logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

    # compute number of batches in one epoch (one full pass over the training dataset)
    train_mean_metrics, train_model_spec = train_session(model_spec, train_pipeline, epoch, train_writer, params)

    for k, v in train_mean_metrics.items():
      train_writer.add_scalar(k, v, global_step=epoch + 1)

    # Evaluate for one epoch on the validation dataset
    valid_mean_metrics = evaluate_session(model_spec, valid_pipeline, eval_writer, params)
    for k, v in valid_mean_metrics.items():
      eval_writer.add_scalar(k, v, global_step=epoch + 1)

    valid_loss_G = valid_mean_metrics['loss_G']

    save_dict = {'epoch': epoch + 1,
                 'G_state_dict': model_spec['models']['model_G'].state_dict(),
                 'G_optim_dict': model_spec['optimizers']['optimizer_G'].state_dict(),
                 'D_state_dict': model_spec['models']['model_D'].state_dict(),
                 'D_optim_dict': model_spec['optimizers']['optimizer_D'].state_dict()
                 }

    if valid_loss_G <= best_valid_loss_G:
      # Store new best loss
      best_valid_loss_G = valid_loss_G

      save_checkpoint_and_weights(model_dir, save_dict, epoch, valid_mean_metrics, checkpoint='best')

      logging.info("Found new best accuracy, after epoch {}".format(epoch + 1))

    save_checkpoint_and_weights(model_dir, save_dict, epoch, valid_mean_metrics, checkpoint='last')

  train_writer.flush()
  eval_writer.flush()

  train_writer.close()
  eval_writer.close()
