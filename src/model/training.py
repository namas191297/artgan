import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.evaluation import evaluate_session
from model.utils import save_dict_to_json


def train_session(model_spec, pipeline, params):
  """
  Train the model on batches given by the input pipeline

  :param model: (torch.nn.Module) the neural network
  :param pipeline: (DataLoader) the training input pipeline
  :param params: (Params) hyper-parameters
  :return: (dict) the batch mean metrics - loss and accuracy etc
  """
  # FIXME define the variables we want from dictionary model_spec

  # set model to training mode
  model.train()

  # summary for current training loop and a running average object for loss
  summ = []

  logging.info("Training Session Running...")
  with tqdm(total=len(pipeline)) as t:
    for i, (batch_X, batch_y) in enumerate(pipeline):
      # FIXME implement the loop
      t.update()

  # compute mean of all metrics in summary
  metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
  metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
  logging.info("- Train metrics: " + metrics_string)

  logging.info("Training Session Finished")

  return metrics_mean


def train_and_validate(model_spec, train_pipeline, valid_pipeline, model_dir, params):
  """
  Train the model and validate every epoch

  :param model_spec: (Dictionary), structure that contains the graph operations or nodes needed for training and validation
  :param train_pipeline: (DataLoader), Training input pipeline
  :param valid_pipeline: (DataLoader), Validation input pipeline
  :param model_dir: (String), directory containing config, weights and logs
  :param params: (Params), contains hyper-parameters of the model. Must define: num_epochs, batch_size, save_summary_steps, ... etc
  """
  begin_at_epoch = 0
  # FIXME in our case it's to evaluate on the loss
  best_valid_accuracy = 0.0
  # For tensorBoard (takes care of writing summaries to files)
  train_writer = SummaryWriter(os.path.join(model_dir, 'train_summaries'))
  eval_writer = SummaryWriter(os.path.join(model_dir, 'eval_summaries'))

  for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
    # Run one epoch
    logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

    # compute number of batches in one epoch (one full pass over the training dataset)
    train_mean_metrics = train_session(model_spec, train_pipeline, params)

    for k, v in train_mean_metrics.items():
      train_writer.add_scalar(k, v, global_step=epoch + 1)

    # Evaluate for one epoch on the validation dataset
    valid_mean_metrics = evaluate_session(model_spec, valid_pipeline, params)
    for k, v in valid_mean_metrics.items():
      eval_writer.add_scalar(k, v, global_step=epoch + 1)

    valid_accuracy = valid_mean_metrics['accuracy']

    if valid_accuracy >= best_valid_accuracy:
      # Store new best accuracy
      best_valid_accuracy = valid_accuracy
      # Save weights
      best_save_directory = os.path.join(model_dir, 'best_weights')
      if not os.path.exists(best_save_directory):
        os.mkdir(best_save_directory)
      # Removes previously stored best model since we found a new one
      files = os.listdir(best_save_directory)
      if len(files) != 0:
        os.remove(os.path.join(best_save_directory, files[0]))

      best_save_path = os.path.join(best_save_directory, 'best_after_epoch_{}.pth.tar'.format(epoch + 1))

      torch.save({'epoch': epoch + 1,
                  'state_dict': model_spec['net_model'].state_dict(),
                  'optim_dict': model_spec['optimiser'].state_dict()},
                 best_save_path)

      logging.info("Found new best accuracy, saving in {}".format(best_save_path))

      # Save best eval metrics in a json file in the model directory
      best_json_path = os.path.join(model_dir, 'metrics_eval_best_weights.json')
      save_dict_to_json(valid_mean_metrics, best_json_path)

  train_writer.flush()
  eval_writer.flush()

  train_writer.close()
  eval_writer.close()
