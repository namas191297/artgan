import os
import logging
import numpy as np
import torch
from tqdm import tqdm

from model.utils import save_dict_to_json


def evaluate_session(model_spec, pipeline, params):
  """
  Validate or test the model on batches given by the input pipeline

  :param model_spec: (Dictionary), structure that contains the graph operations or nodes needed for validation or testing
  :param pipeline: (DataLoader), Validation or Testing input pipeline
  :param params: (Params), contains hyper-parameters of the model. Must define: num_epochs, batch_size, save_summary_steps, ... etc
  :return: (dict) the batch mean metrics - loss and accuracy etc
  """
  # FIXME define the variables we want from dictionary model_spec

  # set model to evaluation mode (useful for dropout and batch normalisation layers)
  model.eval()

  # summary for current evaluation loop and a running average object for loss
  summ = []

  logging.info("Evaluation Session Running...")
  # torch.no_grad() to remove the training effect of BatchNorm in this case as it evaluates the model
  with torch.no_grad():
    with tqdm(total=len(pipeline)) as t:
      for batch_X, batch_y in pipeline:
        # FIXME create the content of the loop. This will count for every batch iteration
        pass
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
  for c in checkpoints:
    best_epochs.append(c.split('.')[0].split('_')[-1])

  checkpoint = os.path.join(model_dir, 'best_weights', restore_from + '{}.pth.tar'.format(best_epochs[-1]))
  if not os.path.exists(checkpoint):
    raise ("File doesn't exist {}".format(checkpoint))

  checkpoint = torch.load(checkpoint)
  model_spec['net_model'].load_state_dict(checkpoint['state_dict'])

  # Inference
  test_metrics = evaluate_session(model_spec, pipeline, params)
  save_path = os.path.join(model_dir, "metrics_test_{}.json".format(restore_from))
  save_dict_to_json(test_metrics, save_path)
