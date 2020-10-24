import argparse
import os
import logging
import torch
from model.utils import Params, set_logger
from model.input_pipeline import fetch_pipeline
from model.model_structure import model_fn
from model.training import train_and_validate

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='experiments/base_model/', help='Experiment directory containing params.json')
parser.add_argument('--data_dir', type=str, default='../../Dataset/ArtNet/', help='Directory containing the dataset')
parser.add_argument('--restore_from', type=str, default=None, help='Optional, directory or file containing weights to reload before training')


def main():
  # Load parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = Params(json_path)

  # use GPU if found
  params.cuda = torch.cuda.is_available()
  if params.cuda:
    params.device = torch.device('cuda:0')
  else:
    params.device = torch.device('cpu')

  # Set a seed for reproducible experiments
  torch.manual_seed(141)
  if params.cuda:
    torch.cuda.manual_seed(141)

  # Set the training logger for updates
  set_logger(os.path.join(args.model_dir, 'train.log'))

  logging.info("Creating input pipelines...")

  data_pipelines = fetch_pipeline(['train', 'validation'], args.data_dir, params)
  train_pipeline = data_pipelines['train']
  logging.info("Completed (Training Dataset)!")
  valid_pipeline = data_pipelines['validation']
  logging.info("Completed (Validation Dataset)!")

  logging.info("Building network model...")
  model_spec = model_fn(params)
  logging.info("Building completed!")

  logging.info("Initiate training procedure!")
  train_and_validate(model_spec, train_pipeline, valid_pipeline, args.model_dir, params, args.restore_from)
  logging.info("Training completed!")


if __name__ == '__main__':
  main()
