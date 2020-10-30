import argparse
import logging
import os
import torch

from model.utils import Params, set_logger
from model.input_pipeline import fetch_pipeline
from model.model_structure import model_fn
from model.evaluation import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='experiments/base_model/', help='Experiment directory containing params.json')
parser.add_argument('--data_dir', type=str, default='../../Dataset/ArtNet/', help='Directory containing the dataset')
parser.add_argument('--restore_from', type=str, default='experiments/best_model/best_weights/best_after_epoch_163.pth.tar',
                    help='Optional, file containing weights to reload before training')


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

  # Set the testing logger for updates
  set_logger(os.path.join(args.model_dir, 'inference.log'))

  logging.info("Creating input pipelines...")

  data_pipelines = fetch_pipeline(['test'], args.data_dir, params)
  test_pipeline = data_pipelines['test']
  logging.info("Completed (Testing Dataset)!")

  logging.info("Building network model...")
  model_spec = model_fn(params)
  logging.info("Building completed!")

  logging.info("Initiate inference procedure!")
  evaluate(model_spec, test_pipeline, args.model_dir, params, args.restore_from)
  logging.info("Inference completed!")


if __name__ == '__main__':
  main()
