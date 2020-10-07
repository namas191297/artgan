import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from model.discriminator import Discriminator_256
from model.generator import DenseUNet


def model_fn(params):
  """
  Model function defining the graph operations

  :param params: (Params), The hyper-parameters inside json file for the model's operations
  :return: (dict), A dictionary that contains the graph operations or nodes needed for training/validation/testing
  """
  # TODO here we will have all the training logit, such as computation of metrics, losses, learning rate scheduler
  # TODO might need to create new files for metrics if we use multiple custom ones... we'll see

  model_D = Discriminator_256(params.num_channels_input, params.features_D).to(params.device)
  model_G = DenseUNet(params.num_channels_input, params.features_G).to(params.device)

  loss_G = nn.BCELoss()
  loss_D = nn.BCELoss()

  L1_loss_G = nn.L1Loss()

  optimizer_D = optim.Adam(model_D.parameters(), lr=params.lr_D, betas=(0.5, 0.999))
  optimizer_G = optim.Adam(model_G.parameters(), lr=params.lr_G, betas=(0.5, 0.999))

  models = {'model_D': model_D, 'model_G': model_G}
  losses = {'loss_G': loss_G, 'loss_D': loss_D, 'L1_loss_G': L1_loss_G}
  optimizers = {'optimizer_D': optimizer_D, 'optimizer_G': optimizer_G}

  # TODO implement metrics
  # metrics = {}

  model_spec = {'models': models,
                'losses': losses,
                'optimizers': optimizers}

  return model_spec


class RunningAverage:
  """
  Class that maintains a running average of a metric across the dataset batches.
  To be used in the loss computation per epoch.
  """

  def __init__(self):
    self.count = 0
    self.total = 0

  def update(self, val):
    self.total += val
    self.count += 1

  def __call__(self):
    return self.total / float(self.count)
