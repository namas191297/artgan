import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import model.model_definition as model_def


def model_fn(params):
  """
  Model function defining the graph operations

  :param params: (Params), The hyper-parameters inside json file for the model's operations
  :return: (dict), A dictionary that contains the graph operations or nodes needed for training/validation/testing
  """
  # TODO here we will have all the training logit, such as computation of metrics, losses, learning rate scheduler
  # TODO might need to create new files for metrics if we use multiple custom ones... we'll see

  # takes cases on whether we want to train from scratch or use pre-trained weights from ResNet-18 in ImageNet dataset and then re-train the model
  # from there
  if params.pretrained_transfer:
    model = torchvision.models.resnet18(pretrained=True)
    linear_feature_maps = model.fc.in_features
    model.fc = nn.Linear(linear_feature_maps, 1)

    model = model.to(params.device)
  else:
    model = model_def.Network(params).to(params.device)
  loss_fn = nn.BCEWithLogitsLoss()
  optimiser = optim.Adam(model.parameters(), lr=params.learning_rate)

  # TODO important create a dictionary with all the model specs and carry this across our code modules
  # create a bundle that contains all the necessary components of the model
  model_spec = {'net_model': model, 'loss_fn': loss_fn, 'optimiser': optimiser, 'metrics': {'accuracy': None}}

  return model_spec
