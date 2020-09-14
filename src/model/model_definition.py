import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
  """
  Our CNN network that follows the given implementation.
  """

  def __init__(self, params):
    super().__init__()
    self.num_channels = params.num_channels
    self.num_filters = params.num_filters

    self.conv1 = ...  # FIXME network abstract construction

    # TODO we will use this trick for dynamically infer the input features in each we need FC layers
    # dummy run of one random image to dynamically infer the input feature maps of the fully connected layer
    dummy_data = torch.randn(672, 224).view(-1, 3, 224, 224)
    self.linear_feature_maps = None
    self.convs_block(dummy_data)

    self.fc1 = nn.Linear(self.linear_feature_maps, 1024)
    self.output_fc = nn.Linear(1024, 1)

  # TODO create a modular block code for clarity
  def convs_block(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2), stride=2)
    x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2), stride=2)

    # infer num_features only once to speed uo training
    if self.linear_feature_maps is None:
      self.linear_feature_maps = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    return x

  def forward(self, input):
    x = self.convs_block(input)
    x = x.view(-1, self.linear_feature_maps)
    x = F.relu(self.fc1(x))
    x = self.output_fc(x)

    return x
