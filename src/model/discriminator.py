import torch
import torch.nn as nn
import torch.nn.functional as F


# input size N x 3 x 256 x 256, for image GAN
# output size N x 1 (maybe, as outputs a single number, not sure if N x 1 x 1 x 1)
class Discriminator_256(nn.Module):
  # initializers
  def __init__(self, num_channels_input, features_D):
    d = features_D
    super(Discriminator_256, self).__init__()
    self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
    self.conv2 = nn.Conv2d(d, d * 2, 4, 4, 1)
    self.conv2_bn = nn.BatchNorm2d(d * 2)

    self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 4, 1)
    self.conv3_bn = nn.BatchNorm2d(d * 4)

    self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 4, 1)
    self.conv4_bn = nn.BatchNorm2d(d * 8)
    self.conv5 = nn.Conv2d(d * 8, 1, 4, 4, 1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input, label):
    x = torch.cat([input, label], 1)
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
    x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
    x = F.sigmoid(self.conv5(x))
    # print(x.shape)
    return x


def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()
