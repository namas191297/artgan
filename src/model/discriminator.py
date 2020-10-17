import torch
import torch.nn as nn


def get_discriminator(patch_size, num_channels_input, features_D):
  if patch_size == 1:
    model = Discriminator_1(num_channels_input, features_D)
  elif patch_size == 16:
    model = Discriminator_16(num_channels_input, features_D)
  elif patch_size == 70:
    model = Discriminator_70(num_channels_input, features_D)
  elif patch_size == 224:
    model = Discriminator_224(num_channels_input, features_D)
  else:
    raise NotImplementedError

  return model


class Discriminator_1(nn.Module):

  def __init__(self, num_channels_input, features_D):
    super().__init__()

    self.conv_1 = nn.Conv2d(in_channels=num_channels_input * 2, out_channels=features_D, kernel_size=1, stride=1)
    self.conv_2 = nn.Conv2d(in_channels=features_D, out_channels=features_D * 2, kernel_size=1, stride=1)
    self.bn_2 = nn.BatchNorm2d(features_D * 2)
    self.conv_out = nn.Conv2d(in_channels=features_D * 2, out_channels=1, kernel_size=1, stride=1)

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    self.sigmoid = nn.Sigmoid()

  # forward method
  def forward(self, input_image, image_masked):
    out = torch.cat([input_image, image_masked], 1)
    out = self.leaky_relu(self.conv_1(out))
    out = self.leaky_relu(self.bn_2(self.conv_2(out)))
    out = self.sigmoid(self.conv_out(out))
    return out


class Discriminator_16(nn.Module):

  def __init__(self, num_channels_input, features_D):
    super().__init__()

    self.conv_1 = nn.Conv2d(in_channels=num_channels_input * 2, out_channels=features_D, kernel_size=4, stride=2, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=features_D, out_channels=features_D * 2, kernel_size=4, stride=2, padding=1)
    self.bn_2 = nn.BatchNorm2d(features_D * 2)
    self.conv_out = nn.Conv2d(in_channels=features_D * 2, out_channels=1, kernel_size=4, stride=4, padding=0)

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    self.sigmoid = nn.Sigmoid()

  # forward method
  def forward(self, input_image, image_masked):
    out = torch.cat([input_image, image_masked], 1)
    out = self.leaky_relu(self.conv_1(out))
    out = self.leaky_relu(self.bn_2(self.conv_2(out)))
    out = self.sigmoid(self.conv_out(out))
    return out


class Discriminator_70(nn.Module):

  def __init__(self, num_channels_input, features_D):
    super().__init__()

    self.conv_1 = nn.Conv2d(in_channels=num_channels_input * 2, out_channels=features_D, kernel_size=4, stride=2, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=features_D, out_channels=features_D * 2, kernel_size=4, stride=2, padding=1)
    self.bn_2 = nn.BatchNorm2d(features_D * 2)

    self.conv_3 = nn.Conv2d(in_channels=features_D * 2, out_channels=features_D * 4, kernel_size=4, stride=2, padding=1)
    self.bn_3 = nn.BatchNorm2d(features_D * 4)

    self.conv_4 = nn.Conv2d(in_channels=features_D * 4, out_channels=features_D * 8, kernel_size=4, stride=2, padding=1)
    self.bn_4 = nn.BatchNorm2d(features_D * 8)

    self.conv_out = nn.Conv2d(in_channels=features_D * 8, out_channels=1, kernel_size=4, stride=4, padding=1)

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    self.sigmoid = nn.Sigmoid()

  # forward method
  def forward(self, input_image, image_masked):
    out = torch.cat([input_image, image_masked], 1)
    out = self.leaky_relu(self.conv_1(out))
    out = self.leaky_relu(self.bn_2(self.conv_2(out)))
    out = self.leaky_relu(self.bn_3(self.conv_3(out)))
    out = self.leaky_relu(self.bn_4(self.conv_4(out)))
    out = self.sigmoid(self.conv_out(out))
    return out


class Discriminator_224(nn.Module):

  def __init__(self, num_channels_input, features_D):
    super().__init__()

    self.conv_1 = nn.Conv2d(in_channels=num_channels_input * 2, out_channels=features_D, kernel_size=4, stride=2, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=features_D, out_channels=features_D * 2, kernel_size=4, stride=2, padding=1)
    self.bn_2 = nn.BatchNorm2d(features_D * 2)

    self.conv_3 = nn.Conv2d(in_channels=features_D * 2, out_channels=features_D * 4, kernel_size=4, stride=2, padding=1)
    self.bn_3 = nn.BatchNorm2d(features_D * 4)

    self.conv_4 = nn.Conv2d(in_channels=features_D * 4, out_channels=features_D * 8, kernel_size=4, stride=2, padding=1)
    self.bn_4 = nn.BatchNorm2d(features_D * 8)

    self.conv_5 = nn.Conv2d(in_channels=features_D * 8, out_channels=features_D * 8, kernel_size=4, stride=2, padding=1)
    self.bn_5 = nn.BatchNorm2d(features_D * 8)

    self.conv_6 = nn.Conv2d(in_channels=features_D * 8, out_channels=features_D * 8, kernel_size=4, stride=2, padding=1)
    self.bn_6 = nn.BatchNorm2d(features_D * 8)

    self.conv_out = nn.Conv2d(in_channels=features_D * 8, out_channels=1, kernel_size=4, stride=4, padding=1)

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    self.sigmoid = nn.Sigmoid()

  # forward method
  def forward(self, input_image, image_masked):
    out = torch.cat([input_image, image_masked], 1)
    out = self.leaky_relu(self.conv_1(out))
    out = self.leaky_relu(self.bn_2(self.conv_2(out)))
    out = self.leaky_relu(self.bn_3(self.conv_3(out)))
    out = self.leaky_relu(self.bn_4(self.conv_4(out)))
    out = self.leaky_relu(self.bn_5(self.conv_5(out)))
    out = self.leaky_relu(self.bn_6(self.conv_6(out)))
    out = self.sigmoid(self.conv_out(out))

    return out
