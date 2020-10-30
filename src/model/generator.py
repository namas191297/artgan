import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.cat((F.relu(x), F.relu(-x)), 1)


class CResBlock(nn.Module):

  def __init__(self, num_channels_input, features_G, dropout_ratio=0.5, leaky_slope=0.2, use_crelu=False, use_avgpool=False):
    super(CResBlock, self).__init__()
    # 224 x 224 x 6 -> 112 x 112 x 16
    self.conv_down_1 = nn.Conv2d(in_channels=num_channels_input, out_channels=features_G, kernel_size=3, padding=1, stride=2)
    self.bn_down_1 = nn.BatchNorm2d(features_G)

    if use_crelu:
      features_G *= 2

    self.conv_down_2 = nn.Conv2d(in_channels=features_G, out_channels=features_G * 2, kernel_size=3, padding=1, stride=2)
    self.bn_down_2 = nn.BatchNorm2d(features_G * 2)

    if use_crelu:
      features_G *= 2

    self.conv_down_3 = nn.Conv2d(in_channels=features_G * 2, out_channels=features_G * 4, kernel_size=3, padding=1, stride=2)
    self.bn_down_3 = nn.BatchNorm2d(features_G * 4)

    if use_crelu:
      features_G *= 2

    self.bottleneck_1 = nn.Conv2d(in_channels=features_G * 4, out_channels=features_G, kernel_size=1)

    self.conv_up_3 = nn.ConvTranspose2d(in_channels=(features_G * 4 + features_G), out_channels=features_G * 4, kernel_size=2, stride=2)
    self.bn_up_3 = nn.BatchNorm2d(features_G * 4)

    if use_crelu:
      self.bottleneck_2 = nn.Conv2d(in_channels=(features_G * 4 + features_G), out_channels=features_G * 4, kernel_size=1)
    else:
      self.bottleneck_2 = nn.Conv2d(in_channels=features_G * 6, out_channels=features_G * 4, kernel_size=1)

    self.conv_up_2 = nn.ConvTranspose2d(in_channels=features_G * 4, out_channels=features_G * 2, kernel_size=2, stride=2)
    self.bn_up_2 = nn.BatchNorm2d(features_G * 2)

    if use_crelu:
      self.conv_up_1 = nn.ConvTranspose2d(in_channels=(features_G * 2 + features_G // 4), out_channels=features_G, kernel_size=2, stride=2)
    else:
      self.conv_up_1 = nn.ConvTranspose2d(in_channels=features_G * 3, out_channels=features_G, kernel_size=2, stride=2)
    self.bn_up_1 = nn.BatchNorm2d(features_G)

    self.bottleneck_3 = nn.Conv2d(in_channels=features_G, out_channels=num_channels_input, kernel_size=1)

    self.out = nn.ConvTranspose2d(in_channels=features_G * 2, out_channels=features_G, kernel_size=2, stride=2)

    self.relu = nn.ReLU()

    if use_crelu:
      self.alt_relu = CReLU()
    else:
      self.alt_relu = nn.LeakyReLU(negative_slope=leaky_slope)

    if use_avgpool:
      self.pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
      self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    # Residual layers
    if use_crelu:
      self.res_bottleneck_1 = nn.Conv2d(in_channels=(features_G * 2 + features_G // 2), out_channels=features_G // 4, kernel_size=1)
      self.res_bottleneck_2 = nn.Conv2d(in_channels=features_G * 6, out_channels=features_G, kernel_size=1)
    else:
      self.res_bottleneck_1 = nn.Conv2d(in_channels=features_G * 4, out_channels=features_G, kernel_size=1)
      self.res_bottleneck_2 = nn.Conv2d(in_channels=features_G * 8, out_channels=features_G * 2, kernel_size=1)

    self.res_bottleneck_3 = nn.Conv2d(in_channels=features_G * 9, out_channels=features_G * 4, kernel_size=1)

    self.dropout = nn.Dropout2d(p=dropout_ratio)

  def forward(self, input, res_layers=None):

    if res_layers is not None:
      res_up_3, res_up_2, res_up_1 = res_layers

      # 112 x 112 x 16
      down_1 = self.alt_relu(self.bn_down_1(self.conv_down_1(input)))
      # 112 x 112 x (16 + 48)
      down_1 = torch.cat((down_1, res_up_1), 1)
      # 112 x 112 x 16
      down_1 = self.res_bottleneck_1(down_1)

      # 56 x 56 x 32
      down_2 = self.alt_relu(self.bn_down_2(self.conv_down_2(down_1)))
      # 56 x 56 x (32 + 96)
      down_2 = torch.cat((down_2, res_up_2), 1)
      # 56 x 56 x 32
      down_2 = self.res_bottleneck_2(down_2)

      # 28 x 28 x 64
      down_3 = self.alt_relu(self.bn_down_3(self.conv_down_3(down_2)))
      # 28 x 28 x (64 + 80)
      down_3 = torch.cat((down_3, res_up_3), 1)
      # 28 x 28 x 64
      down_3 = self.res_bottleneck_3(down_3)

    else:
      # 112 x 112 x 16
      down_1 = self.alt_relu(self.bn_down_1(self.conv_down_1(input)))
      # 56 x 56 x 32
      down_2 = self.alt_relu(self.bn_down_2(self.conv_down_2(down_1)))
      # 28 x 28 x 64
      down_3 = self.alt_relu(self.bn_down_3(self.conv_down_3(down_2)))

    # 28 x 28 x 16
    bottleneck = self.bottleneck_1(down_3)
    # 28 x 28 x (64 + 16)
    up_3_concat = torch.cat((down_3, bottleneck), 1)
    # 56 x 56 x 64
    up_2 = self.relu(self.dropout(self.bn_up_3(self.conv_up_3(up_3_concat))))
    # 56 x 56 x (64 + 32)
    up_2_concat = torch.cat((down_2, up_2), 1)
    # 56 x 56 x 64
    up_2 = self.bottleneck_2(up_2_concat)
    # 112 x 112 x 32
    up_1 = self.relu(self.dropout(self.bn_up_2(self.conv_up_2(up_2))))
    # 112 x 112 x (32 + 16)
    up_1_concat = torch.cat((down_1, up_1), 1)
    # 224 x 224 x 16
    up_1 = self.relu(self.dropout(self.bn_up_1(self.conv_up_1(up_1_concat))))
    # 224 x 224 x 6
    out = self.bottleneck_3(up_1)

    layers = [up_3_concat, up_2_concat, up_1_concat]
    return out, layers


class CResUNet(nn.Module):
  def __init__(self, num_channels_input, features_G, use_crelu, use_avgpool, num_dense_blocks=4, noise_tensor=None):
    super(CResUNet, self).__init__()
    self.num_dense_blocks = num_dense_blocks
    self.noise_tensor = noise_tensor
    self.features_G = features_G

    if self.noise_tensor is not None:
      self.num_channels_input = num_channels_input * 2
    else:
      self.num_channels_input = num_channels_input

    self.num_channels_output = num_channels_input

    self.block_1 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)
    self.block_2 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)
    self.block_3 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)
    self.block_4 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)
    self.block_5 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)
    self.block_6 = CResBlock(self.num_channels_input, self.features_G, use_crelu=use_crelu, use_avgpool=use_avgpool)

    self.out_final = nn.Conv2d(in_channels=self.num_dense_blocks * self.num_channels_input,
                               out_channels=self.num_channels_output, kernel_size=1, stride=1)

    self.tanh = nn.Tanh()

  def forward(self, image_masked, noise_tensor=None):

    if self.noise_tensor is not None:
      input_image = torch.cat((image_masked, noise_tensor), 1)
    else:
      input_image = image_masked

    if self.num_dense_blocks == 6:
      # 224 x 224 x 3
      out_1, layers_1 = self.block_1(input_image)
      out_2, layers_2 = self.block_2(out_1, layers_1)
      out_3, layers_3 = self.block_3(out_2, layers_2)
      out_4, layers_4 = self.block_4(out_3, layers_3)
      out_5, layers_5 = self.block_5(out_4, layers_4)
      out_6, _ = self.block_6(out_5, layers_5)

      out_concat_1 = torch.cat((out_1, out_2), 1)
      out_concat_2 = torch.cat((out_concat_1, out_3), 1)
      out_concat_3 = torch.cat((out_concat_2, out_4), 1)
      out_concat_4 = torch.cat((out_concat_3, out_5), 1)
      out_concat_5 = torch.cat((out_concat_4, out_6), 1)

      out_final = self.out_final(out_concat_5)

    elif self.num_dense_blocks == 4:
      # 224 x 224 x 3
      out_1, layers_1 = self.block_1(input_image)
      out_2, layers_2 = self.block_2(out_1, layers_1)
      out_3, layers_3 = self.block_3(out_2, layers_2)
      out_4, layers_4 = self.block_4(out_3, layers_3)

      out_concat_1 = torch.cat((out_1, out_2), 1)
      out_concat_2 = torch.cat((out_concat_1, out_3), 1)
      out_concat_3 = torch.cat((out_concat_2, out_4), 1)

      out_final = self.out_final(out_concat_3)

    else:
      raise NotImplementedError

    return self.tanh(out_final)
