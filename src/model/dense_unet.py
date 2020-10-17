import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):

  def __init__(self, num_channels_input, num_hidden_channels):
    super().__init__()

    # 224 x 224 x 6 -> 112 x 112 x 16
    self.down_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.down_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.down_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.bottleneck_1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)

    self.up_3 = nn.ConvTranspose2d(in_channels=80, out_channels=64, kernel_size=2, stride=2)
    self.bottleneck_2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)

    self.up_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

    self.up_1 = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=2, stride=2)
    self.bottleneck_3 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=1)

    self.out = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)

    self.relu = nn.ReLU()
    self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    # Residual layers
    self.res_bottleneck_1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)
    self.res_bottleneck_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
    self.res_bottleneck_3 = nn.Conv2d(in_channels=144, out_channels=64, kernel_size=1)

  def forward(self, input, res_layers=None):

    if res_layers is not None:

      res_up_3, res_up_2, res_up_1 = res_layers

      # 112 x 112 x 16
      down_1 = self.maxpool2d(self.relu(self.bn1(self.down_1(input))))
      # 112 x 112 x (16 + 48)
      down_1 = torch.cat((down_1, res_up_1), 1)
      # 112 x 112 x 16
      down_1 = self.res_bottleneck_1(down_1)

      # 56 x 56 x 32
      down_2 = self.maxpool2d(self.relu(self.bn2(self.down_2(down_1))))
      # 56 x 56 x (32 + 96)
      down_2 = torch.cat((down_2, res_up_2), 1)
      # 56 x 56 x 32
      down_2 = self.res_bottleneck_2(down_2)

      # 28 x 28 x 64
      down_3 = self.maxpool2d(self.relu(self.bn3(self.down_3(down_2))))
      # 28 x 28 x (64 + 80)
      down_3 = torch.cat((down_3, res_up_3), 1)
      # 28 x 28 x 64
      down_3 = self.res_bottleneck_3(down_3)


    else:
      # 112 x 112 x 16
      down_1 = self.maxpool2d(self.relu(self.down_1(input)))

      # 56 x 56 x 32
      down_2 = self.maxpool2d(self.relu(self.down_2(down_1)))

      # 28 x 28 x 64
      down_3 = self.maxpool2d(self.relu(self.down_3(down_2)))

    # 28 x 28 x 16
    bottleneck = self.bottleneck_1(down_3)

    # 28 x 28 x (64 + 16)
    up_3_concat = torch.cat((down_3, bottleneck), 1)
    ###################################################

    # 56 x 56 x 64
    up_2 = self.up_3(up_3_concat)

    # 56 x 56 x (64 + 32)
    up_2_concat = torch.cat((down_2, up_2), 1)

    # 56 x 56 x 64
    up_2 = self.bottleneck_2(up_2_concat)

    # 112 x 112 x 32
    up_1 = self.up_2(up_2)

    # 112 x 112 x (32 + 16)
    up_1_concat = torch.cat((down_1, up_1), 1)

    # 224 x 224 x 24
    up_1 = self.up_1(up_1_concat)

    # 224 x 224 x 3
    out = self.bottleneck_3(up_1)

    layers = [up_3_concat, up_2_concat, up_1_concat]

    return out, layers


class DenseUNet(nn.Module):

  def __init__(self, num_channels_input, num_hidden_channels):
    super().__init__()
    num_dense_blocks = 6
    self.block_1 = DenseBlock(num_channels_input, num_hidden_channels)
    self.block_2 = DenseBlock(num_channels_input, num_hidden_channels)
    self.block_3 = DenseBlock(num_channels_input, num_hidden_channels)
    self.block_4 = DenseBlock(num_channels_input, num_hidden_channels)
    self.block_5 = DenseBlock(num_channels_input, num_hidden_channels)
    self.block_6 = DenseBlock(num_channels_input, num_hidden_channels)

    self.out_final = nn.Conv2d(in_channels=num_dense_blocks * 3, out_channels=3, kernel_size=1, stride=1)

  def forward(self, image_masked):
    # 224 x 224 x 3
    out_1, layers_1 = self.block_1(image_masked)
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

    return out_final
