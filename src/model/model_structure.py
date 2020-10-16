import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from model.discriminator import *
from model.generator import DenseUNet
from torch.autograd import Variable
from model.utils import get_random_noise_tensor


def model_fn(params):
  """
  Model function defining the graph operations

  :param params: (Params), The hyper-parameters inside json file for the model's operations
  :return: (dict), A dictionary that contains the graph operations or nodes needed for training/validation/testing
  """

  # model_D = Discriminator_256(params.num_channels, params.features_D).to(params.device)
  noise_tensor = get_random_noise_tensor(params.batch_size, params.num_channels, params.image_size, params)
  model_D = get_discriminator(patch_size=params.patch_size, num_channels_input=params.num_channels, features_D=params.features_D).to(params.device)
  model_G = DenseUNet(params.num_channels, params.features_G, num_dense_blocks=params.num_dense_blocks, noise_tensor=noise_tensor).to(params.device)

  criterion_G = nn.BCELoss()
  criterion_D = nn.BCELoss()

  L1_criterion_G = nn.L1Loss()

  optimizer_D = optim.Adam(model_D.parameters(), lr=params.lr_D, betas=(0.5, 0.999))
  optimizer_G = optim.Adam(model_G.parameters(), lr=params.lr_G, betas=(0.5, 0.999))

  models = {'model_D': model_D, 'model_G': model_G}
  losses = {'criterion_G': criterion_G, 'criterion_D': criterion_D, 'L1_criterion_G': L1_criterion_G}
  optimizers = {'optimizer_D': optimizer_D, 'optimizer_G': optimizer_G}

  metrics = {'MSE': nn.MSELoss(), 'SSIM': SSIM(), 'per_pixel_accuracy': per_pixel_accuracy}

  model_spec = {'models': models,
                'losses': losses,
                'optimizers': optimizers,
                'metrics': metrics}

  return model_spec


def per_pixel_accuracy(real, generated, params):
  total_pixels = 0
  total_pixels += real.nelement()

  distance = real - generated
  threshold = torch.tensor(params.threshold_pp_acc)

  correct_pixels = torch.sum(torch.where(distance < threshold, torch.tensor(1), torch.tensor(0))).item()
  pp_acc = correct_pixels / total_pixels
  return pp_acc


def gaussian(window_size, sigma):
  gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
  return gauss / gauss.sum()


def create_window(window_size, channel):
  _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
  return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
  mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
  mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1 * mu2

  sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
  sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
  sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

  C1 = 0.01 ** 2
  C2 = 0.03 ** 2

  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

  if size_average:
    return ssim_map.mean()
  else:
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
  def __init__(self, window_size=11, channels=3, size_average=True):
    super().__init__()
    self.window_size = window_size
    self.size_average = size_average
    self.channel = channels
    self.window = create_window(window_size, self.channel)

  def forward(self, img1, img2):
    (_, channel, _, _) = img1.size()

    if channel == self.channel and self.window.data.type() == img1.data.type():
      window = self.window
    else:
      window = create_window(self.window_size, channel)

      if img1.is_cuda:
        window = window.cuda(img1.get_device())
      window = window.type_as(img1)

      self.window = window
      self.channel = channel

    return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# def ssim(img1, img2, window_size=11, size_average=True):
#   (_, channel, _, _) = img1.size()
#   window = create_window(window_size, channel)
#
#   if img1.is_cuda:
#     window = window.cuda(img1.get_device())
#   window = window.type_as(img1)
#
#   return _ssim(img1, img2, window, window_size, channel, size_average)
#
#
# def batch_ssim(real, predicted):
#   loss = []
#   _, channel, h, w = real.size()
#   for i in range(real.shape[0]):
#     loss.append(ssim(real[i].reshape(1, channel, h, w), predicted[i].reshape(1, channel, h, w)))
#   return sum(loss) / len(loss)
#
#
# def psnr(original, predicted):
#   pass


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

def get_discriminator_outputs(output_D_real, label_D_real):
  current_batch_size = image_batch_dummy.shape[0]
  start_index = 0
  end_index = image_size - patch_size + 1
  stride = 6
  num_steps_frac = end_index / stride
  num_steps = int(num_steps_frac)

  # handle last stride
  if num_steps_frac - num_steps > 1e-9:
    end_index += stride
  loss_D_running = None

  for index_x in range(start_index, end_index, stride):
    x_start = index_x
    x_end = x_start + patch_size

    if x_end > image_size:
      x_start = image_size - patch_size
      x_end = image_size
    for index_y in range(start_index, end_index, stride):

      y_start = index_y
      y_end = y_start + patch_size
      if y_end > image_size:
        y_start = image_size - patch_size
        y_end = image_size

      current_crop_real = output_D_real[:, :, x_start:x_end, y_start:y_end]
      current_crop_label = label_D_real[:, :, x_start:x_end, y_start:y_end]

      output = model_D(current_crop_real, current_crop_label)
      current_loss = criterion_D(output_D_real, label_D_real)

      if loss_D_running is None:
        loss_D_running = current_loss
      loss_D_running += current_loss
  return loss_D_running