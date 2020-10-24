import json
import logging
import torch
import os


class Params:
  """
  Class that loads hyper-parameters from a json file
  """

  def __init__(self, json_path):
    self.update(json_path)

  def save(self, json_path):
    """
    Saves parameters to json file

    :param json_path: the actual json path
    """
    with open(json_path, 'w') as f:
      json.dump(self.__dict__, f, indent=4)

  def update(self, json_path):
    """
    Loads parameters from json file

    :param json_path: the actual json path
    """
    with open(json_path, 'r') as f:
      params = json.load(f)
      self.__dict__.update(params)

  @property
  def dict(self):
    """
    Gives dict-like access to Params instance by `params.dict['learning_rate']`

    :return: Dictionary
    """
    return self.__dict__


def set_logger(log_path):
  """
  Sets the logger to log info in terminal and file `log_path`

  :param log_path: The log path directory
  """
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s:'))
    logger.addHandler(file_handler)

    # Logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s:'))
    logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
  """
  Saves dictionary of floats to json file

  :param d: dictionary of float-castable values
  :param json_path: path to json file
  """
  with open(json_path, 'w') as f:
    # Require conversion of the values to float. It doesn't accept np.array or np.float
    d = {k: float(v) for k, v in d.items()}
    json.dump(d, f, indent=4)


def get_random_noise_tensor(batch_size, num_channels_input, image_size, params):
  noise_tensor = None
  if params.use_noise:
    noise_tensor = torch.randn((batch_size, num_channels_input, image_size, image_size), dtype=torch.float32).to(params.device)

  return noise_tensor

def convert(source, min_value, max_value, type):
  smin = source.min()
  smax = source.max()

  a = (max_value - min_value) / (smax - smin)
  b = max_value - a * smax
  new_img = (a * source + b).astype(type)

  return new_img

def get_discriminator_loss_conv(image_real, image_masked, patch_size, variant, model_D, criterion_D, image_size, device):
  batch_size = image_real.shape[0]

  if patch_size == 224:
    out_shape = 1
  elif patch_size == 70:
    out_shape = 4
  elif patch_size == 16:
    out_shape = 14
  elif patch_size == 1:
    out_shape = 224

  label = (torch.ones(batch_size, 1, out_shape, out_shape)).to(device)
  if variant is 'real_D':
    label *= 0.9
  elif variant is 'fake_D':
    label *= 0.1
  elif variant is 'fake_G':
    label *= 1

  output = model_D(image_real, image_masked)
  loss_D = criterion_D(output, label)

  confidence_D = output.mean().item()

  return loss_D, confidence_D

def get_discriminator_loss_strided(image_real, image_masked, patch_size, variant, model_D, criterion_D, image_size, device):
  batch_size = image_real.shape[0]
  start_index = 0
  end_index = image_size - patch_size + 1
  stride = patch_size // 2
  if stride < 1:
    stride = 1
  num_steps_frac = end_index / stride
  num_steps = int(num_steps_frac)

  # handle last stride
  if num_steps_frac - num_steps > 1e-9:
    end_index += stride
  loss_D_running = None
  confidence_D_running = None
  num_patches = 0

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

      current_crop_image = image_real[:, :, x_start:x_end, y_start:y_end]
      current_crop_masked = image_masked[:, :, x_start:x_end, y_start:y_end]

      if variant is 'real_D':
        label = (torch.ones(batch_size) * 0.9).to(device)
      elif variant is 'fake_D':
        label = (torch.ones(batch_size) * 0.1).to(device)
      elif variant is 'fake_G':
        label = torch.ones(batch_size).to(device)

      output = model_D(current_crop_image, current_crop_masked)
      current_loss = criterion_D(output.squeeze(), label)

      if loss_D_running is None:
        loss_D_running = current_loss
        confidence_D_running = output.mean().item()
      else:
        loss_D_running += current_loss
        confidence_D_running += output.mean().item()
      num_patches += 1
    loss_D_running /= num_patches
    confidence_D_running /= num_patches

  return loss_D_running, confidence_D_running


def create_output_folder():
  output_path = 'outputs'
  if not os.path.exists(output_path):
    os.makedirs(output_path)

def get_discriminator_loss(image_real, image_masked, model_D, model_G, criterion_D, batch_size, params, noise_tensor=None):
  loss_D_real, confidence_D = get_discriminator_loss_conv(image_real, image_masked, params.patch_size, 'real_D',
                                                          model_D,
                                                          criterion_D,
                                                          params.image_size, params.device)

  # fake image
  if noise_tensor is None:
    noise_tensor = get_random_noise_tensor(batch_size, params.num_channels, params.image_size, params)
  fake = model_G(image_masked, noise_tensor)  # generate fakes, given masked images

  loss_D_fake, _ = get_discriminator_loss_conv(fake.detach(), image_masked, params.patch_size, 'fake_D',
                                               model_D,
                                               criterion_D, params.image_size,
                                               params.device)

  # aggregate discriminator loss
  loss_D = (loss_D_real + (loss_D_fake)) * params.loss_D_factor  # multiplied by 0.5 to slow down discriminator's learning

  return fake, loss_D, model_D, model_G, confidence_D