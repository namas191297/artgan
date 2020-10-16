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

def get_discriminator_loss(image_real, image_masked, patch_size, variant, model_D, criterion_D, image_size, device):

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
      current_loss = criterion_D(output, label)

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
  output_path = 'output'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
