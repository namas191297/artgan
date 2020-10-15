import json
import logging
import torch

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
  if params.use_noise:
    noise_tensor = torch.randn((batch_size, num_channels_input, image_size, image_size), dtype=torch.float32).to(params.device)
  else:
    noise_tensor = None
  return noise_tensor

