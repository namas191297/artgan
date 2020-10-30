import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model.utils import create_output_folder, convert
import cv2
import random

train_transform = transforms.Compose([
  transforms.RandomRotation(10),
  transforms.Resize((256, 256)),
  transforms.RandomCrop((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

mask_transform = transforms.Compose([
  transforms.ColorJitter(brightness=[1, 2], saturation=[1, 2]),
  transforms.RandomRotation(60)])

eval_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ArtNetDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_dir, mask_dir, split):
    # Initialize certain variables that will be used later.
    assert split in ['train', 'validation',
                     'test'], 'Error: Inserted incorrect set. Valid sets are case-sensitive = train or validation or test'
    self.dataset_dir = dataset_dir
    self.heavy_mask_sizes = [(64, 64), (96, 96), (128, 128)]
    self.light_mask_sizes = [(160, 160), (180, 180), (200, 200)]
    self.mask_dir = mask_dir
    self.split = split

    # Initialize transformations that need to be applied on the dataset
    self.train_transform = train_transform
    # No need for mask transform at this stage
    self.mask_transform = mask_transform

    self.eval_transform = eval_transform

    self.toTensor = transforms.ToTensor()

    self.filenames = np.asarray(os.listdir(self.dataset_dir))
    create_output_folder()

  def add_mask(self, img):
    masked_image = np.array(img.cpu().detach().permute(1, 2, 0).numpy())
    height, width, num_channels = masked_image.shape
    mask_type = np.random.choice(os.listdir(self.mask_dir))
    mask_type_dir = os.path.join(self.mask_dir, mask_type)
    if mask_type == 'heavy':
      mask_sizes = self.heavy_mask_sizes
      mask_colour = np.random.choice(os.listdir(mask_type_dir))
      mask_type_dir = os.path.join(mask_type_dir, mask_colour)
    elif mask_type == 'light':
      mask_sizes = self.light_mask_sizes
      mask_colour = 'black'
    else:
      raise TypeError

    mask_size = np.random.randint(len(mask_sizes), size=1)

    mask_file = np.random.choice(os.listdir(mask_type_dir))

    mask_path = os.path.join(mask_type_dir, mask_file)

    mask_height = mask_sizes[mask_size[0]][0]
    mask_width = mask_sizes[mask_size[0]][1]
    dim = mask_sizes[mask_size[0]]

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    alpha = mask[:, :, 3]  # extract it
    if mask_colour == 'white':
      mask = alpha
    else:
      mask = 255 - alpha  # invert b/w

    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

    rand_x = random.randint(0, height - mask_height - 1)
    rand_y = random.randint(0, height - mask_height - 1)

    range_x_start = rand_x
    range_x_end = rand_x + mask_height

    range_y_start = rand_y
    range_y_end = rand_y + mask_width

    w, h = mask.shape
    mask_rgb = np.empty((w, h, 3), dtype=np.uint8)
    mask_rgb[:, :, 2] = mask_rgb[:, :, 1] = mask_rgb[:, :, 0] = mask
    if mask_colour == 'black':
      masked_image[range_x_start:range_x_end, range_y_start:range_y_end, :] = np.where(mask_rgb > 180, masked_image[
                                                                                                       range_x_start:range_x_end,
                                                                                                       range_y_start:range_y_end,
                                                                                                       :],
                                                                                       convert(mask_rgb, -1, 1,
                                                                                               np.float32))
    else:
      masked_image[range_x_start:range_x_end, range_y_start:range_y_end, :] = np.where(mask_rgb < 180, masked_image[
                                                                                                       range_x_start:range_x_end,
                                                                                                       range_y_start:range_y_end,
                                                                                                       :],
                                                                                       convert(mask_rgb, -1, 1,
                                                                                               np.float32))

    masked_image = transforms.ToTensor()(masked_image)

    return masked_image

  def random_coord(self, image):
    return np.random.choice(range(0, np.asarray(image).shape[0] - int(np.asarray(image).shape[0] / 3)))

  # Modify methods to return data in the required format.
  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    current_file = self.filenames[idx]
    self.current_image = Image.open(os.path.join(self.dataset_dir, current_file)).convert('RGB')
    if self.split == 'train':
      real_image = self.train_transform(self.current_image)
    else:
      real_image = self.eval_transform(self.current_image)
    masked_image = self.add_mask(real_image)

    return real_image, masked_image


def fetch_pipeline(types, data_dir, params):
  """
  Method that returns the input pipelines via DataLoader classes, depending on the values of the set types to be passed.

  :param types: (list), referring to the types of sets that are handled by the dataLoader classes.
                (Accepts only `train`, `validation`, and `test`)
  :param data_dir: (String), the path where the dataset lies
  :param params: (Params) hyper-parameters
  :return: (list), data pipelines corresponding to individual sets based on the `types` values
  """
  data_pipelines = {}
  mask_dir = os.path.join(data_dir, 'masks/')
  for set in types:
    set_dir = os.path.join(data_dir, set)
    if set == 'train':
      train_dataset = ArtNetDataset(set_dir, mask_dir, set)
      pipeline = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers,
                            pin_memory=params.cuda)
    elif set == 'test':
      test_dataset = ArtNetDataset(set_dir, mask_dir, set)
      pipeline = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers,
                            pin_memory=params.cuda)
    else:  # validation case
      valid_dataset = ArtNetDataset(set_dir, mask_dir, set)
      pipeline = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers,
                            pin_memory=params.cuda)

    data_pipelines[set] = pipeline

  return data_pipelines
