import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

train_transform = transforms.Compose([
  transforms.ColorJitter(brightness=[1, 2], contrast=[0.5, 1]),
  transforms.RandomRotation(10),
  transforms.Resize((350, 350)),
  transforms.RandomCrop((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  transforms.ToPILImage()])

mask_transform = transforms.Compose([
  transforms.ColorJitter(brightness=[1, 2], saturation=[1, 2]),
  transforms.RandomRotation(60)])

eval_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  transforms.ToPILImage()])


class ArtNetDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_dir, mask_dir, split):
    # Initialize certain variables that will be used later.
    assert split in ['train', 'validation', 'test'], 'Error: Inserted incorrect set. Valid sets are case-sensitive = train or validation or test'
    self.dataset_dir = dataset_dir
    self.mask_sizes = [(32, 32), (64, 64), (128, 128)]
    self.mask_dir = mask_dir
    self.split = split

    # Initialize transformations that need to be applied on the dataset
    self.train_transform = train_transform
    self.mask_transform = mask_transform
    self.eval_transform = eval_transform

    self.toTensor = transforms.ToTensor()

    self.filenames = np.asarray(os.listdir(self.dataset_dir))

  def add_mask(self, img):
    masked_image = img.copy()
    mask_file = np.random.choice(os.listdir(self.mask_dir))
    mask_size = np.random.randint(len(self.mask_sizes), size=1)
    mask_path = os.path.join(self.mask_dir, mask_file)
    mask = Image.open(mask_path).resize(self.mask_sizes[mask_size[0]])
    mask = self.mask_transform(mask)
    # generate random coordinates and paste the mask on the real image
    random_x = self.random_coord(masked_image)
    random_y = self.random_coord(masked_image)
    masked_image.paste(mask, (random_x, random_y), mask)

    return masked_image

  def random_coord(self, image):
    return np.random.choice(range(0, np.asarray(image).shape[0] - int(np.asarray(image).shape[0] / 3)))

  # Modify methods to return data in the required format.
  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    current_file = self.filenames[idx]
    self.current_image = Image.open(os.path.join(self.dataset_dir, current_file))
    if self.split == 'train':
      real_image = self.train_transform(self.current_image)
    else:
      real_image = self.eval_transform(self.current_image)
    masked_image = self.add_mask(real_image)

    return self.toTensor(real_image), self.toTensor(masked_image)


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
      pipeline = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    elif set == 'test':
      test_dataset = ArtNetDataset(set_dir, mask_dir, set)
      pipeline = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
    else:  # validation case
      valid_dataset = ArtNetDataset(set_dir, mask_dir, set)
      pipeline = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    data_pipelines[set] = pipeline

  return data_pipelines
