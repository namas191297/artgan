import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# training and evaluation data augmentations classes
# train_transformer = transforms.Compose([
#   transforms.Resize((224, 224)),  # resize the image to 224x224 (remove if images are already 224x224)
#   transforms.RandomHorizontalFlip(),  # randomly flip image horizontally (defaults to 0.5 probability)
#   transforms.RandomCrop((224, 224), padding=4, fill=0, padding_mode='constant'),  # zero-pad 4 pixels and randomly crop to 224x224
#   transforms.ToTensor(),  # transform it into a torch tensor, it normalises to [0,1]
#   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Use normalisation based on the standard score [-1,1]
#
# # loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#   transforms.Resize((224, 224)),  # resize the image to 224x224 (remove if images are already 224x224)
#   transforms.ToTensor(),  # transform it into a torch tensor, it normalises to [0,1]
#   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Use normalisation based on the standard score [-1,1]


class ArtNetDataset(torch.utils.data.Dataset):

  def __init__(self, directory, transformations=True):
    # Initialize certain variables that will be used later.
    self.directory = directory
    self.damage_sizes = [(32, 32), (64, 64), (128, 128)]
    self.damage_dir = 'original_damages/'
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize transformations that need to be applied on the dataset
    self.img_transform = transforms.Compose([
      transforms.ColorJitter(brightness=[1, 2], contrast=[0.5, 1]),
      transforms.RandomRotation(10),
      transforms.Resize((350, 350)),
      transforms.RandomCrop((256, 256))])

    self.dmg_transform = transforms.Compose([
      transforms.ColorJitter(brightness=[1, 2], saturation=[1, 2]),
      transforms.RandomRotation(60)])

    self.toTensor = transforms.ToTensor()

    self.filenames = np.asarray(os.listdir(self.directory))

  def add_mask(self, img):
    image = img.copy()
    damage_dir = 'original_damages/'
    damage_sizes = [100, 128, 156, 200]
    damage = np.random.choice(os.listdir(damage_dir))
    damage_size = np.random.choice(damage_sizes)
    damage = damage_dir + damage
    damage = Image.open(damage).resize((damage_size, damage_size))
    damage = self.dmg_transform(damage)
    random_x = np.random.choice(range(0, np.asarray(image).shape[0] - int(np.asarray(image).shape[0] / 3)))
    random_y = np.random.choice(range(0, np.asarray(image).shape[1] - int(np.asarray(image).shape[1] / 3)))
    image.paste(damage, (random_x, random_y), damage)
    return image

  # Modify methods to return data in the required format.
  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    f = self.filenames[idx]
    self.X = Image.open(self.directory + f)
    originalX = self.img_transform(self.X)
    maskedX = self.add_mask(originalX)
    return self.toTensor(originalX), self.toTensor(maskedX)


def fetch_pipeline(types, data_dir, params):
  """
  Method that returns the input pipelines via DataLoader classes, depending on the values of the set types to be passed.

  :param types: (list), referring to the types of sets that are handled by the dataLoader classes.
                (Accepts only `training`, `validation`, and `testing`)
  :param data_dir: (String), the path where the dataset lies
  :param params: (Params) hyper-parameters
  :return: (list), data pipelines corresponding to individual sets based on the `types` values
  """
  data_pipelines = {}
  for set in types:
    path_directory = os.path.join(data_dir, set)
    if set == 'training':
      train_dataset = ArtNetDataset(path_directory)
      pipeline = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    elif set == 'testing':
      test_dataset = ArtNetDataset(path_directory)
      pipeline = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
    else:  # testing case
      valid_dataset = ArtNetDataset(path_directory)
      pipeline = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    data_pipelines[set] = pipeline

  return data_pipelines
