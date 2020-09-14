import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# TODO probably change those as well
# training and evaluation data augmentations classes
train_transformer = transforms.Compose([
  transforms.Resize((224, 224)),  # resize the image to 224x224 (remove if images are already 224x224)
  transforms.RandomHorizontalFlip(),  # randomly flip image horizontally (defaults to 0.5 probability)
  transforms.RandomCrop((224, 224), padding=4, fill=0, padding_mode='constant'),  # zero-pad 4 pixels and randomly crop to 224x224
  transforms.ToTensor(),  # transform it into a torch tensor, it normalises to [0,1]
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Use normalisation based on the standard score [-1,1]

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
  transforms.Resize((224, 224)),  # resize the image to 224x224 (remove if images are already 224x224)
  transforms.ToTensor(),  # transform it into a torch tensor, it normalises to [0,1]
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # Use normalisation based on the standard score [-1,1]


class OurAwesomeDataset(Dataset):
  """
  Custom PyTorch dataset to define the Cats and Dogs problem and extends the PyTorch default Dataset class with the appropriate overridden methods.
  """

  def __init__(self, data_dir, transformations=None):
    """
    :param data_dir: (string) directory containing the dataset
    :param transformations: (torchvision.transforms) the transformations to be applied on the image
    """
    self.filenames = os.listdir(data_dir)
    self.filenames = [os.path.join(data_dir, f) for f in self.filenames]

    # set the labels to be 0 for cats and 1 for dogs
    self.labels =  # FIXME
    # expand the labels to the form (?, 1)
    self.labels = self.labels[:, None]
    self.transformations = transformations

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, item):
    image = Image.open(self.filenames[item])
    if self.transformations is not None:
      image = self.transformations(image)
    return image, self.labels[item]


def fetch_pipeline(types, data_dir, params):
  """
  Method that returns the input pipelines via DataLoader classes, depending on the values of the set types to be passed.

  :param types: (list), referring to the types of sets that are handled by the dataLoader classes.
                (Accepts only `training`, `validation`, and `testing`)
  :param data_dir: (String), the path where the dataset lies
  :param params: (Params) hyper-parameters
  :return: (list), data pipelines corresponding to individual sets based on the `types` values
  """
  # TODO we will use a simpler methodology in terms of splits since we will have our datasets ready in advance
  data_pipelines = {}
  for set in types:
    if set == 'testing':
      test_dataset = OurAwesomeDataset(set_path, eval_transformer)
      pipeline = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    else:  # training or validation set cases
      # FIXME set path
      dataset = OurAwesomeDataset(set_path)
      train_dataset, valid_dataset = random_split(dataset, lengths=(18000, 2000), transformations=[train_transformer, eval_transformer],
                                                  generator=torch.default_generator)
      if set == 'training':
        pipeline = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
      else:  # validation set case
        pipeline = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    data_pipelines[set] = pipeline

  return data_pipelines
