from torchvision.models import inception_v3
import torch

device = 'cuda:0'

inception_model = inception_v3(pretrained=True)
inception_model.to(device)
inception_model = inception_model.eval()  # Evaluation mode

identity_layer = torch.nn.Identity()
inception_model.fc = identity_layer


def get_inception_features(real_images, fake_images):
  reals = inception_model(real_images.to(device)).detach().to('cpu')
  fakes = inception_model(fake_images.to(device)).detach().to('cpu')

  return reals, fakes
