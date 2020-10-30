import pickle
import numpy as np
import torch
from scipy import linalg

with open('frehcet.pkl', 'rb') as f:
  frechet_real, frechet_fake = pickle.load(f)


def cov(x):
  return np.cov(x, rowvar=False)


def matsqrt(img):
  data = img.cpu().detach().numpy()
  data = linalg.sqrtm(data).real
  return torch.Tensor(data, device='cpu')


def frechet_distance(mean_real, mean_fake, sigma_real, sigma_fake):
  mean_real = torch.tensor(mean_real)
  mean_fake = torch.tensor(mean_fake)
  sigma_real = torch.tensor(sigma_real)
  sigma_fake = torch.tensor(sigma_fake)
  return (torch.norm(mean_real - mean_fake) ** 2) + torch.trace(sigma_real + sigma_fake - (2 * matsqrt(sigma_real @ sigma_fake)))


reals_list = []
reals_list.append([i[0] for i in frechet_real])
reals = np.array(reals_list).reshape(36, -1)

fakes_list = []
fakes_list.append([i[0] for i in frechet_fake])
fakes = np.array(fakes_list).reshape(36, -1)

real_cov = cov(reals)
fake_cov = cov(fakes)

real_means = np.mean(reals, axis=0)
fake_means = np.mean(fakes, axis=0)

distance = frechet_distance(real_means, fake_means, real_cov, fake_cov)

print(f'Frechet Inception Distance: {distance}')
