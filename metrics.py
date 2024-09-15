"""Implements different image metrics."""
import torch
from skimage.metrics import structural_similarity

def ncc(img1, img2):
  """Takes two image and compute the negative normalized cross correlation.
     Lower the value, better the alignment.
  """

  #  Convert the images into one-dimensional arrays (flatten) for easier calculation.
  img1_flat = img1.flatten()
  img2_flat = img2.flatten()

  # Calculate the average brightness (mean) of each image.
  img1_mean = img1_flat.mean()
  img2_mean = img2_flat.mean()
  
  # Subtract the average brightness from each pixel to focus on differences rather than absolute brightness.
  img1_zero_mean = img1_flat - img1_mean
  img2_zero_mean = img2_flat - img2_mean

  # Multiply the zero-mean versions of the images element-wise and sum the result. 
  # This shows how much the images vary together.
  numerator = torch.sum(img1_zero_mean * img2_zero_mean)

  # Compute the norms of the zero-mean images
  denominator = torch.sqrt(torch.sum(img1_zero_mean ** 2) * torch.sum(img2_zero_mean ** 2))

  # Prevent division by zero
  if denominator == 0:
      return 0

  # Compute the normalized cross-correlation
  ncc_value = numerator / denominator

  return -ncc_value


def mse(img1, img2): 
  """Takes two image and compute the mean squared error.
     Lower the value, better the alignment.
  """

  # Compute the squared differences between each pixel in the two images
  squared_diff = (img1 - img2) ** 2

  # Compute the sum of squared differences (SSD)
  ssd_value = torch.sum(squared_diff)

  # Compute mean squared error
  # numel returns the total number of elements in the tensor
  mse_value = ssd_value / img1.numel()

  return mse_value


def ssim(img1, img2):
  """Takes two image and compute the negative structural similarity.

  This function is given to you, nothing to do here.

  Please refer to the classic paper by Wang et al. of Image quality 
  assessment: from error visibility to structural similarity.
  """
  img1 = img1.numpy()
  img2 = img2.numpy()
  return -structural_similarity(img1, img2, data_range=img1.max() - img2.min())