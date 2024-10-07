"""Implements Misc helper functions."""

import torch
import torch.nn.functional as F
import numpy as np

def custom_shifts(input, shifts, dims=None, padding='circular'):
  """Shifts the input tensor by the specified shifts along the specified dimensions.
     Supports circular and zero padding.
  """
  ret = torch.roll(input, shifts, dims)
  if padding == 'zero':
    ret[:shifts[0], :shifts[1]] = 0
  return ret 


def detect_local_maxima(scale_space, image, n):

  maxima = torch.zeros_like(image)
  sigma_list = []

  for i in range(1, n-1):
        prev, _ = scale_space[i-1]
        current, current_sigma = scale_space[i]
        next, _ = scale_space[i+1]
        
        # Resize all to the size of 'current'
        prev = F.interpolate(prev.unsqueeze(0).unsqueeze(0), size=current.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        next = F.interpolate(next.unsqueeze(0).unsqueeze(0), size=current.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        local_max = (current > prev) & (current > next)
        local_max &= (current > torch.roll(current, 1, 0)) & (current > torch.roll(current, -1, 0))
        local_max &= (current > torch.roll(current, 1, 1)) & (current > torch.roll(current, -1, 1))
        
        # Resize to original image size
        maxima_level = F.interpolate(local_max.float().unsqueeze(0).unsqueeze(0), size=image.shape[1:], mode='nearest').squeeze(0).squeeze(0)
        current_resized = F.interpolate(current.unsqueeze(0).unsqueeze(0), size=image.shape[1:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        new_maxima = torch.max(maxima, maxima_level * current_resized)
        sigma_list.extend([current_sigma] * int(torch.sum(new_maxima > maxima).item()))
        maxima = new_maxima
  
  return maxima, sigma_list


def laplacian_of_gaussian(ksize, sigma):
  """Generate a Laplacian of Gaussian filter."""
  # Reference https://medium.com/@rajilini/laplacian-of-gaussian-filter-log-for-image-processing-c2d1659d5d2

  # Create a 2D Gaussian kernel
  x = torch.linspace(-ksize // 2, ksize // 2, ksize)
  y = torch.linspace(-ksize // 2, ksize // 2, ksize)
  x, y = torch.meshgrid(x, y)
  gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
  gaussian /= (2 * np.pi * sigma**2)
  
  # Compute the Laplacian of the Gaussian
  log_filter = (x**2 + y**2 - 2 * sigma**2) * gaussian
  return log_filter
  


