"""Implements helper functions."""

import torch


def custom_shifts(input, shifts, dims=None, padding='circular'):
  """Shifts the input tensor by the specified shifts along
     the specified dimensions. Supports circular and zero padding.

     Input: Tensor
     Returns: Shifted Tensor along the specified dimension
       padded following the padding scheme.
  """
  if dims is None:
        dims = range(input.dim())

  # Ensure dims are within the valid range
  dims = [d if d < input.dim() else input.dim() - 1 for d in dims]
    
  if padding == 'circular':
      return torch.roll(input, shifts, dims=dims)
  
  return input
