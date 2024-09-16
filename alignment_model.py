"""Implements the alignment algorithm."""

import torch
import torchvision
from metrics import ncc, mse, ssim
from helpers import custom_shifts
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class AlignmentModel:
  def __init__(self, image_name, metric='ssim', padding='circular'):
    # Image name
    self.image_name = image_name
    # Metric to use for alignment
    self.metric = metric
    # Padding mode for custom_shifts
    self.padding = padding

  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def align(self):
    """Aligns the image using the metric specified in the constructor.
       Experiment with the ordering of the alignment.

       Finally, outputs the rgb image in self.rgb.
    """
    self.img = self._load_image()
    
    b_channel, g_channel, r_channel = self._crop_and_divide_image()
    delta = (10, 10)

    # Align Green and Blue channels to the Red channel
    g_shift = self._align_pairs(r_channel, g_channel, delta)
    b_shift = self._align_pairs(r_channel, b_channel, delta)

    # Apply shifts using custom_shifts
    aligned_g = custom_shifts(g_channel, g_shift, dims=(1, 2), padding=self.padding)
    aligned_b = custom_shifts(b_channel, b_shift, dims=(1, 2), padding=self.padding)

    aligned_r = r_channel

    # Stack the channels to create the final RGB image
    self.rgb = torch.stack([aligned_r, aligned_g, aligned_b], dim=0)

  def _load_image(self):
    """Load the image from the image_name path,
       typecast it to float, and normalize it.

       Returns: torch.Tensor of shape (H, W)
    """
    ret = None
    image = Image.open(self.image_name)

    # Convert to tensor and then to float.
    ret = transforms.ToTensor()(image).float()

    return ret

  
  def _crop_and_divide_image(self):
    """Crop the image boundary and divide the image into three parts, padded to the same size.

       Feel free to be creative about this.
       You can eyeball the boundary values, or write code to find approximate cut-offs.
       Hint: Plot out the average values per row / column and visualize it!

       Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
    """
    height, width = self.img.shape[-2:]

    # Remove the border by cropping a fixed number of pixels.
    cropped_img = self.img[:, 10:height - 10, 10:width - 10]

    # Divide the image into three parts
    h_third = cropped_img.shape[1] // 3

    b_channel = cropped_img[:, :h_third]
    g_channel = cropped_img[:, h_third:2 * h_third]
    r_channel = cropped_img[:, 2 * h_third:]

    # Adjust for any remainder by padding the channels to match the largest one
    max_height = max(b_channel.shape[1], g_channel.shape[1], r_channel.shape[1])

    # Padding each channel to match the max height
    if b_channel.shape[1] < max_height:
        pad_size = max_height - b_channel.shape[1]
        b_channel = torch.cat([b_channel, torch.zeros((b_channel.shape[0], pad_size, b_channel.shape[2]), dtype=b_channel.dtype)], dim=1)
    if g_channel.shape[1] < max_height:
        pad_size = max_height - g_channel.shape[1]
        g_channel = torch.cat([g_channel, torch.zeros((g_channel.shape[0], pad_size, g_channel.shape[2]), dtype=g_channel.dtype)], dim=1)
    if r_channel.shape[1] < max_height:
        pad_size = max_height - r_channel.shape[1]
        r_channel = torch.cat([r_channel, torch.zeros((r_channel.shape[0], pad_size, r_channel.shape[2]), dtype=r_channel.dtype)], dim=1)

    # Print shapes for debugging
    print(f"cropped_img shape: {cropped_img.shape}")
    print(f"b_channel shape: {b_channel.shape}")
    print(f"g_channel shape: {g_channel.shape}")
    print(f"r_channel shape: {r_channel.shape}")

    g_channel_resized = self._remove_redundant_dimensions(g_channel)
    r_channel_resized = self._remove_redundant_dimensions(r_channel)
    b_channel_resized = self._remove_redundant_dimensions(b_channel)
    
    return b_channel_resized, g_channel_resized, r_channel_resized


  def _remove_redundant_dimensions(self, img_channel):

    # Reference:- https://stackoverflow.com/questions/68079012/valueerror-win-size-exceeds-image-extent-if-the-input-is-a-multichannel-color
    # Using numpy.squeeze for removing the redundant dimension:    
    img_channel = np.squeeze(img_channel)

    return img_channel


  def _align_pairs(self, img1, img2, delta):
    """
    Aligns two images using the metric specified in the constructor.
    Returns: Tuple of (u, v) shifts that minimizes the metric.
    """

    if not isinstance(img1, torch.Tensor):
        raise TypeError("img1 should be a PyTorch tensor")
    if not isinstance(img2, torch.Tensor):
        raise TypeError("img2 should be a PyTorch tensor")
    
    x, y = delta
    align_idx = (0, 0)
    best_score = float('inf')  # Initialize with a large number

    # brute force technique to find the align index.
    for dx in range(-x, x + 1):
        for dy in range(-y, y + 1):
            shifted_img2 = custom_shifts(img2, (dx, dy), dims=(1, 2), padding=self.padding)
            
            # Calculate the alignment score using the specified metric
            if self.metric == 'ncc':
                score = ncc(img1, shifted_img2)
            elif self.metric == 'mse':
                score = mse(img1, shifted_img2)
            elif self.metric == 'ssim':
                score = ssim(img1, shifted_img2)
            
            # Update best shift if the new score is lower (better alignment)
            if score < best_score:
                best_score = score
                align_idx = (dx, dy)

    print(align_idx)
    return align_idx
