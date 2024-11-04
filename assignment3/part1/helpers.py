"""Implements helper functions."""
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image
import numpy as np
import cv2 
import torch
import torchvision.transforms as transforms

##############################################
### Provided code - nothing to change here ###
##############################################


def plot_inlier_matches(ax, img1, img2, inliers):
  """
  Plot the matches between two images according to the matched keypoints
  :param ax: plot handle
  :param img1: left image
  :param img2: right image
  :inliers: x,y in the first image and x,y in the second image (Nx4)

  Usage:
    fig, ax = plt.subplots(figsize=(20,10))
    plot_inlier_matches(ax, img1, img2, computed_inliers)
  """
  res = np.hstack([img1, img2])
  ax.set_aspect('equal')
  ax.imshow(res, cmap='gray')

  ax.plot(inliers[:, 0], inliers[:, 1], '+r')
  ax.plot(inliers[:, 2] + img1.shape[1], inliers[:, 3], '+r')
  ax.plot([inliers[:, 0], inliers[:, 2] + img1.shape[1]],
          [inliers[:, 1], inliers[:, 3]], 'r', linewidth=0.4)
  ax.axis('off')


"""
Harris Corner Detector
Usage: Call the function harris(filename) for corner detection
Reference   (Code adapted from):
             http://www.kaij.org/blog/?p=89
             Kai Jiang - Harris Corner Detector in Python

"""


def harris(img, min_distance=6, threshold=0.06):
  """
  image: h x w tensor (grayscale image). (It was filename before: Path of image file)
  threshold: (optional)Threshold for corner detection
  min_distance : (optional)Minimum number of pixels separating
   corners and image boundary
  """
  # im = np.array(Image.open(img).convert("L"))
  harrisim = compute_harris_response(img)
  filtered_coords = get_harris_points(harrisim, min_distance, threshold)
  plot_harris_points(img, filtered_coords)
  return filtered_coords


def gauss_derivative_kernels(size, sizey=None):
  """ returns x and y derivatives of a 2D
      gauss kernel array for convolutions """
  size = int(size)
  if not sizey:
    sizey = size
  else:
    sizey = int(sizey)
  y, x = mgrid[-size:size+1, -sizey:sizey+1]
  # x and y derivatives of a 2D gaussian with standard dev half of size
  # (ignore scale factor)
  gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
  gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2)))
  return gx, gy


def gauss_kernel(size, sizey=None):
  """ Returns a normalized 2D gauss kernel array for convolutions """
  size = int(size)
  if not sizey:
    sizey = size
  else:
    sizey = int(sizey)
  x, y = mgrid[-size:size+1, -sizey:sizey+1]
  g = exp(-(x**2/float(size)+y**2/float(sizey)))
  return g / g.sum()


def compute_harris_response(im):
  """ compute the Harris corner detector response function
      for each pixel in the image"""
  # derivatives
  gx, gy = gauss_derivative_kernels(3)
  imx = signal.convolve(im, gx, mode='same')
  imy = signal.convolve(im, gy, mode='same')
  # kernel for blurring
  gauss = gauss_kernel(3)
  # compute components of the structure tensor
  Wxx = signal.convolve(imx*imx, gauss, mode='same')
  Wxy = signal.convolve(imx*imy, gauss, mode='same')
  Wyy = signal.convolve(imy*imy, gauss, mode='same')
  # determinant and trace
  Wdet = Wxx*Wyy - Wxy**2
  Wtr = Wxx + Wyy
  return Wdet / Wtr


def get_harris_points(harrisim, min_distance=10, threshold=0.1):
  """ return corners from a Harris response image
      min_distance is the minimum nbr of pixels separating
      corners and image boundary"""
  # find top corner candidates above a threshold
  corner_threshold = max(harrisim.ravel()) * threshold
  harrisim_t = (harrisim > corner_threshold) * 1
  # get coordinates of candidates
  candidates = harrisim_t.nonzero()
  coords = [(candidates[0][c], candidates[1][c])
            for c in range(len(candidates[0]))]
  # ...and their values
  candidate_values = [harrisim[c[0]][c[1]] for c in coords]
  # sort candidates
  index = argsort(candidate_values)
  # store allowed point locations in array
  allowed_locations = zeros(harrisim.shape)
  allowed_locations[min_distance:-min_distance, min_distance:-min_distance] = 1
  # select the best points taking min_distance into account
  filtered_coords = []
  for i in index:
    if allowed_locations[coords[i][0]][coords[i][1]] == 1:
      filtered_coords.append(coords[i])
      allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                        (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0
  return filtered_coords


def plot_harris_points(image, filtered_coords):
  """ plots corners found in image"""
  figure()
  gray()
  imshow(image)
  plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'r*')
  axis('off')
  show()

# Usage:
# harris('./path/to/image.jpg')

def get_pixel_descriptors(image, keypoints, neighborhood_size=5):
    """
    Extract descriptors from the image based on local neighborhoods around keypoints.
    
    Parameters:
        image: Grayscale image (2D array).
        keypoints: List of keypoint coordinates (y, x, response).
        neighborhood_size: The size of the neighborhood to extract around each keypoint.
        
    Returns:
        descriptors: List of normalized 1D descriptors for each keypoint.
    """
    descriptors = []
    half_size = neighborhood_size // 2
    for kp in keypoints:
        y, x, _ = kp  # Unpack y, x, and ignore the response

        # Ensure y and x are integers
        y = int(y)
        x = int(x)

        # Ensure that the keypoint is far enough from the border
        if (y - half_size >= 0 and y + half_size < image.shape[0] and 
            x - half_size >= 0 and x + half_size < image.shape[1]):
            # Extract the neighborhood around the keypoint
            neighborhood = image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
            
            # Flatten the neighborhood into a 1D vector
            descriptor = neighborhood.flatten()
            
            # Subtract the mean and normalize the descriptor to have unit norm
            descriptor = descriptor - np.mean(descriptor)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            
            # Append the descriptor to the list
            descriptors.append(descriptor)

    return descriptors

def get_hynet_descriptors(image, keypoints, model):
  descriptors = []

  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  for kp in keypoints:
      y, x, _ = kp
      y = int(y)
      x = int(x)

      neighborhood_size = 224
      half_size = neighborhood_size // 2
      
      if (y - half_size >= 0 and y + half_size < image.shape[0] and 
          x - half_size >= 0 and x + half_size < image.shape[1]):
          neighborhood = image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
          
          if len(neighborhood.shape) == 2:  # Convert grayscale to RGB
              neighborhood = np.stack([neighborhood]*3, axis=-1)

          neighborhood = transform(neighborhood)
          neighborhood = neighborhood.unsqueeze(0)
          
          with torch.no_grad():
              model.eval()
              features = model(neighborhood)
              
          descriptor = features.flatten().cpu().numpy()
          descriptors.append(descriptor)

  # Stack descriptors into a 2D NumPy array
  descriptors = np.vstack(descriptors) if descriptors else np.empty((0, model.fc.out_features))

  return descriptors




