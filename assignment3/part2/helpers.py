"""Helper functions for part2."""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
##############################################
### Provided code - nothing to change here ###
##############################################

# Image loading and saving


def LoadFaceImages(pathname, subject_name, num_images):
  """
  Load the set of face images.
  The routine returns
      ambimage: image illuminated under the ambient lighting
      imarray: a 3-D array of images, h x w x Nimages
      lightdirs: Nimages x 3 array of light source directions
  """

  def load_image(fname):
    return np.asarray(Image.open(fname))

  def fname_to_ang(fname):
    yale_name = os.path.basename(fname)
    return int(yale_name[12:16]), int(yale_name[17:20])

  def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

  ambimage = load_image(
      os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
  im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
  if num_images <= len(im_list):
    im_sub_list = np.random.choice(im_list, num_images, replace=False)
  else:
    print(
        'Total available images is less than specified.\nProceeding with %d images.\n'
        % len(im_list))
    im_sub_list = im_list
  im_sub_list.sort()
  imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
  Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

  x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
  lightdirs = np.stack([y, z, x], axis=-1)

  ambimage = torch.from_numpy(ambimage.copy()).float()
  imarray = torch.from_numpy(imarray.copy()).float().permute(2, 0, 1)
  lightdirs = torch.from_numpy(lightdirs.copy()).float()
  return ambimage, imarray, lightdirs


def save_outputs(subject_name, albedo_image, surface_normals):
  im = Image.fromarray((albedo_image*255).astype(np.uint8))
  im.save("%s_albedo.jpg" % subject_name)
  im = Image.fromarray((surface_normals[:, :, 0]*128+128).astype(np.uint8))
  im.save("%s_normals_x.jpg" % subject_name)
  im = Image.fromarray((surface_normals[:, :, 1]*128+128).astype(np.uint8))
  im.save("%s_normals_y.jpg" % subject_name)
  im = Image.fromarray((surface_normals[:, :, 2]*128+128).astype(np.uint8))
  im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map
def set_aspect_equal_3d(ax):
  """https://stackoverflow.com/questions/13685386"""
  """Fix equal aspect bug for 3D plots."""
  xlim = ax.get_xlim3d()
  ylim = ax.get_ylim3d()
  zlim = ax.get_zlim3d()
  from numpy import mean
  xmean = mean(xlim)
  ymean = mean(ylim)
  zmean = mean(zlim)
  plot_radius = max([
      abs(lim - mean_)
      for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
      for lim in lims
  ])
  ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
  ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
  ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map, view_angle=[25, 55]):
  fig = plt.figure()
  plt.imshow(albedo_image, cmap='gray')
  plt.axis('off')

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(projection='3d')
  ax.view_init(view_angle[0], view_angle[1])
  X = np.arange(albedo_image.shape[0])
  Y = np.arange(albedo_image.shape[1])
  X, Y = np.meshgrid(Y, X)
  H = np.flipud(np.fliplr(height_map))
  A = np.flipud(np.fliplr(albedo_image))
  A = np.stack([A, A, A], axis=-1)
  ax.xaxis.set_ticks([])
  ax.xaxis.set_label_text('Z')
  ax.yaxis.set_ticks([])
  ax.yaxis.set_label_text('X')
  ax.zaxis.set_ticks([])
  ax.yaxis.set_label_text('Y')
  surf = ax.plot_surface(
      H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
  set_aspect_equal_3d(ax)


# Plot the surface normals
def plot_surface_normals(surface_normals):
  """
  surface_normals: h x w x 3 matrix.
  """
  fig = plt.figure()
  ax = plt.subplot(1, 3, 1)
  ax.axis('off')
  ax.set_title('X')
  im = ax.imshow(surface_normals[:, :, 0])
  ax = plt.subplot(1, 3, 2)
  ax.axis('off')
  ax.set_title('Y')
  im = ax.imshow(surface_normals[:, :, 1])
  ax = plt.subplot(1, 3, 3)
  ax.axis('off')
  ax.set_title('Z')
  im = ax.imshow(surface_normals[:, :, 2])
