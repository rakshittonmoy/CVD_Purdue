"""Implements Shape from shading."""
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from helpers import LoadFaceImages, display_output


class ShapeFromShading(object):
  def __init__(self, full_path, subject_name, integration_method):
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path,
                                                        subject_name,
                                                        64)
    # Preprocess the data
    processed_imarray = self._preprocess(ambient_image, imarray)

    # Compute albedo and surface normals
    albedo_image, surface_normals = self._photometric_stereo(processed_imarray,
                                                             light_dirs)
    # Save the output
    self.save_outputs(subject_name, albedo_image, surface_normals)

    # Compute height map
    height_map = self._get_surface(surface_normals, integration_method)

    # Save output results
    display_output(albedo_image, height_map)
    plt.savefig('%s_height_map_%s.jpg' % (subject_name, integration_method))

  def _preprocess(self, ambimage, imarray):
    """
    preprocess the data:
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: Nimages x h x w
    Outputs:
        processed_imarray: Nimages x h x w
    """
    processed_imarray = None 
    return processed_imarray

  def _photometric_stereo(self, imarray, light_dirs):
    """
    Inputs:
        imarray:  N x h x w
        light_dirs: N x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    albedo_image = None 
    surface_normals = None

    return albedo_image, surface_normals

  def _get_surface(self, surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    height_map = None
    return height_map
    
  def save_outputs(self, subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).numpy().astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray(
        (surface_normals[:, :, 0]*128+128).numpy().astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray(
        (surface_normals[:, :, 1]*128+128).numpy().astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray(
        (surface_normals[:, :, 2]*128+128).numpy().astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)
