import numpy as np
import torch
from torch import Tensor
import torchvision

def torch_loadtxt(filename, dtype=None):
    return torch.from_numpy(np.loadtxt(filename)).to(dtype)

def torch_read_image(filename: str, gray=True) -> Tensor:
    if gray:
        mode = torchvision.io.ImageReadMode.GRAY
    else:
        mode = torchvision.io.ImageReadMode.UNCHANGED
    return torchvision.io.read_image(
        filename, mode) / 255.

def torch_save_image(image: Tensor, filename: str) -> None:
    torchvision.utils.save_image(image, filename)