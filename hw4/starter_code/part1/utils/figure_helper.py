import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from utils.math_helper import get_residual

def visualize_matches(
    image1: Tensor,
    image2: Tensor,
    matches: Tensor,
    out_fname: str,
    sample: int = 1
):
    r'''
    Display two images side-by-side with matches (red arrows)
    '''
    _, height, width = image1.shape
    canvas = torch.zeros([3, height, width*2])
    canvas[:, :, :width] = image1
    canvas[:, :, width:] = image2
    
    fig, ax = plt.subplots(dpi=200)
    ax.set_aspect('equal')
    ax.imshow(canvas.permute(1, 2, 0))
    
    matches = matches[::sample]
    x1, y1 = matches[:, 0], matches[:, 1]
    x2, y2 = matches[:, 2] + width, matches[:, 3]
    
    ax.plot(x1, y1, '+r')
    ax.plot(x2, y2, '+r')
    ax.plot([x1, x2],[y1, y2], 'r', linewidth=1)

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()

def visualize_fundamental(
    image, 
    matches, 
    pt_line_dist: Tensor,
    direction: Tensor, 
    out_fname: str,
    offset: int = 40,    
):
    r'''
    Display second image with epipolar lines reprojected 
    from the first image
    '''
    N = len(matches)
    closest_pt = matches[:, 2:] - direction[:, :2] * torch.ones([N, 2]).to(matches) * pt_line_dist[:, None]

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset# offset from the closest point is 10 pixels
    pt2 = closest_pt + torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots(dpi=200)
    ax.set_aspect('equal')
    ax.imshow(image.permute(1, 2, 0))
    ax.plot(matches[:, 2], matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:,3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()