import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt

def visualize_keypoints_comparison(
    image: Tensor, 
    keypoints1: Tensor,
    keypoints2: Tensor,
    out_name: str
):
    r"""
    - `keypoint`: [2 x M]
    """
    _, _, width = image.shape
    fig, ax = plt.subplots(dpi=200)
    ax.set_aspect('equal')
    canvas = image.repeat(1, 1, 2)
    ax.imshow(canvas.permute(1, 2, 0))
    
    ax.plot(keypoints1[0], keypoints1[1],  '+r')
    ax.plot(keypoints2[0] + width, keypoints2[1],  '+r')
    
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'results/{out_name}.png', bbox_inches='tight')
    plt.close()

def visualize_structure(
    structure: Tensor, 
    colors: list[float],
    out_name: str
):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    for (x, y, z), color in zip(structure, colors):
        ax.scatter(
            x, y, z, 
            c=[[color, color, color]], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Structure of S')

    plt.tight_layout()
    plt.savefig(f'results/{out_name}-1.png', bbox_inches='tight')

    ax.view_init(elev=30, azim=45)
    plt.savefig(f'results/{out_name}-2.png', bbox_inches='tight')

    ax.view_init(elev=-30, azim=-90)
    plt.savefig(f'results/{out_name}-3.png', bbox_inches='tight')
    plt.close()

def visualize_errors(
    errors: Tensor, 
    out_name: str
):
    r"""
    - `keypoint`: [2 x M]
    """
    fig, ax = plt.subplots(dpi=200)
    
    plt.plot(errors, marker='o', linestyle='-', color='b')

    plt.xlabel('Frame index')
    plt.ylabel('MSE')
    plt.title('Errors of each frames')
    
    plt.tight_layout()
    plt.savefig(f'results/{out_name}.png', bbox_inches='tight')
    plt.close()