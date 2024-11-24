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

# def visualize_fundamental(
#     image, 
#     matches, 
#     pt_line_dist: Tensor,
#     direction: Tensor, 
#     out_fname: str,
#     offset: int = 40,    
# ):
#     r'''
#     Display second image with epipolar lines reprojected 
#     from the first image
#     '''
#     N = len(matches)
#     closest_pt = matches[:, 2:] - direction[:, :2] * torch.ones([N, 2]).to(matches) * pt_line_dist[:, None]

#     # find endpoints of segment on epipolar line (for display purposes)
#     pt1 = closest_pt - torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset# offset from the closest point is 10 pixels
#     pt2 = closest_pt + torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset

#     # display points and segments of corresponding epipolar lines
#     fig, ax = plt.subplots(dpi=200)
#     ax.set_aspect('equal')
#     ax.imshow(image.permute(1, 2, 0))
#     ax.plot(matches[:, 2], matches[:, 3],  '+r')
#     ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:,3], closest_pt[:, 1]], 'r')
#     ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    
#     plt.tight_layout()
#     plt.axis('off')
#     plt.savefig(out_fname, bbox_inches='tight')
#     plt.close()

# def visualize_fundamental(
#     image: Tensor, 
#     matches: Tensor, 
#     pt_line_dist: Tensor,
#     direction: Tensor, 
#     out_fname: str,
#     offset: int = 20,  # Reduce offset for better visualization
#     sample: int = 10   # Draw every 10th match for clarity
# ):
#     """
#     Display the second image with epipolar lines reprojected from the first image.
#     """
#     N = len(matches)
#     matches = matches[::sample]
#     pt_line_dist = pt_line_dist[::sample]
#     direction = direction[::sample]
#     closest_pt = matches[:, 2:] - direction[:, :2] * torch.ones([len(matches), 2]).to(matches) * pt_line_dist[:, None]

#     # Find endpoints of segment on epipolar line (for display purposes)
#     pt1 = closest_pt - torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset
#     pt2 = closest_pt + torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset

#     # Convert PyTorch tensor to numpy for matplotlib
#     img_np = image.permute(1, 2, 0).cpu().numpy()

#     # Display points and segments of corresponding epipolar lines
#     fig, ax = plt.subplots(dpi=150)
#     ax.set_aspect('equal')
#     ax.imshow(img_np)
    
#     # Draw the red '+' markers for matched points
#     ax.plot(matches[:, 2], matches[:, 3], '+r', label='Matched Points')
    
#     # Draw lines from the matched points to the closest point on the epipolar line
#     for i in range(len(matches)):
#         ax.plot([matches[i, 2], closest_pt[i, 0]], [matches[i, 3], closest_pt[i, 1]], 'r')
#         ax.plot([pt1[i, 0], pt2[i, 0]], [pt1[i, 1], pt2[i, 1]], 'g')

#     plt.title("Epipolar Lines Visualization")
#     plt.tight_layout()
#     plt.axis('off')
#     plt.savefig(out_fname, bbox_inches='tight')
#     plt.close()
#     print(f"Epipolar lines visualization saved to {out_fname}")

def visualize_fundamental(
    image: Tensor,
    matches: Tensor,
    pt_line_dist: Tensor,
    direction: Tensor,
    out_fname: str,
    offset: int = 50,  # Offset for epipolar lines
    sample: int = 5,   # Subset every 5th match for clarity
    line_color: str = 'blue',  # Epipolar line color
    point_color: str = 'red'   # Point marker color
):
    """
    Display the second image with epipolar lines reprojected from the first image.
    Args:
        image: Tensor of shape (C, H, W) (the second image).
        matches: Tensor of matched points, shape (N, 4) [(x1, y1, x2, y2)].
        pt_line_dist: Residual distances from points to epipolar lines.
        direction: Direction vectors of epipolar lines.
        out_fname: Path to save the output visualization.
        offset: Length of epipolar lines drawn.
        sample: Subset matches to every `sample`th match.
    """
    # Subset matches for clarity
    matches = matches[::sample]
    pt_line_dist = pt_line_dist[::sample]
    direction = direction[::sample]

    # Compute closest points on epipolar lines
    closest_pt = matches[:, 2:] - direction[:, :2] * pt_line_dist[:, None]

    # Compute line endpoints
    pt1 = closest_pt - torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset
    pt2 = closest_pt + torch.stack([direction[:, 1], -direction[:, 0]], dim=1) * offset

    # Convert image from Tensor to numpy for visualization
    img_np = image.permute(1, 2, 0).cpu().numpy()

    # Visualization
    fig, ax = plt.subplots(dpi=150)
    ax.imshow(img_np)  # Show the second image
    ax.set_aspect('equal')

    # Plot epipolar lines
    for i in range(len(matches)):
        ax.plot([pt1[i, 0].item(), pt2[i, 0].item()],
                [pt1[i, 1].item(), pt2[i, 1].item()],
                color=line_color, linewidth=1.5)

    # Plot red '+' for the matched points
    ax.scatter(matches[:, 2], matches[:, 3], c=point_color, label='Matched Points', s=10, zorder=5)

    plt.title("Epipolar Lines with Subset of Matches")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {out_fname}")
