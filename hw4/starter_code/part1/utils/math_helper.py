import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

def evaluate_points(
    projection: Tensor, 
    points_2d: Tensor, 
    points_3d: Tensor
):
    r"""
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    
    Arguments
    - `projection`: projection matrix `3` x `4`
    - `points_2d`: 2D points `N` x `2`
    - `points_3d`: 3D points `N` x `3`
    
    Return
    - `N` x `2`
    - MSE
    """ 
    N = len(points_3d)
    points_3d_homo = torch.cat([points_3d, torch.ones([N, 1])], dim=1) # N, 4
    points_2d_pred = (points_3d_homo @ projection.t())#.t() 
    points_2d_pred = points_2d_pred[:, :2] / points_2d_pred[:, 2:]

    residual = (points_2d_pred - points_2d).square().sum(1).mean(0)
    return points_2d_pred, residual

def get_residual(
    F: Tensor, matches: Tensor):
    """
    Function to compute the average residual on frame 2
    param: F (3x3): fundamental matrix: (pt in frame 2).T * F * (pt in frame 1) = 0
    param: p1 (Nx2): 2d points on frame 1
    param: p2 (Nx2): 2d points on frame 2
    """
    N = len(matches)
    p1 = torch.cat([matches[:, :2], torch.ones([N, 1])], dim=1) # N x 3
    p2 = torch.cat([matches[:, 2:], torch.ones([N, 1])], dim=1) # N x 3

    L2 = p1 @ F.t() # N x 3
    L2_norm = torch.sqrt(L2[:, 0]**2 + L2[:, 1]**2)
    L2 = L2 / L2_norm[:, None]
   
    pt_line_dist = (L2 * p2).sum(dim=1)
    # return np.mean(np.square(pt_line_dist))
    return pt_line_dist, L2
