import torch
from torch import Tensor

def fit_fundamental_unnormalized(matches: Tensor) -> Tensor:
    """Computes fundamental matrix using unnormalized 8-point algorithm"""
    # Construct A matrix for Af = 0
    A = []
    for match in matches:
        x1, y1, x2, y2 = match
        A.append([
            x2*x1, x2*y1, x2, 
            y2*x1, y2*y1, y2, 
            x1, y1, 1
        ])
    A = torch.tensor(A, dtype=matches.dtype)
    
    # Solve using SVD
    _, _, Vt = torch.linalg.svd(A)
    f = Vt[-1]  # Last row of Vt
    F = f.reshape(3, 3)
    
    # Enforce rank 2 constraint
    U, S, Vt = torch.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ torch.diag(S) @ Vt

    # F = F / F[-1, -1].clone()
    
    return F

def normalize_points(points):
    """Normalize 2D points using isotropic scaling"""
    mean = points.mean(dim=0)
    centered = points - mean
    scale = torch.sqrt((centered**2).sum(dim=1).mean() * 2)
    normalized = centered / scale
    
    T = torch.eye(3, dtype=points.dtype)
    T[0:2, 0:2] = torch.eye(2) / scale
    T[0:2, 2] = -mean / scale
    
    return normalized, T

def fit_fundamental_normalized(matches: Tensor) -> Tensor:
    """Computes fundamental matrix using normalized 8-point algorithm"""
    # Split points and normalize them
    pts1 = matches[:, :2]
    pts2 = matches[:, 2:]
    
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)
    
    # Stack normalized points back together
    norm_matches = torch.cat([norm_pts1, norm_pts2], dim=1)
    
    # Compute F matrix for normalized coordinates
    F_norm = fit_fundamental_unnormalized(norm_matches)
    
    # Denormalize
    F = T2.t() @ F_norm @ T1

    # Scale F so F[2, 2] = 1
    # F = F / F[-1, -1].clone()
    
    return F

def camera_calibration(pts_3d: Tensor, pts_2d: Tensor) -> Tensor:
    """Computes camera projection matrix using DLT"""
    A = []
    for X, x in zip(pts_3d, pts_2d):
        X = X.tolist()
        x, y = x.tolist()
        
        A.append([
            X[0], X[1], X[2], 1, 0, 0, 0, 0, -x*X[0], -x*X[1], -x*X[2], -x
        ])
        A.append([
            0, 0, 0, 0, X[0], X[1], X[2], 1, -y*X[0], -y*X[1], -y*X[2], -y
        ])
    
    A = torch.tensor(A, dtype=pts_3d.dtype)
    
    # Solve using SVD
    _, _, Vt = torch.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    
    return P

def triangulation_single(x1, y1, x2, y2, P1, P2) -> Tensor:
    """Triangulates a single 3D point from 2D correspondences"""
    A = torch.stack([
        y1 * P1[2] - P1[1],
        P1[0] - x1 * P1[2],
        y2 * P2[2] - P2[1],
        P2[0] - x2 * P2[2]
    ])
    
    _, _, Vt = torch.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]  # Normalize homogeneous coordinates
    
    return X[:3]

def triangulation(matches: Tensor, proj1: Tensor, proj2: Tensor) -> Tensor:
    """Triangulates multiple 3D points from 2D correspondences"""
    points_3d = []
    for match in matches:
        x1, y1, x2, y2 = match
        X = triangulation_single(x1, y1, x2, y2, proj1, proj2)
        points_3d.append(X)
    
    return torch.stack(points_3d)