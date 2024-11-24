import os
import argparse
import torch
from torch import Tensor
from utils.io_helper import torch_read_image, torch_loadtxt
from utils.figure_helper import visualize_matches, visualize_fundamental
from utils.math_helper import get_residual, evaluate_points
from camera import fit_fundamental_unnormalized, fit_fundamental_normalized, camera_calibration, triangulation

device = torch.device("cpu")

def main(args) -> None:
    os.makedirs('results', exist_ok=True)
    
    # Load the images and matches
    image1_fname = f'data/{args.image_name}1.jpg'
    image2_fname = f'data/{args.image_name}2.jpg'
    matches_fname = f'data/{args.image_name}_matches.txt'

    image1 = torch_read_image(image1_fname, gray=False)
    image2 = torch_read_image(image2_fname, gray=False)
    matches = torch_loadtxt(matches_fname, dtype=image1.dtype)

    print(f"Loaded {matches.shape[0]} matches")
    print(f"Matches sample:\n{matches[:5]}")

    # Visualize the matches
    visualize_matches(image1, image2, matches, f'results/{args.image_name}-matches.png')

    # Step 1: Compute the Unnormalized Fundamental Matrix
    fundamental_unnorm = fit_fundamental_unnormalized(matches)
    print('Fundamental Matrix (Unnormalized):\n', fundamental_unnorm)
    pt_line_dist, direction = get_residual(fundamental_unnorm, matches)
    print(f'Unnormalized MSE = {torch.mean(pt_line_dist ** 2).item()}')
    visualize_fundamental(image2, matches, pt_line_dist, direction, f'results/{args.image_name}-F_unnorm.png')

    # Step 2: Compute the Normalized Fundamental Matrix
    fundamental_norm = fit_fundamental_normalized(matches)
    print('Fundamental Matrix (Normalized):\n', fundamental_norm)
    pt_line_dist, direction = get_residual(fundamental_norm, matches)
    print(f'Normalized MSE = {torch.mean(pt_line_dist ** 2).item()}')
    visualize_fundamental(image2, matches, pt_line_dist, direction, f'results/{args.image_name}-F_norm.png')

    # Step 3: Camera Calibration or Load Precomputed Camera Matrices
    match args.image_name:
        case 'lab':
            point3d_fname = f'data/{args.image_name}_3d.txt'
            points3d = torch_loadtxt(point3d_fname, dtype=image1.dtype)

            # Perform camera calibration
            image1_proj = camera_calibration(points3d, matches[:, :2])
            image2_proj = camera_calibration(points3d, matches[:, 2:])

            _, residual1 = evaluate_points(image1_proj, matches[:, :2], points3d)
            _, residual2 = evaluate_points(image2_proj, matches[:, 2:], points3d)

            print(f'Image 1 MSE = {residual1}')
            print(f'Image 2 MSE = {residual2}')

        case 'library':
            # Load precomputed camera projection matrices
            proj1_fname = f'data/{args.image_name}1_camera.txt'
            proj2_fname = f'data/{args.image_name}2_camera.txt'
            image1_proj = torch_loadtxt(proj1_fname, dtype=image1.dtype)
            image2_proj = torch_loadtxt(proj2_fname, dtype=image1.dtype)

        case _:
            raise ValueError(f"Unknown dataset: {args.image_name}")

    print('Image 1 Camera Projection Matrix:\n', image1_proj)
    print('Image 2 Camera Projection Matrix:\n', image2_proj)

    # Step 4: Perform Triangulation
    points3d_pred = triangulation(matches, image1_proj, image2_proj)

    if args.image_name == 'lab':
        # Calculate 3D triangulation error
        residual = torch.mean(torch.sum((points3d_pred - points3d) ** 2, dim=1)).item()
        print(f'Triangulation 3D MSE = {residual}')

    # Step 5: Calculate and Print Reprojection Errors
    _, proj2d_mse_1 = evaluate_points(image1_proj, matches[:, :2], points3d_pred)
    _, proj2d_mse_2 = evaluate_points(image2_proj, matches[:, 2:], points3d_pred)
    print('2D Reprojection Error for Image 1:', proj2d_mse_1.item())
    print('2D Reprojection Error for Image 2:', proj2d_mse_2.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 4 Part 1')
    parser.add_argument(
        '-i', '--image_name', default='library',
        type=str, help='Input image name (e.g., "lab" or "library")')
    args = parser.parse_args()
    main(args)

