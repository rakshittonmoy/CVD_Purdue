import argparse
import os
import torch
from utils.io_helper import torch_read_image
from utils.draw_helper import draw_all_circles
from utils.misc_helper import detect_local_maxima, laplacian_of_gaussian
import torch.nn.functional as F


def main(args) -> None:
    os.makedirs('outputs', exist_ok=True)
    if args.input_name == 'all':
        run_all(args)
        return
    blob_detection(
        args.input_name, 'outputs/blob.jpg',
        ksize=args.ksize, sigma=args.sigma, n=args.n)

def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in [
        'butterfly', 'einstein', 'fishes', 'sunflowers'
    ]:
        input_name = 'data/part2/%s.jpg' % image_name
        output_name = 'outputs/%s-blob.jpg' % image_name
        blob_detection(
            input_name, output_name, 
            ksize=args.ksize, sigma=args.sigma, n=args.n)


def blob_detection(
    input_name: str, 
    output_name: str,
    ksize: int,
    sigma: float,
    n: int
) -> None:
    
    # Step 1: Read RGB image as Grayscale
    image = torch_read_image(input_name, gray=True)
    
    # Step 2: Build Laplacian kernel
    log_filter = torch.tensor(laplacian_of_gaussian(ksize, sigma), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Step 3: Build feature pyramid
    k = 1.15  # scale factor
    scale_space = []
    current_image = image
    current_sigma = sigma

    for _ in range(n):
        # Apply LoG filter
        response = F.conv2d(current_image.unsqueeze(0), log_filter, padding=ksize//2, stride=1)
        scale_space.append((response.squeeze(0).squeeze(0), current_sigma))  # store response and sigma

        # Downsample image for next iteration
        current_image = F.interpolate(current_image.unsqueeze(0), scale_factor=1/k, mode='bilinear', align_corners=False).squeeze(0)
        current_sigma *= k
    
    # Step 4: Extract and visualize Keypoints
    # Non-maximum suppression in scale space
    maxima = torch.zeros_like(image)
    sigma_list = []  # store sigma values in a list instead of a tensor
    
    maxima, sigma_list = detect_local_maxima(scale_space, image, n)
    
    # Apply threshold
    threshold = 0.18
    # Reference:- https://www.reddit.com/r/Rlanguage/comments/i9b93o/a_way_to_find_local_maxima_above_certain_threshold/
    # If it's above the threshold do nothing, else set to 0
    maxima[maxima < threshold] = 0 

    # Ensure maxima a 2D tensor by removing the extra dimension, thereby making it like:- [356, 493]
    maxima = maxima.squeeze(0)

    # Filter out small blobs based on size (using sigma for radius)
    keypoints = torch.nonzero(maxima)

    filtered_keypoints = []
    for _, keypoint in enumerate(keypoints):
        cy, cx = keypoint[0].item(), keypoint[1].item()

        # debugging
        # print(f"Keypoint {idx}: cy={cy}, cx={cx}")
        # print(f"Maxima shape: {maxima.shape}")

        current_sigma = sigma_list[len(filtered_keypoints)]

        # Calculate the radius
        radius = 2.0 * current_sigma

        filtered_keypoints.append((cy, cx, radius))

    
    def is_far_enough(new_circle, existing_circles, min_distance=9):
        for existing_circle in existing_circles:
            if (new_circle[0] - existing_circle[0]) ** 2 + (new_circle[1] - existing_circle[1]) ** 2 < min_distance ** 2:
                return False
        return True

    final_circles = []
    for keypoint in filtered_keypoints:
        # doing this to prevent overcrowing of the keypoints on the image.
        if is_far_enough(keypoint, final_circles):
            final_circles.append(keypoint)
        
        
    if final_circles:
        cy = [circle[0] for circle in final_circles]
        cx = [circle[1] for circle in final_circles]
        rad = [circle[2] for circle in final_circles]
    else:
        cy, cx, rad = [], [], []

    # Visualize keypoints
    draw_all_circles(image.squeeze().numpy(), cx, cy, rad, output_name)

    print(f"Detected {len(cx)} blobs.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 2 Part 2')
    parser.add_argument('-i', '--input_name', required=True, type=str, help='Input image path')
    parser.add_argument('-s', '--sigma', type=float)
    parser.add_argument('-k', '--ksize', type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    assert(args.ksize % 2 == 1)
    main(args)
