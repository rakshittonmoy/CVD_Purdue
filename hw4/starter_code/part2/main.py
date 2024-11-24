import os
import argparse
import numpy as np

import torch
from torch import Tensor

from affine import (
    normalize_measurements,
    get_structure_and_motion,
    get_Q
)

from utils.io_helper import torch_loadtxt, torch_read_image
from utils.figure_helper import (
    visualize_structure, 
    visualize_keypoints_comparison,
    visualize_errors
)

def main(args) -> None:
    os.makedirs('results', exist_ok=True)
    
    D = torch_loadtxt('data/measurement_matrix.txt')
    num_frames = len(D) // 2

    frames: list[Tensor] = []
    for i in range(num_frames):
        image_fname = f'data/frame{i+1:08d}.jpg'
        frames.append(torch_read_image(image_fname, gray=True))
    colors = [ 
        (frames[0][:, int(y), int(x)]).item() \
        for (x, y) in D[:2].t() ]

    D, mean = normalize_measurements(D)
    M, S = get_structure_and_motion(D)
    visualize_structure(S.t(), colors, out_name='S1')

    Q = get_Q(M)

    MQ = M @ Q
    QS = Q.inverse() @ S
    visualize_structure(QS.t(), colors, out_name='S2')

    errors = []
    for i in range(num_frames):
        m = MQ[2*i:2*i+2]
        a = m @ QS
        # keypoints1 = D[2*i:2*i+2] + mean[i]
        keypoints1 = D[2*i:2*i+2] + mean[i].view(2, 1)
        # keypoints2 = a + mean[i]      
        keypoints2 = a + mean[i].view(2, 1)  
        visualize_keypoints_comparison(
            frames[i], keypoints1, keypoints2, out_name=f'k{i+1:08d}')
        mse = (keypoints1 - keypoints2).square().sum(0).mean()
        print(mse)
        errors.append(mse)

    visualize_errors(errors, out_name=f'mse')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS59300CVD Assignment 4 Part 2')
    args = parser.parse_args()
    main(args)

