"""Implements image stitching."""

import numpy as np
import torch
import kornia
import cv2
import torch.nn.functional as F
from helpers import plot_inlier_matches, harris, get_pixel_descriptors, get_hynet_descriptors
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform, warp
from scipy.ndimage import gaussian_filter
from torchvision import models



class ImageStitcher(object):
  def __init__(self, img1, img2, keypoint_type='harris', descriptor_type='pixel'):
    """
    Inputs:
        img1: h x w tensor.
        img2: h x w tensor.
        keypoint_type: string in ['harris']
        descriptor_type: string in ['pixel', 'hynet']
    """
    # The loaded images are already in the grayscale and converted to tensors.
    self.img1 = img1
    self.img2 = img2
    self.keypoint_type = keypoint_type
    self.descriptor_type = descriptor_type

    print(f"img1 shape: {self.img1.shape}") # 4D dimension (1,1,683,1024)

    # 2. Gaussian filtering
    filtering = True
    if filtering:
      # Create the Gaussian filter parameters
      sigma = 1.5

      # Apply Gaussian filter to both images 
      im_gauss_filt_left = gaussian_filter(self.img1.squeeze(), sigma=sigma) # squeeze is gonna remobve 1,1
      im_gauss_filt_right = gaussian_filter(self.img2.squeeze(), sigma=sigma)

    #### Your Implementation Below #### 

    # Extract keypoints
    self.keypoints1 = self._get_keypoints(im_gauss_filt_left) 
    self.keypoints2 = self._get_keypoints(im_gauss_filt_right) 

    # Extract descriptors at each keypoint
    self.desc1 = self._get_descriptors(im_gauss_filt_left, self.keypoints1)
    self.desc2 = self._get_descriptors(im_gauss_filt_right, self.keypoints2)

    print("Descriptors", self.desc1)

    # Compute putative matches and match the keypoints.
    matched_keypoints = self._get_putative_matches(self.keypoints1, self.keypoints2, self.desc1, self.desc2)

    # Perform RANSAC to find the best homography and inliers
    best_inliers, best_homography, mean_residual = self._ransac(matched_keypoints)

    print("Number of inliers", len(best_inliers))

    print("Mean Residual", mean_residual)

    self._visualize_inliers(self.img1, self.img2, matched_keypoints, best_inliers)

    # Retrieve the indices of inliers
    inlier_indices = torch.where(best_inliers)[0]  # Get indices where inliers are True

    # Now, filter `matched_keypoints` using inlier indices
    inlier_keypoints = matched_keypoints[inlier_indices]

    # Plot the inliers
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(
        ax,
        kornia.utils.tensor_to_image(img1),
        kornia.utils.tensor_to_image(img2),
        inlier_keypoints
    )
    plt.savefig(f'inlier_matches_{self.descriptor_type}.png')

    # Refit with all inliers to get the final homography
    stitched = self.stitch_images(img1, img2, best_homography) 

    plt.gray()
    plt.savefig('stitched_%s.png' % self.descriptor_type)

    plt.imshow(stitched, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()  # Display the image

  def _get_keypoints(self, img):
    # Get Harris responses
    keypoints = harris(img)
    keypoints = np.array(keypoints)

    # Check if keypoints only have (x, y) and add a default response if needed
    if keypoints.shape[1] == 2:
        # Adding a dummy response value of 1.0 for each keypoint
        keypoints = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

    # Threshold on response (though it's less meaningful with dummy values)
    min_response = 0.01
    keypoints = keypoints[keypoints[:, 2] > min_response]

    # Non-maximal suppression
    def nms(keypoints, window_size=5):
        result = []
        for i, kp in enumerate(keypoints):
            x, y = int(kp[0]), int(kp[1])
            window = keypoints[max(0, i - window_size):min(len(keypoints), i + window_size)]
            if kp[2] == max(window[:, 2]):  # If strongest in local window
                result.append(kp)
        return np.array(result)

    keypoints = nms(keypoints)

    return keypoints

  def _get_descriptors(self, image, keypoints):
    """
    Extract descriptors from the image at the given keypoints.

    Inputs:
        image: h x w tensor.
        keypoints: N x 2 tensor.
    Outputs:
        descriptors: N x D tensor.
    """
    if self.descriptor_type == 'pixel':
      descriptors = get_pixel_descriptors(image, keypoints) 
    elif self.descriptor_type == 'hynet':
      # Load a pre-trained model
      model = models.resnet18(pretrained=True)
      # Remove the last layer to obtain features
      model = torch.nn.Sequential(*(list(model.children())[:-1]))
      descriptors = get_hynet_descriptors(image, keypoints, model)
      
    # Convert the list of numpy array to numpy array first, then to torch tensor
    desc = np.array(descriptors)

    # Now convert the numpy arrays to torch tensors
    descriptors = torch.tensor(desc)
    print("shape", descriptors.shape)
    return descriptors

  def _get_putative_matches(self, keypoints1, keypoints2, desc1, desc2, max_num_matches=500):
    distances = torch.cdist(desc1, desc2, p=2).cpu().numpy()
    
    # Add distance threshold
    max_distance = 0.6
    
    matches = []
    for i in range(distances.shape[0]):
        # Get the two closest matches
        dist_i = distances[i, :]
        closest_two = np.partition(dist_i, 1)[:2]
        
        if closest_two[0] < 0.75 * closest_two[1] and closest_two[0] < max_distance:
            j = np.argmin(dist_i)
            matches.append([
                keypoints1[i][0], keypoints1[i][1],
                keypoints2[j][0], keypoints2[j][1]
            ])
            
    matches = matches[:max_num_matches]

    return torch.tensor(matches)
      
  def _homography_inliers(self, H, matched_keypoints, inlier_threshold=10):
    # Ensure H is of type float32 for consistency
    H = H.float()  # Convert H to float32 if it's not already
    matched_keypoints = matched_keypoints.float()
    # Extract points from matched keypoints
    p1 = matched_keypoints[:, :2]
    p2 = matched_keypoints[:, 2:]

    ones = torch.ones((p1.shape[0], 1), dtype=p1.dtype)  # Ensure ones is the same dtype as p1
    p1_homog = torch.cat([p1, ones], dim=1)

    # Perform matrix multiplication
    p1_proj_homog = (H @ p1_homog.T).T
    p1_proj = p1_proj_homog[:, :2] / p1_proj_homog[:, 2:]

    # Calculate distances and determine inliers
    distances = torch.sum((p2 - p1_proj) ** 2, dim=1)
    inliers = distances < inlier_threshold ** 2

    return inliers, distances[inliers]

  def _visualize_inliers(self, img_left, img_right, matched_keypoints, inliers):
    """
    Visualize the inliers between two images.

    Inputs:
        img_left: Left image.
        img_right: Right image.
        matched_keypoints: N x 4 tensor of matched keypoints.
        inliers: Boolean mask tensor indicating which matches are inliers.
    """
    # Squeeze the images to remove any extra dimensions
    img_left = img_left.squeeze()  # Remove dimensions of size 1
    img_right = img_right.squeeze()

    # Check the number of dimensions and adjust accordingly
    if img_left.dim() == 2:  # Grayscale image
        img_left = img_left.unsqueeze(2)  # Add a channel dimension
        img_left = img_left.expand(-1, -1, 3)  # Convert to 3-channel

    if img_right.dim() == 2:  # Grayscale image
        img_right = img_right.unsqueeze(2)  # Add a channel dimension
        img_right = img_right.expand(-1, -1, 3)  # Convert to 3-channel

    # Create a new figure for visualization
    plt.figure(figsize=(12, 6))

    # Plot the left image
    plt.subplot(1, 2, 1)
    plt.imshow(img_left.permute(1, 0, 2).cpu().numpy())  # Ensure it is HxWxC
    plt.scatter(matched_keypoints[inliers, 0].cpu(), matched_keypoints[inliers, 1].cpu(), color='green', label='Inliers')
    plt.title('Left Image with Inliers')
    plt.axis('off')
    plt.legend()

    # Plot the right image
    plt.subplot(1, 2, 2)
    plt.imshow(img_right.permute(1, 0, 2).cpu().numpy())  # Ensure it is HxWxC
    plt.scatter(matched_keypoints[inliers, 2].cpu(), matched_keypoints[inliers, 3].cpu(), color='green', label='Inliers')
    plt.title('Right Image with Inliers')
    plt.axis('off')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
  
  def _get_homography(self, matched_keypoints):

    matched_keypoints = matched_keypoints.float()

    # Check the shape
    if matched_keypoints.dim() == 1:
        print("Reshaping matched_keypoints to 2D")
        matched_keypoints = matched_keypoints.unsqueeze(0)  # Reshape to 2D if necessary

    print("Shape of matched_keypoints:", matched_keypoints.shape)

    # Ensure that matched_keypoints has the correct dimensions
    if matched_keypoints.shape[1] != 4:
        raise ValueError("Expected matched_keypoints to have shape [N, 4]")

    # Normalize coordinates
    p1 = matched_keypoints[:, :2]
    p2 = matched_keypoints[:, 2:]
    
    # Compute centroids
    c1 = torch.mean(p1, dim=0)
    c2 = torch.mean(p2, dim=0)
    
    # Compute scaling
    s1 = torch.tensor(2.).sqrt() / torch.std(p1 - c1)  # Changed this line
    s2 = torch.tensor(2.).sqrt() / torch.std(p2 - c2)
    
    # Create normalization matrices
    T1 = torch.tensor([[s1, 0, -s1*c1[0]], 
                      [0, s1, -s1*c1[1]], 
                      [0, 0, 1]], dtype=torch.float32)
    
    T2 = torch.tensor([[s2, 0, -s2*c2[0]], 
                      [0, s2, -s2*c2[1]], 
                      [0, 0, 1]], dtype=torch.float32)
    
    # Normalize points
    p1_norm = (T1 @ torch.cat([p1, torch.ones(p1.shape[0], 1)], dim=1).T).T[:, :2]
    p2_norm = (T2 @ torch.cat([p2, torch.ones(p2.shape[0], 1)], dim=1).T).T[:, :2]
    
    # Construct A matrix
    A = []
    for (x1, y1), (x2, y2) in zip(p1_norm, p2_norm):
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    
    A = torch.tensor(A, dtype=torch.float32)
    
    # Solve for homography
    _, _, V = torch.svd(A)
    H_norm = V[:, -1].view(3, 3)
    
    # Denormalize
    H = torch.inverse(T2) @ H_norm @ T1
    
    return H / H[2,2]  # Normalize to ensure H[2,2] = 1

  def _ransac(self, matched_keypoints, num_iterations=800, inlier_threshold=10):
      max_inliers_count = 0
      best_homography = None
      best_inliers = None
      mean_residual = None
      min_samples = 4
      
      # Add early termination condition
      target_inliers = 0.7 * len(matched_keypoints)  # 70% of matches
      
      for _ in range(num_iterations):
          # Randomly sample minimum number of points
          idxs = torch.randperm(matched_keypoints.shape[0])[:min_samples]
          sample_matches = matched_keypoints[idxs]
          
          # Compute homography from the sample

          H = self._get_homography(sample_matches)
          
          # Find inliers
          inliers, distances = self._homography_inliers(H, matched_keypoints, inlier_threshold)
          inliers_count = inliers.sum().item()
          
          if inliers_count > max_inliers_count:
              max_inliers_count = inliers_count
              best_inliers = inliers
              best_homography = H
              mean_residual = distances.mean().item() if inliers_count > 0 else None
          
          # Early termination if we found a good solution
          if inliers_count > target_inliers:
              break
      
      # Recompute homography using all inliers
      if best_inliers is not None and best_inliers.sum() >= 4:
          inlier_matches = matched_keypoints[best_inliers]
          best_homography = self._get_homography(inlier_matches)
      
      return best_inliers, best_homography, mean_residual
  
  def stitch_images(self, img1, img2, H):
    # Convert tensors to numpy arrays
    img1_np = kornia.utils.tensor_to_image(img1.squeeze())  # Remove batch and channel dims
    img2_np = kornia.utils.tensor_to_image(img2.squeeze())
    
    # Ensure 3 channels
    if len(img1_np.shape) == 2:
        img1_np = np.stack([img1_np] * 3, axis=-1)
    if len(img2_np.shape) == 2:
        img2_np = np.stack([img2_np] * 3, axis=-1)
    
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    
    # Convert homography to numpy
    H_np = H.detach().cpu().numpy()
    
    # Find corners of warped image
    corners = np.array([[0, 0, 1],
                       [w2-1, 0, 1],
                       [w2-1, h2-1, 1],
                       [0, h2-1, 1]])
    warped_corners = H_np @ corners.T
    warped_corners = warped_corners / warped_corners[2]
    
    # Find bounding box
    min_x = min(0, warped_corners[0].min())
    max_x = max(w1, warped_corners[0].max())
    min_y = min(0, warped_corners[1].min())
    max_y = max(h1, warped_corners[1].max())
    
    # Create translation matrix
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]])
    
    # Compute output size
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))
    
    # Warp images
    warped_img2 = cv2.warpPerspective(img2_np, T @ H_np, (out_w, out_h))
    output_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    output_img[-int(min_y):h1-int(min_y), -int(min_x):w1-int(min_x)] = img1_np
    
    # Create mask for blending
    mask = (warped_img2 != 0).any(axis=2)
    output_img[mask] = warped_img2[mask]
    
    return output_img






   