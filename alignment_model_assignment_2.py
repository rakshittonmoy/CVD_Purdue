import torch
import torchvision
from metrics import ncc, mse, ssim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class AlignmentModel_2:
    def __init__(self, image_name, metric='ssim', padding='circular'):
        # Image name
        self.image_name = image_name
        # Metric to use for alignment
        self.metric = metric
        # Padding mode for custom_shifts
        self.padding = padding

    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def align(self):
        """Aligns the image using the metric specified in the constructor.
           Experiment with the ordering of the alignment.

           Finally, outputs the rgb image in self.rgb.
        """
        self.img = self._load_image()
        
        b_channel, g_channel, r_channel = self._crop_and_divide_image()

        # Align Green and Blue channels to the Red channel using gradient descent
        aligned_g, displacement_vectors = self._align_channel(r_channel, g_channel)
        aligned_b, displacement_vectors = self._align_channel(r_channel, b_channel)

        aligned_r = r_channel

        # Stack the channels to create the final RGB image
        self.rgb = torch.stack([aligned_r, aligned_g, aligned_b], dim=0)

        for i, (dx, dy) in enumerate(displacement_vectors):
            print(f"Iteration {i + 1}: delta_x = {dx}, delta_y = {dy}")

    def _load_image(self):
        """Load the image from the image_name path,
           typecast it to float, and normalize it.

           Returns: torch.Tensor of shape (C, H, W)
        """
        ret = None
        image = Image.open(self.image_name)

        # Convert to tensor and then to float.
        ret = transforms.ToTensor()(image).float()

        return ret

    def _crop_and_divide_image(self):
        """Crop the image boundary and divide the image into three parts, padded to the same size.

           Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
        """
        height, width = self.img.shape[-2:]

        # Remove the border by cropping a fixed number of pixels.
        cropped_img = self.img[:, 10:height - 10, 10:width - 10]

        # Divide the image into three parts
        h_third = cropped_img.shape[1] // 3

        b_channel = cropped_img[:, :h_third]
        g_channel = cropped_img[:, h_third:2 * h_third]
        r_channel = cropped_img[:, 2 * h_third:]

        # Adjust for any remainder by padding the channels to match the largest one
        max_height = max(b_channel.shape[1], g_channel.shape[1], r_channel.shape[1])

        # Padding each channel to match the max height
        if b_channel.shape[1] < max_height:
            pad_size = max_height - b_channel.shape[1]
            b_channel = torch.cat([b_channel, torch.zeros((b_channel.shape[0], pad_size, b_channel.shape[2]), dtype=b_channel.dtype)], dim=1)
        if g_channel.shape[1] < max_height:
            pad_size = max_height - g_channel.shape[1]
            g_channel = torch.cat([g_channel, torch.zeros((g_channel.shape[0], pad_size, g_channel.shape[2]), dtype=g_channel.dtype)], dim=1)
        if r_channel.shape[1] < max_height:
            pad_size = max_height - r_channel.shape[1]
            r_channel = torch.cat([r_channel, torch.zeros((r_channel.shape[0], pad_size, r_channel.shape[2]), dtype=r_channel.dtype)], dim=1)

        # Print shapes for debugging
        print(f"cropped_img shape: {cropped_img.shape}")
        print(f"b_channel shape: {b_channel.shape}")
        print(f"g_channel shape: {g_channel.shape}")
        print(f"r_channel shape: {r_channel.shape}")

        g_channel_resized = self._remove_redundant_dimensions(g_channel)
        r_channel_resized = self._remove_redundant_dimensions(r_channel)
        b_channel_resized = self._remove_redundant_dimensions(b_channel)
        
        return b_channel_resized, g_channel_resized, r_channel_resized

    def _remove_redundant_dimensions(self, img_channel):

        # Reference:- https://stackoverflow.com/questions/68079012/valueerror-win-size-exceeds-image-extent-if-the-input-is-a-multichannel-color
        # Using numpy.squeeze for removing the redundant dimension: 
        img_channel = np.squeeze(img_channel)

        return img_channel


    
    def _align_channel(self, reference_channel, source_channel):
        delta_x = torch.tensor(0.0, requires_grad=True)
        delta_y = torch.tensor(0.0, requires_grad=True)
        lr = 0.01  # Learning rate
        num_iterations = 100  # Number of gradient descent steps
        # Store displacement vectors
        displacement_vectors = []

        # Added a batch and channel dimension to the input tensors to requirement of a 4D tensor for F.affine_grid 
        # to be used for Assignment 2.
        reference_channel = reference_channel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        source_channel = source_channel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        optimizer = torch.optim.SGD([delta_x, delta_y], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # create the affine transformation matrix
            theta = torch.tensor([[1, 0, delta_x], [0, 1, delta_y]], dtype=torch.float32, requires_grad=True)

            # affine_grid
            grid = F.affine_grid(theta.unsqueeze(0), source_channel.size(), align_corners=False)

            # Apply grid_sample
            transformed_image = F.grid_sample(source_channel, grid, align_corners=False)

            # Store the displacement vectors
            displacement_vectors.append((delta_x.item(), delta_y.item()))

            # Zero the gradients after updating
            delta_x.grad = None
            delta_y.grad = None

            loss = None

            # Compute the loss.
            if self.metric == 'ncc':
                loss = ncc(reference_channel.squeeze(1), transformed_image.squeeze(1))

            else:
                loss = mse(reference_channel.squeeze(1), transformed_image.squeeze(1))


            # Backward pass
            loss.backward()

            optimizer.step()

        # After the optimization step, apply the final shifts
        theta_final = torch.tensor([[1, 0, delta_x.item()], [0, 1, delta_y.item()]], dtype=torch.float32)
        final_grid = F.affine_grid(theta_final.unsqueeze(0), source_channel.size(), align_corners=False)
        aligned_image = F.grid_sample(source_channel, final_grid, align_corners=False)

        return aligned_image.squeeze(0).squeeze(0), displacement_vectors  # Remove batch and channel dimensions





