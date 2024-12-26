import numpy as np
from skimage.morphology import medial_axis
from skimage.util import invert
from PIL import Image
import matplotlib.pyplot as plt
import os

# Paths
image_paths = [
    "../1.PNG",
    "../2.PNG",
    "../3.PNG",
    "../4.PNG",
    "../5.PNG",
    "../6.PNG",
    "../7.PNG",
    "../8.PNG",
    "../9.PNG",
    "../10.PNG",
]
output_dir = "medial_axis_output"

# Process images and compute medial axis
for image_path in image_paths:
    # Load the image
    image = Image.open(image_path).convert('L')
    binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

    # Compute medial axis
    binary_image = binary_image.astype(bool)

    # Compute medial axis
    medial_axis_image, _ = medial_axis(binary_image, return_distance=True)

    # Plot and save original image and medial axis side by side
    image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
    output_path = os.path.join(output_dir, f"{image_name}_medial_axis.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(binary_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(medial_axis_image, cmap='gray')
    axes[1].set_title("Medial Axis")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Medial axis for {image_name} saved to {output_path}")

print(f"Medial axis images for all images saved to {output_dir}.")
