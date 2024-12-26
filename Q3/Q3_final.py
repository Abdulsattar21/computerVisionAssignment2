import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
output_dir = "signatures"


# Process images
angles_list = []
distances_list = []
images = []
image_names = []

for image_path in image_paths:
    # Load the image
    image = Image.open(image_path).convert('L')
    binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

    # Compute 1D signature
    object_coords = np.column_stack(np.where(binary_image == 1))

    # Calculate the centroid of the object
    centroid = object_coords.mean(axis=0)

    # Define angles (0 to 2*pi)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    distances = np.zeros_like(angles)

    for i, angle in enumerate(angles):
        # Compute the unit vector for the current angle
        direction = np.array([np.cos(angle), np.sin(angle)])

        # Project the object's coordinates onto the direction vector
        projections = np.dot(object_coords - centroid, direction)

        # Get the maximum projection (distance to the boundary)
        distances[i] = projections.max()

    # Append results for combined plotting
    angles_list.append(angles)
    distances_list.append(distances)
    images.append(binary_image)
    image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
    image_names.append(image_name)

# Plot all signatures with images together
fig, axes = plt.subplots(len(image_names), 2, figsize=(12, len(image_names) * 4))
if len(image_names) == 1:
    axes = [axes]  # Ensure axes is iterable for a single image

for ax_row, angles, distances, image, image_name in zip(axes, angles_list, distances_list, images, image_names):
    # Plot the binary image
    ax_row[0].imshow(image, cmap='gray')
    ax_row[0].set_title(f"{image_name} Binary Image")
    ax_row[0].axis('off')

    # Plot the signature
    ax_row[1].plot(angles, distances)
    ax_row[1].set_title(f"{image_name} Signature")
    ax_row[1].set_xlabel("Angle (θ) [radians]")
    ax_row[1].set_ylabel("Distance from center (r)")
    ax_row[1].grid(True)

plt.tight_layout()
plt.savefig("signatures_with_images_combined.png")
plt.show()

