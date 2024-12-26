import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_1d_signature(binary_image):
    """
    Computes the 1D signature of the object in a binary image by calculating the radial distances
    from the centroid to the object's boundary at various angles.

    Parameters:
        binary_image (numpy.ndarray): A 2D binary image (values should be 0 or 1).

    Returns:
        tuple: A tuple containing:
            - angles (numpy.ndarray): Array of angles in radians.
            - distances (numpy.ndarray): Array of distances for each angle.
    """
    # Get the coordinates of the object's pixels
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

    return angles, distances

def plot_1d_signatures_with_images(angles_list, distances_list, images, image_names):
    """
    Plots multiple 1D signatures of the objects alongside their binary images.

    Parameters:
        angles_list (list of numpy.ndarray): List of angles arrays for each image.
        distances_list (list of numpy.ndarray): List of distances arrays for each image.
        images (list of numpy.ndarray): List of binary images.
        image_names (list of str): List of image names being processed.
    """
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

def process_images_for_signature(image_paths, output_dir):
    """
    Processes multiple binary images to compute and save their 1D signatures alongside images.

    Parameters:
        image_paths (list of str): Paths to the binary images.
        output_dir (str): Directory to save the signature plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    angles_list = []
    distances_list = []
    images = []
    image_names = []

    for image_path in image_paths:
        # Load the image
        image = Image.open(image_path).convert('L')
        binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

        # Compute 1D signature
        angles, distances = compute_1d_signature(binary_image)

        # Append results for combined plotting
        angles_list.append(angles)
        distances_list.append(distances)
        images.append(binary_image)
        image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
        image_names.append(image_name)

    # Plot all signatures with images together
    plot_1d_signatures_with_images(angles_list, distances_list, images, image_names)

if __name__ == "__main__":
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
    output_dir = "signatures"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    process_images_for_signature(image_paths, output_dir)
