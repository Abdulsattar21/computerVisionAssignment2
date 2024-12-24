import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

for image_path in image_paths:
    # Load the image
    image = Image.open(image_path)
    binary_image = np.array(image) > 23  # Convert to binary (thresholding)

    # Don't forget to mention the threshold value in the report

    # Compute projections+
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    # Plot the results
    image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
    """
    Plots the binary image alongside its horizontal and vertical projections.

    Parameters:
        binary_image (numpy.ndarray): The binary image.
        horizontal_projection (numpy.ndarray): Horizontal projection array.
        vertical_projection (numpy.ndarray): Vertical projection array.
        image_name (str): Name of the image being processed.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the binary image
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title(f'Binary Image - {image_name}')
    ax1.axis('off')

    # Plot the horizontal projection
    ax2.plot(horizontal_proj, range(len(horizontal_proj)))
    ax2.invert_yaxis()
    ax2.set_title('Horizontal Projection')
    ax2.set_xlabel('Sum of pixels')
    ax2.set_ylabel('Rows')

    # Plot the vertical projection
    ax3.plot(range(len(vertical_proj)), vertical_proj)
    ax3.set_title('Vertical Projection')
    ax3.set_xlabel('Columns')
    ax3.set_ylabel('Sum of pixels')

    plt.tight_layout()
    plt.savefig(f"{image_name}_projections.png")
    plt.show()


