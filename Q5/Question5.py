import cv2
import numpy as np
from PIL import Image

def calculate_compactness(binary_image):
    """
    Computes the compactness of the object in a binary image.

    Parameters:
        binary_image (numpy.ndarray): A 2D binary image (values should be 0 or 1).

    Returns:
        float: The compactness of the object.
    """
    # Convert binary image to uint8 type for OpenCV compatibility
    binary_image_uint8 = (binary_image * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # No object found

    # Calculate area and perimeter of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calculate compactness
    if perimeter == 0:
        return None  # Avoid division by zero

    compactness = (4 * np.pi * area) / (perimeter ** 2)
    return compactness

def process_images_for_compactness(image_paths, output_file):
    """
    Processes multiple images to calculate and save their compactness.

    Parameters:
        image_paths (list of str): Paths to the binary images.
        output_file (str): Path to the file where compactness values will be saved.
    """
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            # Load the image
            image = Image.open(image_path).convert('L')
            binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

            # Calculate compactness
            compactness = calculate_compactness(binary_image)

            # Save compactness to file
            image_name = image_path.split('/')[-1]  # Extract image name
            f.write(f"Compactness of {image_name}: {compactness}\n")
            print(f"Compactness for {image_name} calculated: {compactness}")

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
    output_file = "compactness.csv"

    # Process images and calculate compactness
    process_images_for_compactness(image_paths, output_file)

    print(f"Compactness values for all images saved to {output_file}.")
