import cv2
import numpy as np
from PIL import Image

def calculate_hu_moments(binary_image):
    """
    Calculates the Hu Moments of a binary image.

    Parameters:
        binary_image (numpy.ndarray): A 2D binary image (values should be 0 or 1).

    Returns:
        numpy.ndarray: An array of seven Hu Moments.
    """
    # Convert binary image to uint8 type for OpenCV compatibility
    binary_image_uint8 = (binary_image * 255).astype(np.uint8)

    # Calculate moments
    moments = cv2.moments(binary_image_uint8)

    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()

    return hu_moments

def process_images_for_hu_moments(image_paths, output_file):
    """
    Processes multiple images to calculate and save their Hu Moments.

    Parameters:
        image_paths (list of str): Paths to the binary images.
        output_file (str): Path to the file where Hu Moments will be saved.
    """
    with open(output_file, 'w') as f:
        f.write("Image Name,Hu1,Hu2,Hu3,Hu4,Hu5,Hu6,Hu7\n")
        for image_path in image_paths:
            # Load the image
            image = Image.open(image_path).convert('L')
            binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

            # Calculate Hu Moments
            hu_moments = calculate_hu_moments(binary_image)

            # Save Hu Moments to file
            image_name = image_path.split('/')[-1]  # Extract image name
            hu_moments_str = ",".join(map(str, hu_moments))
            f.write(f"{image_name},{hu_moments_str}\n")
            print(f"Hu Moments for {image_name} saved.")


if __name__ == "__main__":
    import os

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
    output_file = "hu_moments.csv"

    # Process images and calculate Hu Moments
    process_images_for_hu_moments(image_paths, output_file)

    print(f"Hu Moments for all images saved to {output_file}.")
