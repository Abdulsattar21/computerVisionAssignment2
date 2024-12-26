import cv2
import numpy as np
from PIL import Image
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
output_file = "compactness.txt"


with open(output_file, 'w') as f:
    for image_path in image_paths:
        # Load the image
        image = Image.open(image_path).convert('L')
        binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

        # Calculate compactness
        binary_image_uint8 = (binary_image * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate area and perimeter of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        compactness = (4 * np.pi * area) / (perimeter ** 2)
        # Save compactness to file
        image_name = image_path.split('/')[-1]  # Extract image name
        f.write(f"Compactness of {image_name}: {compactness}\n")
        print(f"Compactness for {image_name} calculated: {compactness}")

print(f"Compactness values for all images saved to {output_file}.")




