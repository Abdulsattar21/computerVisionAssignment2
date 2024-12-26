import numpy as np
from PIL import Image
import os

def rle_to_binary_image(rle, width, height):

    binary_image = np.zeros((height, width), dtype=int).flatten()
    idx = 0
    for value, count in rle:
        binary_image[idx:idx+count] = value
        idx += count
    return binary_image.reshape((height, width))

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

output_dir = "rle_output"

# Process images and save RLE
for image_path in image_paths:
    # Load the image
    image = Image.open(image_path)
    binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

    # Apply RLE
    rle = []
    for row in binary_image:
        value = row[0]
        count = 0
        for pixel in row:
            if pixel == value:
                count += 1
            else:
                rle.append((value, count))
                value = pixel
                count = 1
        rle.append((value, count))  # Add the last run

    # Save RLE to a text file
    image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
    output_path = os.path.join(output_dir, f"{image_name}_rle.txt")
    with open(output_path, 'w') as f:
        for value, count in rle:
            f.write(f"{value} {count}\n")
    print(f"RLE saved to {output_path}")


# creating photos
rle_dir = "rle_output"
output_dir = "reconstructed_images"

width = 768
height = 576

# Process RLE files and save reconstructed images
for rle_file in os.listdir(rle_dir):
    if rle_file.endswith("_rle.txt"):
        input_path = os.path.join(rle_dir, rle_file)
        rle = []
        with open(input_path, 'r') as f:
            for line in f:
                value, count = map(int, line.strip().split())
                rle.append((value, count))
        binary_image = rle_to_binary_image(rle, width, height)

        # Save the reconstructed binary image
        image_name = rle_file.replace("_rle.txt", "_reconstructed.png")
        output_path = os.path.join(output_dir, image_name)
        Image.fromarray((binary_image * 255).astype(np.uint8)).save(output_path)
        print(f"Reconstructed binary image saved to {output_path}")
