import numpy as np
from PIL import Image

def load_rle_from_text_file(input_path):
    """
    Reads RLE data from a text file.

    Parameters:
        input_path (str): The path to the input text file.

    Returns:
        list: The Run-Length Encoding data.
    """
    rle = []
    with open(input_path, 'r') as f:
        for line in f:
            value, count = map(int, line.strip().split())
            rle.append((value, count))
    return rle

def rle_to_binary_image(rle, width, height):
    """
    Converts RLE data back to a binary image.

    Parameters:
        rle (list): The Run-Length Encoding of the binary image.
        width (int): The width of the binary image.
        height (int): The height of the binary image.

    Returns:
        numpy.ndarray: The reconstructed binary image.
    """
    binary_image = np.zeros((height, width), dtype=int).flatten()
    idx = 0
    for value, count in rle:
        binary_image[idx:idx+count] = value
        idx += count
    return binary_image.reshape((height, width))

def process_rle_files_to_images(rle_dir, output_dir, width, height):
    """
    Processes RLE text files to reconstruct and save binary images.

    Parameters:
        rle_dir (str): Directory containing the RLE text files.
        output_dir (str): Directory to save the reconstructed binary images.
        width (int): The width of the binary images.
        height (int): The height of the binary images.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for rle_file in os.listdir(rle_dir):
        if rle_file.endswith("_rle.txt"):
            input_path = os.path.join(rle_dir, rle_file)
            rle = load_rle_from_text_file(input_path)
            binary_image = rle_to_binary_image(rle, width, height)

            # Save the reconstructed binary image
            image_name = rle_file.replace("_rle.txt", "_reconstructed.png")
            output_path = os.path.join(output_dir, image_name)
            Image.fromarray((binary_image * 255).astype(np.uint8)).save(output_path)
            print(f"Reconstructed binary image saved to {output_path}")

if __name__ == "__main__":
    import os

    # Paths
    rle_dir = "rle_output"
    output_dir = "reconstructed_images"

    # Define the dimensions of the binary images
    width = 100  # Replace with the actual width
    height = 100  # Replace with the actual height

    # Process RLE files and save reconstructed images
    process_rle_files_to_images(rle_dir, output_dir, width, height)
