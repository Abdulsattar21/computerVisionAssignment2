import numpy as np
from PIL import Image

def run_length_encoding(binary_image):
    """
    Applies Run-Length Encoding (RLE) to a binary image.

    Parameters:
        binary_image (numpy.ndarray): A 2D binary image (values should be 0 or 1).

    Returns:
        list: A list of tuples representing the RLE.
    """
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
    return rle

def save_rle_to_text_file(rle, output_path):
    """
    Writes the RLE data to a text file.

    Parameters:
        rle (list): The Run-Length Encoding of the binary image.
        output_path (str): The path to the output text file.
    """
    with open(output_path, 'w') as f:
        for value, count in rle:
            f.write(f"{value} {count}\n")

def process_images_with_rle(image_paths, output_dir):
    """
    Processes multiple binary images, applies RLE, and saves the results to text files.

    Parameters:
        image_paths (list of str): Paths to the binary images.
        output_dir (str): Directory to save the RLE text files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        # Load the image
        image = Image.open(image_path).convert('L')
        binary_image = (np.array(image) > 23).astype(int)  # Convert to binary (thresholding)

        # Apply RLE
        rle = run_length_encoding(binary_image)

        # Save RLE to a text file
        image_name = image_path.split('/')[-1].split('.')[0]  # Extract image name without extension
        output_path = os.path.join(output_dir, f"{image_name}_rle.txt")
        save_rle_to_text_file(rle, output_path)
        print(f"RLE saved to {output_path}")

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
    process_images_with_rle(image_paths, output_dir)
    # Paths
    rle_dir = "rle_output"
    output_dir = "reconstructed_images"

    # Define the dimensions of the binary images
    width = 768  # Replace with the actual width
    height = 576  # Replace with the actual height

    # Process RLE files and save reconstructed images
    process_rle_files_to_images(rle_dir, output_dir, width, height)
