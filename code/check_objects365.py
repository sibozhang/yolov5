import os
import numpy as np

BASE_DIR = "/mnt/scratch/sibo/datasets/Objects365"
LABEL_DIR = os.path.join(BASE_DIR, "labels/train/")
IMAGE_DIR = os.path.join(BASE_DIR, "images/train/")

# List all .npy files in the image directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')]

counter = 0  # Initialize counter variable

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)

    try:
        # Load the image data
        im = np.load(image_path)
        
        # Reshape or perform whatever operation that might trigger the error
        # For example, here I am assuming that you are trying to reshape it to (1024, 1007, 3)
        # Replace this shape with the shape you are expecting
        im = im.reshape((1024, 1007, 3))

    except ValueError as e:
        print(f"Error occurred while reshaping image file: {image_path}")
        print(f"Exception: {e}")

    # Increment counter
    counter += 1

    # Print message every 1000 files processed
    if counter % 1000 == 0:
        print(f"{counter} files processed.")

    # Optionally, you can break here to stop at the first error encountered
    # break
