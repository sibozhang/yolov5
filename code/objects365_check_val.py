from PIL import Image
import os

val_dir = "/mnt/scratch/sibo/datasets/Objects365/images/val"
output_file_path = "/mnt/scratch/sibo/datasets/Objects365/images/val.txt"

# Initialize counter for monitoring progress
counter = 0

# Open the output file in write mode
with open(output_file_path, "w") as f:
    # Loop through each file in the val directory
    for filename in os.listdir(val_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more formats if needed
            image_path = os.path.join(val_dir, filename)

            # Open the image using PIL
            with Image.open(image_path) as img:
                # Get dimensions
                width, height = img.size

            # Write dimensions to output file
            f.write(f"{filename},{width},{height}\n")

            # Increment counter
            counter += 1

            # Print progress every 1000 images
            if counter % 1000 == 0:
                print(f"{counter} images processed.")

print("Done. Image dimensions have been saved to val.txt.")
