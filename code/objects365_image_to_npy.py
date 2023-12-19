from PIL import Image
import numpy as np
import os

MAX_SIZE = (1280, 720)  # Maximum allowed image size (width, height)

def image_to_npy(image_path, save_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # Check for non-empty image
        if img_array.size == 0:
            return

        # Check for oversized image
        if img_array.shape[0] > MAX_SIZE[1] or img_array.shape[1] > MAX_SIZE[0]:
            return

        np.save(save_path, img_array)
        
    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")

if __name__ == '__main__':
    image_folder = '/mnt/scratch/sibo/datasets/Objects365/images/train_original'
    save_folder = '/mnt/scratch/sibo/datasets/Objects365/images/train'
    
    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_file)
            save_path = os.path.join(save_folder, os.path.splitext(img_file)[0] + '.npy')
            
            image_to_npy(img_path, save_path)
