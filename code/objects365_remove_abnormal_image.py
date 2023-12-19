import glob
import os
import shutil
import multiprocessing
from PIL import Image, ImageFile

counter = None  # Will be set in init()
lock = None  # Will be set in init()

def init(args):
    global counter, lock
    counter, lock = args

def check_abnormal_image(image_path):
    global counter, lock
    reason = None
    file_size = None

    # Check for empty files
    if os.path.getsize(image_path) == 0:
        reason = 'Empty'

    try:
        # Check for corrupted images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(image_path) as img:
            # Check for overly large images
            file_size = img.size[0] * img.size[1]
            if file_size > Image.MAX_IMAGE_PIXELS:
                reason = f'Overly Large, Size: {file_size}'

            img.verify()

    except Exception as e:
        reason = 'Corrupted'

    # If abnormal, move the image and log reason
    if reason:
        dest_path = f"/mnt/scratch/sibo/datasets/Objects365/images/abnormal_images/{os.path.basename(image_path)}"
        shutil.move(image_path, dest_path)
        with open('/mnt/scratch/sibo/datasets/Objects365/images/abnormal_images.txt', 'a') as f:
            f.write(f"{os.path.basename(image_path)}\t{reason}\n")

    with lock:
        counter.value += 1
        if counter.value % 1000 == 0:
            print(f"Checked {counter.value} images.")

if __name__ == '__main__':
    # Ensure the abnormal_images folder exists
    os.makedirs('/mnt/scratch/sibo/datasets/Objects365/images/abnormal_images', exist_ok=True)

    paths = glob.glob('/mnt/scratch/sibo/datasets/Objects365/images/train/*.jpg')

    # Number of processes to spawn
    num_processes = multiprocessing.cpu_count()

    # Shared counter and lock
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    # Create a multiprocessing Pool and map the function to the paths
    with multiprocessing.Pool(num_processes, initializer=init, initargs=((counter, lock),)) as pool:
        pool.map(check_abnormal_image, paths)
