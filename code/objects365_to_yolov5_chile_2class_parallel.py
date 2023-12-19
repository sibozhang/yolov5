import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from PIL import Image  # Import PIL for image dimension and integrity check

BASE_DIR = "/mnt/scratch/sibo/datasets/Objects365"

def process_label_file(label_input_address, label_output_address, image_input_address, category_map, label_file):
    src_file_path = os.path.join(label_input_address, label_file)
    dest_file_path = os.path.join(label_output_address, label_file)

    image_file_path = os.path.join(image_input_address, os.path.splitext(label_file)[0] + ".jpg")

    # Check if the corresponding image file exists
    if not os.path.exists(image_file_path):
        return None

    # Check if the file is overly large (e.g., more than 10MB)
    if os.path.getsize(image_file_path) > 10 * 1024 * 1024:  # 10MB
        return None
    
    try:
        # Check image dimensions and if the image is corrupt
        with Image.open(image_file_path) as img:
            width, height = img.size
            if width > 1280 or height > 720:
                return None
    except Exception as e:
        print(f"Could not open image {image_file_path}. Error: {e}")
        return None

    lines_to_write = set()

    with open(src_file_path, encoding="utf8", errors='ignore') as f:
        for row_content in f:
            parts = row_content.split()
            cat_id = int(parts[0])
            new_cat_id = category_map.get(cat_id)
            if new_cat_id is not None:
                lines_to_write.add(f"{new_cat_id} {' '.join(parts[1:])}\n")

    if lines_to_write:
        with open(dest_file_path, "w+") as t:
            t.writelines(lines_to_write)
        return label_file
    return None

def process_data(mode):
    non_empty_label_files = set()

    paths = {
        "label_input": os.path.join(BASE_DIR, f"labels_all/{mode}/"),
        "label_output": os.path.join(BASE_DIR, f"labels_2class/{mode}/"),
        "image_input": os.path.join(BASE_DIR, f"images/{mode}/")
    }

    label_input_address = paths["label_input"]
    label_output_address = paths["label_output"]
    image_input_address = paths["image_input"]

    os.makedirs(label_output_address, exist_ok=True)

    label_files = [file for file in os.listdir(label_input_address) if file.endswith('.txt')]

    category_map = {
        0: 0,
        61: 3,
    }

    # Set max_workers to 40 to use 40 CPUs
    with ProcessPoolExecutor(max_workers=40) as executor:
        func = partial(process_label_file, label_input_address, label_output_address, image_input_address, category_map)
        for index, label_file in enumerate(executor.map(func, label_files)):
            if label_file:
                non_empty_label_files.add(label_file)
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} label files for mode: {mode}.")

    output_file_path = os.path.join(BASE_DIR, f"{mode}_chile_2class.txt")

    with open(output_file_path, 'w') as file:
        for label_file in non_empty_label_files:
            file_name = os.path.join(f"./images/{mode}/", os.path.splitext(label_file)[0] + ".jpg")
            file.write(file_name + '\n')

if __name__ == "__main__":
    process_data("train")
    process_data("val")
