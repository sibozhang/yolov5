import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

BASE_DIR = "/mnt/data/sibo/coco/"

def process_label_file(label_input_address, label_output_address, category_map, label_file):
    src_file_path = os.path.join(label_input_address, label_file)
    dest_file_path = os.path.join(label_output_address, label_file)

    lines_to_write = []

    with open(src_file_path, encoding="utf8", errors='ignore') as f:
        for row_content in f:
            parts = row_content.split()
            cat_id = int(parts[0])
            new_cat_id = category_map.get(cat_id)
            if new_cat_id is not None:
                lines_to_write.append(f"{new_cat_id} {' '.join(parts[1:])}\n")

    if lines_to_write:
        with open(dest_file_path, "w+") as t:
            t.writelines(lines_to_write)

def process_data(mode):
    paths = {
        m: {
            "label_input": os.path.join(BASE_DIR, f"labels_all/{m}2017/"),
            "label_output": os.path.join(BASE_DIR, f"labels/{m}2017/")
        }
        for m in ["train", "val"]
    }

    label_input_address = paths[mode]["label_input"]
    label_output_address = paths[mode]["label_output"]

    os.makedirs(label_output_address, exist_ok=True)

    label_files = [file for file in os.listdir(label_input_address) if file.endswith('.txt')]

    category_map = {
        0: 0,  # person
        2: 1,  # car
        7: 2,  # truck
        67: 3  # cell phone
    }

    with ProcessPoolExecutor() as executor:
        func = partial(process_label_file, label_input_address, label_output_address, category_map)
        for index, _ in enumerate(executor.map(func, label_files)):
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} label files for mode: {mode}.")

    output_file_path = os.path.join(BASE_DIR, f"{mode}2017_chile_4class.txt")

    with open(output_file_path, 'w') as file:
        for label_file in label_files:
            file_name = os.path.join(f"./images/{mode}2017/", os.path.splitext(label_file)[0] + ".jpg")
            file.write(file_name + '\n')

if __name__ == "__main__":
    process_data("train")
    process_data("val")
