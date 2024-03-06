import os
import itertools
from concurrent.futures import ProcessPoolExecutor
import logging

def process_label_file(label_file, label_input_address, label_output_address, category_map):
    """
    Process a single label file and write filtered content to a new file.
    """
    source_file_path = os.path.join(label_input_address, label_file)
    destination_file_path = os.path.join(label_output_address, label_file)

    valid_class_found = False
    with open(source_file_path, encoding="utf8", errors='ignore') as f:
        lines_to_write = []
        for row_content in f:
            row_parts = row_content.split()
            cat_id = int(row_parts[0])
            new_cat_id = category_map.get(cat_id)
            if new_cat_id is not None:
                valid_class_found = True
                lines_to_write.append(f"{new_cat_id} {' '.join(row_parts[1:])}\n")

        if valid_class_found:
            with open(destination_file_path, "w+") as t:
                t.writelines(lines_to_write)

def write_output_file(mode, base_dir, label_output_address):
    """
    Write a file listing the paths of processed label files.
    """
    #modify this to change the output file name
    output_file_path = os.path.join(base_dir, f"{mode}2017_chile_3class.txt")
    label_files = [file for file in os.listdir(label_output_address) if file.endswith('.txt')]

    with open(output_file_path, 'w') as file:
        for label_file in sorted(label_files):
            file_name = os.path.join(base_dir, f"images/{mode}2017/", os.path.splitext(label_file)[0] + ".jpg")
            file.write(file_name + '\n')

def process_data(mode, base_dir, only_write=False):
    """
    Process label files for a given mode (train/val) in sorted order 
    and create a file listing image paths. If only_write is True, only write the output file.
    """
    #modify this to change the input and output directories
    paths = {
        mode: {
            "label_input": os.path.join(base_dir, f"labels_all/{mode}2017/"),
            "label_output": os.path.join(base_dir, f"labels/{mode}2017/")
        }
        for mode in ["train", "val"]
    }

    label_input_address = paths[mode]["label_input"]
    label_output_address = paths[mode]["label_output"]
    category_map = {0: 0}  # person

    if only_write:
        write_output_file(mode, base_dir, label_output_address)
        return

    if not os.path.exists(label_output_address):
        os.makedirs(label_output_address, exist_ok=True)
        logging.info(f"Created directory: {label_output_address}")

    label_files = sorted([file for file in os.listdir(label_input_address) if file.endswith('.txt')])

    with ProcessPoolExecutor() as executor:
        for index, _ in enumerate(executor.map(process_label_file, label_files, 
                                               itertools.repeat(label_input_address), 
                                               itertools.repeat(label_output_address), 
                                               itertools.repeat(category_map))):
            if (index + 1) % 1000 == 0:
                logging.info(f"Processed {index + 1} label files for mode: {mode}.")

    write_output_file(mode, base_dir, label_output_address)

if __name__ == "__main__":
    BASE_DIR = "/mnt/data/sibo/coco/"
    logging.basicConfig(level=logging.INFO)
    process_data("train", BASE_DIR, only_write=True)
    # process_data("val", BASE_DIR, only_write=False)
    logging.info(f"Files written to: {BASE_DIR}")
