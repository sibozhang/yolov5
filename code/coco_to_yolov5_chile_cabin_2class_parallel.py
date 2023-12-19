import os
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = "/mnt/data/sibo/coco/"

def process_label_file(label_file, label_input_address, label_output_address, category_map):
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

def process_data(MODE):
    paths = {
        mode: {
            "label_input": os.path.join(BASE_DIR, f"labels_all/{mode}2017/"),
            "label_output": os.path.join(BASE_DIR, f"labels/{mode}2017/")
        }
        for mode in ["train", "val"]
    }

    label_input_address = paths[MODE]["label_input"]
    label_output_address = paths[MODE]["label_output"]
    category_map = {0: 0, 67: 1}  # person, cell phone

    if not os.path.exists(label_output_address):
        os.makedirs(label_output_address, exist_ok=True)

    label_files = [file for file in os.listdir(label_input_address) if file.endswith('.txt')]

    with ProcessPoolExecutor() as executor:
        for index, _ in enumerate(executor.map(process_label_file, label_files, 
                                               [label_input_address] * len(label_files), 
                                               [label_output_address] * len(label_files), 
                                               [category_map] * len(label_files))):
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} label files for mode: {MODE}.")

    output_file_path = f"{MODE}2017_chile_2class.txt"
    with open(output_file_path, 'w') as file:
        for label_file in label_files:
            file_name = os.path.join(f"./images/{MODE}2017/", os.path.splitext(label_file)[0] + ".jpg")
            file.write(file_name + '\n')

if __name__ == "__main__":
    process_data("train")
    process_data("val")

# import os
# from concurrent.futures import ProcessPoolExecutor

# BASE_DIR = "/mnt/data/sibo/coco/"

# def process_data(MODE):
#     paths = {
#         mode: {
#             "label_input": os.path.join(BASE_DIR, f"labels_all/{mode}2017/"),
#             "label_output": os.path.join(BASE_DIR, f"labels/{mode}2017/")
#         }
#         for mode in ["train", "val"]
#     }

#     label_input_address = paths[MODE]["label_input"]
#     label_output_address = paths[MODE]["label_output"]

#     if not os.path.exists(label_output_address):
#         os.mkdir(label_output_address)

#     label_files = [file for file in os.listdir(label_input_address) if file.endswith('.txt')]

#     category_map = {
#         0: 0,  # person
#         67: 1  # cell phone
#     }

#     def process_label_file(label_file):
#         source_file_path = os.path.join(label_input_address, label_file)
#         destination_file_path = os.path.join(label_output_address, label_file)

#         valid_class_found = False
#         with open(source_file_path, encoding="utf8", errors='ignore') as f:
#             lines_to_write = []
#             for row_content in f:
#                 row_parts = row_content.split()
#                 cat_id = int(row_parts[0])
#                 new_cat_id = category_map.get(cat_id)
#                 if new_cat_id is not None:
#                     valid_class_found = True
#                     lines_to_write.append(f"{new_cat_id} {' '.join(row_parts[1:])}\n")

#             if valid_class_found:
#                 with open(destination_file_path, "w+") as t:
#                     t.writelines(lines_to_write)

#     with ProcessPoolExecutor() as executor:
#         for index, _ in enumerate(executor.map(process_label_file, label_files)):
#             if (index + 1) % 100 == 0:
#                 print(f"Processed {index + 1} label files for mode: {MODE}.")

#     output_file_path = f"{MODE}2017_chile_2class.txt"

#     with open(output_file_path, 'w') as file:
#         for label_file in label_files:
#             file_name = os.path.join(f"./images/{MODE}2017/", os.path.splitext(label_file)[0] + ".jpg")
#             file.write(file_name + '\n')

# if __name__ == "__main__":
#     process_data("train")
#     process_data("val")
