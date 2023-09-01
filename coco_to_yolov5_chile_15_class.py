import os
import shutil

MODE = "val"  # Change to "train" for training data

# Dictionary to hold paths based on mode
paths = {
    "train": {
        "label_input": "/mnt/data/sibo/coco/labels_all/train2017/",
        "label_output": "/mnt/data/sibo/coco/labels/train2017/",
    },
    "val": {
        "label_input": "/mnt/data/sibo/coco/labels_all/val2017/",
        "label_output": "/mnt/data/sibo/coco/labels/val2017/",
    }
}

label_input_address = paths[MODE]["label_input"]
label_output_address = paths[MODE]["label_output"]

# Check and create the output directories if they do not exist
for address in [label_output_address, image_output_address]:
    if not os.path.exists(address):
        os.mkdir(address)

label_files = [file for file in os.listdir(label_input_address) if file.endswith('.txt')]

# Define a dictionary for category mapping
category_map = {
    0: 0,  # person
    2: 1,  # car
    7: 2,  # truck
    67: 3  # cell phone
}

for label_file in label_files:
    source_file_path = os.path.join(label_input_address, label_file)
    destination_file_path = os.path.join(label_output_address, label_file)

    # Use a flag to track if we've found a valid class_id
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

        # Only write to the output label directory if a valid class_id was found in the input
        if valid_class_found:
            with open(destination_file_path, "w+") as t:
                t.writelines(lines_to_write)

            # Now, copy the corresponding image file if the label file was written
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            source_image_path = os.path.join(image_input_address, image_name)
            destination_image_path = os.path.join(image_output_address, image_name)
            
            # Check if the image exists in the source directory before copying
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
