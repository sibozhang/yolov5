import os

def modify_files_in_directory(input_directory_path, output_directory_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

    # Create a mapping for class IDs
    class_mapping = {
        0: 0, # 'person' to 'person'
        1: 1, # 'car' to 'car'
        2: 2, # 'truck' to 'truck'
        7: 3, # 'cone' to 'cone'
        11: 4, # 'helmet' to 'helmet'
        4: 5, # 'excavator' to 'machinery vehicle'
        5: 5, # 'loader' to 'machinery vehicle'
        6: 5, # 'crane' to 'machinery vehicle'
        9: 5  # 'shovel' to 'machinery vehicle'
        8: 6, # 'hook' to 'hook'
    }

    # Set of classes to remove
    classes_to_remove = {3, 10, 12, 13, 14}  # 'cell phone', 'payload', 'bar', 'rope', 'barrier'

    # Iterate over all files in the directory
    for filename in os.listdir(input_directory_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            # Full file path
            input_file_path = os.path.join(input_directory_path, filename)
            output_file_path = os.path.join(output_directory_path, filename)

            # Read the file
            with open(input_file_path, 'r') as f:
                lines = f.readlines()

            # Modify the file
            with open(output_file_path, 'w') as f:
                for line in lines:
                    parts = line.split(' ')
                    cat_id = int(parts[0])

                    # Skip classes to remove
                    if cat_id in classes_to_remove:
                        continue

                    # Map the class IDs
                    new_cat_id = class_mapping.get(cat_id, cat_id)

                    # Write the line if not in classes to remove
                    f.write(f'{new_cat_id} {" ".join(parts[1:])}')

    print('Writing modified files in directory:', output_directory_path)

# Call the function for the two directories
# modify_files_in_directory(
#     '/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train/labels_15class',
#     '/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train/labels'
# )
modify_files_in_directory(
    '/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train/labels_15class',
    '/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train/labels'
)
