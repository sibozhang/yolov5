import os

def modify_files_in_directory(input_directory_path, output_directory_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

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
                # print(f'Writing to file: {output_file_path}')
                for line in lines:
                    parts = line.split(' ')
                    cat_id = int(parts[0])
                    if cat_id in [11, 12]:
                        print(f'Changing line: {output_file_path} {line}, cat_id: {cat_id} to 0')
                        cat_id = 0
                    elif cat_id in [13, 14, 15, 16]:
                        print(f'Changing line: {output_file_path} {line}, cat_id: {cat_id} to {cat_id - 2}')
                        cat_id -= 2
                    f.write(f'{cat_id} {" ".join(parts[1:])}')

    print('writing modified files in directory:', output_directory_path)

# Call the function for the two directories
modify_files_in_directory(
    '/mnt/data/sibo/GP45/20230517-0523/dino_2class/train/labels_17class',
    '/mnt/data/sibo/GP45/20230517-0523/dino_2class/train/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/GP45/20230517-0523/dino_2class/valid/labels_17class',
    '/mnt/data/sibo/GP45/20230517-0523/dino_2class/valid/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/GP45/20230606-0612/dino_2class/train/labels_17class',
    '/mnt/data/sibo/GP45/20230606-0612/dino_2class/train/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/GP45/20230606-0612/dino_2class/valid/labels_17class',
    '/mnt/data/sibo/GP45/20230606-0612/dino_2class/valid/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train/labels_17class',
    '/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/Hard_Hat_Workers.v1i.yolov5pytorch/train/labels_17class',
    '/mnt/data/sibo/Hard_Hat_Workers.v1i.yolov5pytorch/train/labels'
)
modify_files_in_directory(
    '/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train/labels_17class',
    '/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train/labels'
)
