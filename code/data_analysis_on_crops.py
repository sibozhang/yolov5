import os
from PIL import Image
from collections import defaultdict

def get_image_sizes(folder_path, output_file, image_counter, image_dimensions):
    image_count = 0

    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(folder_path, filename)
                with Image.open(image_path) as img:
                    width, height = img.size
                    f.write(f"{filename} {width} {height}\n")

                    image_count += 1
                    image_counter[filename] += 1
                    image_dimensions[filename] = (width, height)

    print(f"Total number of images in {folder_path}: {image_count}")

if __name__ == "__main__":
    base_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp139/crops/'
    helmet_counter = defaultdict(int)
    person_counter = defaultdict(int)
    helmet_dimensions = {}
    person_dimensions = {}

    helmet_path = os.path.join(base_dir, 'helmet')
    helmet_size_file = os.path.join(base_dir, 'helmet_image_sizes.txt')
    get_image_sizes(helmet_path, helmet_size_file, helmet_counter, helmet_dimensions)
    
    person_path = os.path.join(base_dir, 'person')
    person_size_file = os.path.join(base_dir, 'person_image_sizes.txt')
    get_image_sizes(person_path, person_size_file, person_counter, person_dimensions)

    all_images = set(helmet_counter.keys()) | set(person_counter.keys())

    output_file_path = os.path.join(base_dir, 'person_helmet_size.txt')
    with open(output_file_path, 'w') as output_file:
        for image in all_images:
            person_count = person_counter[image]
            helmet_count = helmet_counter[image]
            person_dim = person_dimensions.get(image, (None, None))
            helmet_dim = helmet_dimensions.get(image, (None, None))

            output_str = ""
            if person_count > 0:
                output_str += f"{image}: Person count = {person_count}, Person dimensions = {person_dim}"
            
            if helmet_count > 0:
                if output_str:
                    output_str += ", "
                else:
                    output_str += f"{image}: "
                output_str += f"Helmet count = {helmet_count}, Helmet dimensions = {helmet_dim}"
            elif helmet_count == 0 and person_count == 0:
                continue  # Skip the output if both counts are zero
            
            output_file.write(output_str + '\n')

