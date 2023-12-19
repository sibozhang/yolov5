import os
from collections import Counter

# Path to the directory containing the label files
label_dir = "/mnt/scratch/sibo/datasets/Objects365/labels/train"

# Names of the classes
class_names = ['person', 'car', 'truck', 'cell phone', 'cone', 'helmet', 'machinery vehicle', 'hook']

# Initialize Counter object to store class counts
class_counter = Counter()

# Initialize counter for printing progress
progress_counter = 0

# Loop through each label file in the directory
for label_file in os.listdir(label_dir):
    label_file_path = os.path.join(label_dir, label_file)
    
    # Increment the progress counter
    progress_counter += 1
    
    # Print progress every 1000 files
    if progress_counter % 1000 == 0:
        print(f"Processed {progress_counter} label files.")
    
    # Open and read each label file
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Assuming the class index is the first element in each line of the label file
            class_index = int(line.strip().split()[0])
            class_name = class_names[class_index]
            class_counter[class_name] += 1

# Print the counts for each class
for class_name, count in class_counter.items():
    print(f"{class_name}: {count}")
