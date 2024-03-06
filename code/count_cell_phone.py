import os

# First set of base directory and experiment folders
base_dir1 = "/mnt/scratch/sibo/yolov5/runs/detect/"
experiment_class_numbers1 = {
    # "exp290": '67',
    # "exp291": '61',
    "exp292": '67',
    "exp293": '1',
    "exp294": '61',
    "exp295": '1',
    "exp296": '1',
}

# Second set of base directory and experiment folders
base_dir2 = "/mnt/scratch/sibo/ultralytics/runs/detect/"
experiment_class_numbers2 = {
    # "predict30": '67'
}

# Function to count occurrences of a specific class number in a file
def count_specific_class_in_file(file_path, class_number):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().split()[0] == class_number:
                count += 1
    return count

# Function to count occurrences for a given experiment folder
def count_specific_class_for_experiment(base_dir, exp_folder, class_number):
    # Differentiate the label directory based on the base directory
    if base_dir == base_dir1:
        label_dir = os.path.join(base_dir, exp_folder, "labels")
    else:  # base_dir2
        label_dir = os.path.join(base_dir, exp_folder)

    total_count = 0
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(label_dir, filename)
                total_count += count_specific_class_in_file(file_path, class_number)
    else:
        print(f"Warning: Directory '{label_dir}' not found.")
    return total_count

# Process and report the occurrences for each experiment in both sets
for exp_folder, class_number in experiment_class_numbers1.items():
    count = count_specific_class_for_experiment(base_dir1, exp_folder, class_number)
    print(f"Total number of class {class_number} in {exp_folder} (base_dir1): {count}")

for exp_folder, class_number in experiment_class_numbers2.items():
    count = count_specific_class_for_experiment(base_dir2, exp_folder, class_number)
    print(f"Total number of class {class_number} in {exp_folder} (base_dir2): {count}")
