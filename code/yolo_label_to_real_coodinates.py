import os

def convert_labels_to_real(input_dir, output_dir, image_width, image_height, class_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                elements = line.split()
                if len(elements) != 6:
                    continue  # Skipping malformed lines

                label, center_x, center_y, width, height, conf = map(float, elements)

                # Convert to real coordinates from normalized coordinates
                center_x = int(center_x * image_width)
                center_y = int(center_y * image_height)
                width = int(width * image_width)
                height = int(height * image_height)

                # Ensure label is within the range of class_names
                class_name = class_names[int(label)] if int(label) < len(class_names) else "unknown"

                # Write the real coordinates to the new file
                outfile.write(f"{class_name} {center_x} {center_y} {width} {height} {conf}\n")

# Example usage
if __name__ == "__main__":
    YOLO_OUTDOOR_MODEL = "20231116m" 
    YOLO_CABIN_MODEL = "20231107m_cabin"

    # Class names for both models
    class_names_outdoor = ['person', 'car', 'truck', 'cone', 'helmet', 'machinery vehicle', 'hook']
    class_names_cabin = ['person', 'cell phone']

    # folder_names = ["20230829_114310", "20230901_101413", "20230904_135906", "20230906_101400", "20230912_145556", "20230830_220436", "20230831_210234", "20230902_231627", "20230905_193841", "20230906_190938"]
    # folder_names = ["20231011", "20231012", "20231013", "20231016", "20231017", "20231018", "20231019", "20231020"] 
    folder_names = ["20231101", "20231102"]
    data_folder = "202311"

    for folder_name in folder_names:
        # VIDEO_FOLDER_PATH = f"/mnt/data/sibo/GP45/overall_selected/GP45_{folder_name}"
        VIDEO_FOLDER_PATH = f"/mnt/data/sibo/GP45/overall_selected/{data_folder}/{folder_name}"

        # Fetch all subdirectories (assuming they are numeric as per your previous logic)
        subdirectories = [d for d in os.listdir(VIDEO_FOLDER_PATH) if os.path.isdir(os.path.join(VIDEO_FOLDER_PATH, d))]

        for model_path in subdirectories:
            if model_path == YOLO_CABIN_MODEL:
                class_names = class_names_cabin
                image_width, image_height = 1280, 720
            elif model_path == YOLO_OUTDOOR_MODEL:
                class_names = class_names_outdoor
                image_width, image_height = 1920, 1080
            else:
                continue

            for filename in os.listdir(os.path.join(VIDEO_FOLDER_PATH, model_path)):
                INPUT_DIR = os.path.join(VIDEO_FOLDER_PATH, model_path, str(filename), "labels")
                OUTPUT_DIR = os.path.join(VIDEO_FOLDER_PATH, model_path, str(filename), "labels_real")

                convert_labels_to_real(INPUT_DIR, OUTPUT_DIR, image_width, image_height, class_names)
                print(f"Processed {folder_name} - {filename}: Output directory - {OUTPUT_DIR}")
