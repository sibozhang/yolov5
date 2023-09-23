import os

def read_label_file(file_path):
    """Read a YOLO label file and return a list of bounding boxes for each class."""
    boxes = {0: [], 5: []}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)
            if class_id in [0, 5]:
                boxes[class_id].append((x_center, y_center, width, height))
    return boxes

def check_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    x1, y1 = x1_center - w1 / 2, y1_center - h1 / 2
    x2, y2 = x2_center - w2 / 2, y2_center - h2 / 2
    
    overlap = not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    return overlap

def main():
    base_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp140'
    labels_folder = os.path.join(base_dir, 'labels')
    
    analysis_dir = os.path.join(base_dir, 'data_analysis')
    if not os.path.exists(analysis_dir):
        print(f"Directory {analysis_dir} does not exist. Creating it...")
        os.makedirs(analysis_dir)
    else:
        print(f"Directory {analysis_dir} already exists.")
    
    output_file_path = os.path.join(analysis_dir, '202308_person_helmet.txt')
    
    print(f"Writing data to {output_file_path}...")
    with open(output_file_path, 'w') as output_file:
        for label_file in os.listdir(labels_folder):
            if label_file.endswith('.txt'):
                file_path = os.path.join(labels_folder, label_file)
                
                boxes = read_label_file(file_path)
                
                for person_box in boxes[0]:
                    wearing_helmet = False
                    
                    for helmet_box in boxes[5]:
                        if check_overlap(person_box, helmet_box):
                            wearing_helmet = True
                            break
                    
                    x_center, y_center, width, height = person_box
                    abs_width = width * 1280
                    abs_height = height * 720
                    abs_x_center = x_center * 1280
                    abs_y_center = y_center * 720
                    
                    if wearing_helmet:
                        output_file.write(f"Person with helmet in image {label_file} - Position: ({abs_x_center}, {abs_y_center}), Width: {abs_width}, Height: {abs_height}\n")
                    else:
                        output_file.write(f"Person without helmet in image {label_file} - Position: ({abs_x_center}, {abs_y_center}), Width: {abs_width}, Height: {abs_height}\n")
    
    print(f"Data analysis complete. Results written to {output_file_path}")

if __name__ == '__main__':
    main()
