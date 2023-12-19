import cv2
import numpy as np
import pickle
# from sklearn.preprocessing import LabelEncoder
import os

def compute_color_histogram(image, bins=32):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the histogram in the HSV channels
    hist = np.concatenate([cv2.calcHist([image_hsv], [i], None, [bins], [0, 256]) for i in range(3)]).flatten()

    # Normalize the histogram
    hist = cv2.normalize(hist, hist)

    return hist

def svm_predict(model_path, image_path):
    # load the model from disk
    loaded_model = pickle.load(open(model_path, 'rb'))

    # load and preprocess a new image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # resize to fixed size
    feature_vector = compute_color_histogram(image)

    # reshape to 2D array (1, -1) because we are predicting on one instance
    feature_vector = feature_vector.reshape(1, -1) 

    # predict the class of the new image
    prediction = loaded_model.predict(feature_vector)

    # # Write text on image and save it
    # image = cv2.putText(image, f"{prediction[0]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    test_image_name = os.path.basename(image_path)
    # output_path = os.path.join(output_dir, f"predicted_{test_image_name}")
    # cv2.imwrite(output_path, image)
    print(f"Predicted {test_image_name} as {prediction[0]}")
    if prediction[0] == 0:
        print('person, person type label 0')
    elif prediction[0] == 1:
        print('rigger, person type label 1')
    elif prediction[0] == 2:
        print('windbreaker, person type label 2')
    return prediction[0]

def main():
    model_path = '/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_model/svm_model.pkl'  # replace with your model path
    image_path = '/mnt/scratch/sibo/yolov5/runs/detect/exp52/person_crops/rigger/20220712_165614_cam_0_2_move_load_bad_mp4-0_jpg.rf.d380c53f4ca507f816218ef9021d0541.jpg'  # replace with your image path
    # output_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_predict'  # replace with your output directory

    print(svm_predict(model_path, image_path))

if __name__ == "__main__":
    main()
