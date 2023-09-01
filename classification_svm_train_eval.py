import os
import cv2
import numpy as np
import pickle
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

def compute_color_histogram(image, bins=32):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the histogram in the HSV channels
    hist = np.concatenate([cv2.calcHist([image_hsv], [i], None, [bins], [0, 256]) for i in range(3)]).flatten()

    # Normalize the histogram
    hist = cv2.normalize(hist, hist)

    return hist

def load_dataset(directory, histgram_directory):
    features = []
    labels = []
    image_paths = []
    classes = os.listdir(directory)
    
    for class_name in classes:
        class_directory = os.path.join(directory, class_name)
        for image_name in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_name)
            # Note: using cv2 to read image in BGR format for histogram
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 64))
            # resize to fixed size
            color_histogram = compute_color_histogram(image)
            feature_vector = color_histogram

            features.append(feature_vector)
            labels.append(class_name)
            image_paths.append(image_path)

        feature_average = sum(features) / len(feature_vector)
        # print(class_name, ' feature_average:', feature_average)
        # savefig in empty canvas
        plt.figure()
        plt.hist(feature_average, bins=32)
        plt.savefig(os.path.join(histgram_directory, f"{class_name}_average.png"))

    return np.array(features), np.array(labels), image_paths, classes

def main():
    directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/person_crops/"  # replace with your directory
    train_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/person_crops_split/train/"
    test_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/person_crops_split/test/"
    predict_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_predict"  # replace with your output directory for KNN predictions
    histgram_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_histgram"  # replace with your output directory for KNN predictions
    model_save_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_model/svm_model.pkl'
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    if not os.path.exists(predict_directory):
        os.makedirs(predict_directory)
    if not os.path.exists(histgram_directory):
        os.makedirs(histgram_directory)
    
    features, labels, image_paths, classes = load_dataset(directory, histgram_directory)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    # print(encoded_labels, labels)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(features, encoded_labels, range(len(image_paths)), test_size=0.2, random_state=42)
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")
    #save train and test data as images under directory
    # Copy training images to train_directory with labels subdirectories
    for idx in idx_train:
        label_dir = os.path.join(train_directory, labels[idx])
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        shutil.copy(image_paths[idx], label_dir)
    
    # Copy testing images to test_directory with labels subdirectories
    for idx in idx_test:
        label_dir = os.path.join(test_directory, labels[idx])
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        shutil.copy(image_paths[idx], label_dir)

     
    # Using Support Vector Machine Classifier
    model = SVC(kernel='rbf', C=1)
    model.fit(X_train, y_train)

    # Save the trained model as a pickle string.
    saved_model = pickle.dumps(model)

    # Save the model to disk
    pickle.dump(model, open(model_save_dir, 'wb'))
    print(f"Model saved to {model_save_dir}")

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy*100:.2f}%")

    # Predict the test set results
    y_pred = model.predict(X_test)

    # Evaluate the model accuracy score by class
    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Test all images in the test set
    for i, index in enumerate(idx_test):
        test_feature = X_test[i]
        test_feature = test_feature.reshape(1, -1)  # reshape to 2D array
        
        # Predict
        prediction = model.predict(test_feature)
        predicted_class = label_encoder.inverse_transform(prediction)
        # print(f"Predicted class: {predicted_class[0]}")
        
        # Write text on image and save it
        test_image_path = image_paths[index]
        image = cv2.imread(test_image_path)
        image = cv2.putText(image, f"{predicted_class[0]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        test_image_name = os.path.basename(test_image_path)
        output_path = os.path.join(predict_directory, f"predicted_{test_image_name}.png")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    main()
