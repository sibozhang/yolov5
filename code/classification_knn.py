import os
import cv2
import numpy as np
from skimage import io, color, feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.feature import hog

def compute_color_histogram(image, bins=(32, 32, 32)):
    # Compute the color histogram of the image
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_hog_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    return hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

def load_dataset(directory):
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
            image = cv2.resize(image, (128, 64))# resize to fixed size
            # Change here: using color histogram instead of HOG
            # Compute color histogram features
            color_histogram = compute_color_histogram(image)
            # Compute HOG features
            hog_features = compute_hog_features(image)
            
            # Concatenate color_histogram and hog_features
            feature_vector = np.hstack((color_histogram, hog_features))
            
            features.append(feature_vector)
            labels.append(class_name)
            image_paths.append(image_path)

    return np.array(features), np.array(labels), image_paths, classes

def main():
    directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp32/person_crops/"  # replace with your directory
    knn_predict_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp32/knn_predict"  # replace with your output directory for KNN predictions
    
    if not os.path.exists(knn_predict_directory):
        os.makedirs(knn_predict_directory)

    features, labels, image_paths, classes = load_dataset(directory)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(features, encoded_labels, range(len(image_paths)), test_size=0.2, random_state=42)
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy*100:.2f}%")

    # Test all images in the test set
    for i, index in enumerate(idx_test):
        test_feature = X_test[i]
        test_feature = test_feature.reshape(1, -1)  # reshape to 2D array
        
        # Predict
        prediction = model.predict(test_feature)
        predicted_class = label_encoder.inverse_transform(prediction)
        
        # Write text on image and save it
        test_image_path = image_paths[index]
        image = cv2.imread(test_image_path)
        image = cv2.putText(image, f"{predicted_class[0]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        output_path = os.path.join(knn_predict_directory, f"predicted_{i}.png")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    main()
