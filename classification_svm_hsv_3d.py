import os
import cv2
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

def compute_color_histogram(image, bins=32):
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the 3D histogram in the HSV space
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [bins]*3, [0, 180, 0, 256, 0, 256])

    # Flatten the histogram
    hist = hist.flatten()

    return hist

def plot_3d_hist(features, plot_directory, bins=32, class_name='Unknown'):
    # Reshape the features
    features = features.reshape((bins, bins, bins))

    # Generate x, y, z coordinates for histogram values
    x, y, z = np.mgrid[0:bins, 0:bins, 0:bins]

    # Keep only the features that exceed a certain threshold
    keep = (features > 0.1)

    # Create a figure and plot the 3D histogram
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[keep], y[keep], z[keep], c=features[keep], s=20)
    ax.set_title(f'3D HSV Histogram for {class_name}')
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    plt.savefig(os.path.join(plot_directory, f"{class_name}_hsv_3d.png"))
    plt.close(fig)  # Close the figure

def load_dataset(directory, plot_directory):
    features = []
    labels = []
    image_paths = []
    classes = os.listdir(directory)
    
    for class_name in classes:
        class_directory = os.path.join(directory, class_name)
        class_features = []
        for image_name in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            color_histogram = compute_color_histogram(image)
            feature_vector = color_histogram
            features.append(feature_vector)
            class_features.append(feature_vector)
            labels.append(class_name)
            image_paths.append(image_path)

        feature_average = sum(class_features) / len(class_features)
        plot_3d_hist(feature_average, plot_directory,bins=32, class_name=class_name)

    return np.array(features), np.array(labels), image_paths, classes

def main():
    directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/person_crops/"  # replace with your directory
    predict_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_predict"  # replace with your output directory for KNN predictions
    plot_directory = "/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_histgram"  # replace with your output directory for KNN predictions
    model_save_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp52/svm_model/svm_model_3d.pkl'

    if not os.path.exists(predict_directory):
        os.makedirs(predict_directory)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    features, labels, image_paths, classes = load_dataset(directory, plot_directory)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(features, encoded_labels, range(len(image_paths)), test_size=0.2, random_state=42)
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    model = SVC(kernel='rbf', C=1)
    model.fit(X_train, y_train)
    saved_model = pickle.dumps(model)
    accuracy = model.score(X_test, y_test)

    y_pred = model.predict(X_test)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    pickle.dump(model, open(model_save_dir, 'wb'))

    for i, index in enumerate(idx_test):
        test_feature = X_test[i]
        test_feature = test_feature.reshape(1, -1) 
        
        prediction = model.predict(test_feature)
        predicted_class = label_encoder.inverse_transform(prediction)
        
        test_image_path = image_paths[index]
        image = cv2.imread(test_image_path)
        image = cv2.putText(image, f"{predicted_class[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        test_image_name = os.path.basename(test_image_path)
        output_path = os.path.join(predict_directory, f"predicted_{test_image_name}")
        cv2.imwrite(output_path, image)

        plot_3d_hist(image, plot_directory=plot_directory)

if __name__ == "__main__":
    main()
