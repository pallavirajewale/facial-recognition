import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
import os
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Paths
DATASET_PATH = "faces_dataset"
MODEL_PATH = "face_pnn_model.pkl"
LABELS_PATH = "pnn_labels.pkl"

# Ensure dataset directory exists
os.makedirs(DATASET_PATH, exist_ok=True)

# Initialize the MTCNN detector
detector = MTCNN()

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def upload_image():
    """
    Opens a file dialog to upload an image from the device.
    """
    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            print("Image successfully loaded.")
            return image
        else:
            print("Error loading image.")
    else:
        print("No file selected.")
    return None

def recognize_faces_in_uploaded_image():
    """
    Recognize faces in an uploaded image using the trained PNN model.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("PNN model or labels not found. Please train the model first.")
        return

    # Load the trained PNN models, label mapping, and thresholds
    with open(MODEL_PATH, "rb") as model_file:
        pnn_models = pickle.load(model_file)

    with open(LABELS_PATH, "rb") as label_file:
        label_mapping = pickle.load(label_file)

    with open("pnn_thresholds.pkl", "rb") as threshold_file:
        thresholds = pickle.load(threshold_file)

    # Upload an image from the device
    image = upload_image()
    if image is None:
        return

    # Detect faces using MTCNN
    results = detector.detect_faces(image)

    for result in results:
        bounding_box = result['box']
        keypoints = result['keypoints']

        x, y, width, height = bounding_box
        face_roi = image[y:y + height, x:x + width]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (100, 100))
        face_roi = apply_clahe(face_roi)  # Ensure CLAHE is consistently applied
        face_resized = face_roi.flatten() / 255.0  # Normalize to [0, 1]

        log_likelihoods = {}
        for label, kde in pnn_models.items():
            log_likelihoods[label] = kde.score_samples([face_resized])[0]

        max_label = max(log_likelihoods, key=log_likelihoods.get)
        max_score = log_likelihoods[max_label]

        # Determine if the face is known or unknown
        if max_score < thresholds[max_label]:
            name = "Unknown"
        else:
            name = label_mapping[max_label]

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw landmarks
        for key, point in keypoints.items():
            cv2.circle(image, tuple(point), 2, (255, 0, 0), 5)

    # Display the image with recognized faces
    cv2.imshow("Face Recognition and Landmark Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n1: Train PNN model")
        print("2: Test model accuracy")
        print("3: Recognize faces in uploaded image")
        print("4: Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            train_pnn_model()
        elif choice == "2":
            test_accuracy()
        elif choice == "3":
            recognize_faces_in_uploaded_image()
        elif choice == "4":
            break
        else:
            print("Invalid option. Please try again.")
