import cv2
import numpy as np
import os
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to extract features using DeepFace
def extract_features(image):
    try:
        # Use DeepFace to get embeddings for the face
        face_embedding = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)
        return np.array(face_embedding[0]['embedding'])
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(128)  # Return a zero vector if feature extraction fails

# Function to validate and preprocess an image
def validate_and_preprocess_image(image):
    """
    Ensure the image has valid dimensions and resize to (160, 160, 3).
    """
    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        return None  # Skip invalid images
    return cv2.resize(image, (160, 160))

# Create a folder for storing images if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize SVM for classification
svm = SVC(kernel='linear', probability=True)

# Initialize StandardScaler for feature normalization
scaler = StandardScaler()

# Initialize lists to hold face features and labels
raw_images = []
raw_labels = []
name_to_id = {}
id_to_name = {}
img_counter = 0

# Ask the user how many people they want to capture faces for
num_people = int(input("Enter the number of people to capture faces for: "))

# Start capturing faces for training
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

for i in range(num_people):
    print(f"Capturing faces for person {i + 1} of {num_people}")
    name = input("Enter the name of the person: ")

    if name not in name_to_id:
        label = len(name_to_id)
        name_to_id[name] = label
        id_to_name[label] = name
    else:
        label = name_to_id[name]

    print(f"Starting to capture images for {name}. Press 'Esc' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        # Detect faces in the frame using MTCNN
        results = detector.detect_faces(frame)
        for result in results:
            x, y, w, h = result['box']
            face = frame[y:y + h, x:x + w]

            # Validate and preprocess face
            valid_image = validate_and_preprocess_image(face)
            if valid_image is not None:
                raw_images.append(valid_image)
                raw_labels.append(label)
                img_counter += 1

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to stop
            break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {img_counter} images.")

# Augment the dataset using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images, augmented_labels = [], []
for img, label in zip(raw_images, raw_labels):
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    for batch in datagen.flow(img, batch_size=1):
        augmented_images.append(batch[0].astype(np.uint8))  # Ensure uint8
        augmented_labels.append(label)
        if len(augmented_images) >= 5 * len(raw_images):  # Augment each image 5 times
            break

# Extract features from augmented images
face_features, face_labels = [], []
for img, label in zip(augmented_images, augmented_labels):
    features = extract_features(img)
    if features.any():  # Add valid embeddings
        face_features.append(features)
        face_labels.append(label)

# Normalize the features
face_features = scaler.fit_transform(face_features)

# Train the SVM classifier
svm.fit(face_features, face_labels)
print("SVM trained successfully.")

# Real-time face recognition
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Starting face recognition... Press 'Esc' to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    # Detect faces in the frame
    results = detector.detect_faces(frame)
    for result in results:
        x, y, w, h = result['box']
        face = frame[y:y + h, x:x + w]

        # Validate and preprocess face
        valid_image = validate_and_preprocess_image(face)
        if valid_image is not None:
            features = extract_features(valid_image).reshape(1, -1)
            features = scaler.transform(features)

            # Predict the label of the face using SVM
            predicted_label = svm.predict(features)
            predicted_name = id_to_name.get(predicted_label[0], "Unknown")

            # Display the name of the predicted person
            cv2.putText(frame, f"Person: {predicted_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with detected faces and predictions
    cv2.imshow("Face Recognition", frame)

    # Press 'Esc' to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
