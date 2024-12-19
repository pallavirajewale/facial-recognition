import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
FACE_DATASET_PATH = "faces_dataset"  # Directory to store face images
EMBEDDINGS_PATH = "face_embeddings.pkl"
CLASSIFIER_PATH = "face_classifier.pkl"
FACENET_MODEL_PATH = "facenet_keras.h5"  # Update this path if the file is saved elsewhere

# Ensure dataset directory exists
os.makedirs(FACE_DATASET_PATH, exist_ok=True)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_facenet_model():
    """
    Load the pre-trained FaceNet model. Handle missing files gracefully.
    """
    if not os.path.exists(FACENET_MODEL_PATH):
        print(f"Error: Model file not found at {FACENET_MODEL_PATH}")
        print("Please download the model from: https://github.com/nyoki-mtl/keras-facenet")
        print("Save it in the same directory as this script or update FACENET_MODEL_PATH.")
        exit(1)

    print("Loading FaceNet model...")
    return load_model(FACENET_MODEL_PATH)

# Load FaceNet model
facenet_model = load_facenet_model()

def preprocess_face(face):
    """
    Preprocess face for CNN model.
    """
    face = cv2.resize(face, (160, 160))  # Resize to model input size
    face = img_to_array(face)
    face = preprocess_input(face)
    return np.expand_dims(face, axis=0)

def extract_embeddings(face):
    """
    Extract embeddings from the face using the CNN model.
    """
    preprocessed_face = preprocess_face(face)
    embedding = facenet_model.predict(preprocessed_face)
    return embedding.flatten()

def register_faces():
    """
    Register faces through webcam and save them with embeddings.
    """
    cap = cv2.VideoCapture(0)
    face_id = input("Enter the name for the person being registered: ").strip()

    if not face_id:
        print("Name cannot be empty. Exiting...")
        return

    person_path = os.path.join(FACE_DATASET_PATH, face_id)
    os.makedirs(person_path, exist_ok=True)

    print("Position your face in front of the camera. Press 'q' to stop capturing.")

    embeddings = []
    labels = []
    count = 0

    # Data augmentation setup
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]  # Use color image for embeddings
            embedding = extract_embeddings(face_roi)
            embeddings.append(embedding)
            labels.append(face_id)

            face_file = os.path.join(person_path, f"face_{count}.jpg")
            cv2.imwrite(face_file, cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            count += 1

            # Augment face image
            augmented_faces = datagen.flow(np.expand_dims(face_roi, axis=0), batch_size=1)
            for _ in range(5):  # Generate 5 augmented images
                augmented_face = augmented_faces.next()
                augmented_embedding = extract_embeddings(augmented_face[0])
                embeddings.append(augmented_embedding)
                labels.append(face_id)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing Face {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Registering Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:  # Capture 20 images per person
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save embeddings and labels
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as file:
            data = pickle.load(file)
        data['embeddings'].extend(embeddings)
        data['labels'].extend(labels)
    else:
        data = {"embeddings": embeddings, "labels": labels}

    with open(EMBEDDINGS_PATH, "wb") as file:
        pickle.dump(data, file)

    print(f"Registration completed for {face_id}.")

def train_recognizer():
    """
    Train the classifier on extracted embeddings.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        print("No embeddings found. Please register faces first.")
        return

    with open(EMBEDDINGS_PATH, "rb") as file:
        data = pickle.load(file)

    embeddings = np.array(data['embeddings'])
    labels = np.array(data['labels'])

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Scaling the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Train an SVM classifier
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(embeddings_scaled, labels_encoded)

    # Save classifier and label encoder
    with open(CLASSIFIER_PATH, "wb") as file:
        pickle.dump({"classifier": classifier, "label_encoder": le, "scaler": scaler}, file)

    print("Face recognizer trained and saved successfully.")

def recognize_faces():
    """
    Recognize faces in the webcam feed.
    """
    if not os.path.exists(CLASSIFIER_PATH):
        print("Face recognizer model not found. Please train the model first.")
        return

    with open(CLASSIFIER_PATH, "rb") as file:
        data = pickle.load(file)

    classifier = data["classifier"]
    label_encoder = data["label_encoder"]
    scaler = data["scaler"]

    cap = cv2.VideoCapture(0)
    print("Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            embedding = extract_embeddings(face_roi)

            # Scale the embedding before predicting
            embedding_scaled = scaler.transform([embedding])

            probabilities = classifier.predict_proba(embedding_scaled)[0]
            max_index = np.argmax(probabilities)
            confidence = probabilities[max_index]

            if confidence > 0.6:  # Adjust threshold as needed
                name = label_encoder.inverse_transform([max_index])[0]
                label_text = f"{name} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                label_text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n1: Register a new face")
        print("2: Train face recognizer")
        print("3: Start face recognition")
        print("4: Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            register_faces()
        elif choice == "2":
            train_recognizer()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")