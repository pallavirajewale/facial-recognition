import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import pickle
from tkinter import filedialog
from tkinter import Tk
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import mediapipe as mp

# Tapo Camera RTSP configuration
camera_ip = "192.168.1.68"  # Replace with your camera's IP address
username = "easyparkai"  # RTSP username from Tapo app
password = "easypark@123"  # RTSP password from Tapo app
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/stream1"

# Initialize necessary components
mtcnn = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Known face encodings and names
known_face_encodings = []
known_face_names = []
face_images = []

# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

# Load dataset and register faces
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images
    print("Select the folder containing dataset images.")
    root = Tk()
    root.withdraw()
    dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
    if not dataset_folder:
        print("No dataset folder selected. Exiting.")
        return

    for root_dir, subdirs, files in os.walk(dataset_folder):
        for image_name in files:
            if image_name.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root_dir, image_name)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = mtcnn.detect_faces(rgb_image)
                for face in faces:
                    x, y, w, h = face['box']
                    face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(image_name.split('.')[0])
                    face_images.append(image)

    # Train the PNN model
    print("Training PNN model...")
    face_embeddings_scaled = scaler.fit_transform(known_face_encodings)
    pnn_model.fit(face_embeddings_scaled, known_face_names)
    print("PNN model trained successfully.")

# Function to calculate cosine similarity between two face encodings
def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

# Function to apply contrast enhancement to the frame
def apply_crystal_contrast_enhancement(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_frame = clahe.apply(gray_frame)
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
    return enhanced_frame

def stream_and_recognize_faces():
    global known_face_encodings, known_face_names, face_images

    # Open the RTSP video stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open video stream. Check your RTSP URL or camera settings.")
        return

    print("Streaming video from Tapo camera. Press 'q' to exit.")

    # Initialize FPS variables
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = mtcnn.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

            if face_encoding:
                face_encoding = face_encoding[0]

                # Compare the detected face with the known faces in the dataset
                similarities = [np.dot(face_encoding, known_encoding) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)) for known_encoding in known_face_encodings]
                max_similarity = max(similarities)
                best_match_index = similarities.index(max_similarity)

                if max_similarity > 0.6:
                    name = known_face_names[best_match_index]
                    matched_image = face_images[best_match_index]
                    
                    # Resize matched image to fit on screen
                    matched_image_resized = cv2.resize(matched_image, (100, 100))  # Resize to 100x100 or whatever fits
                else:
                    name = "Unknown"
                    matched_image_resized = np.zeros((100, 100, 3), dtype=np.uint8)  # Blank image for unknown faces

                # Draw rectangle and name around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Place the matched image on the frame (next to the face)
                frame[10:110, 10:110] = matched_image_resized

        # Calculate FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time  # Reset start time for next frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display FPS on the video
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the video feed with face recognition
        cv2.imshow("Tapo Camera Live Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load the dataset and register faces
load_dataset_and_register_faces()

# Start recognizing faces using webcam (or streaming from Tapo camera)
stream_and_recognize_faces()
