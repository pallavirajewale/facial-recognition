import cv2
import face_recognition
import numpy as np
import threading
import mediapipe as mp
from sklearn.neural_network import MLPClassifier
import os
import tkinter as tk
from tkinter import filedialog
from mtcnn import MTCNN

# Global variables
frame = None
frame_lock = threading.Lock()

mtcnn = MTCNN()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to load face encodings from a selected dataset
def load_face_encodings(image_folder):
    face_encodings = []
    face_labels = []

    for root_dir, subdirs, files in os.walk(image_folder):
        for image_name in files:
            if image_name.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root_dir, image_name)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = mtcnn.detect_faces(rgb_image)
                for face in faces:
                    x, y, w, h = face['box']
                    try:
                        face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])[0]
                        face_encodings.append(face_encoding)
                        face_labels.append(image_name.split('.')[0])
                        # face_images.append(image)
                    except IndexError:
                        print(f"Face not detected in {image_name}. Skipping.")
    
    
    return face_encodings, face_labels

# Function to open a dialog for selecting a dataset directory
def select_dataset():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    dataset_directory = filedialog.askdirectory(title="Select Dataset Directory")
    return dataset_directory

# Select the dataset using Tkinter
image_folder = select_dataset()

if not image_folder:
    print("No dataset selected. Exiting...")
    exit()

# Load training data
X, y = load_face_encodings(image_folder)

# Initialize the PNN model (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
model.fit(X, y)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

def capture_frame():
    global frame
    while True:
        ret, captured_frame = video_capture.read()
        if ret:
            with frame_lock:
                frame = captured_frame

def process_frame():
    global frame
    while True:
        with frame_lock:
            if frame is not None:
                # Convert the frame to RGB (MediaPipe uses RGB images)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame using MediaPipe Face Mesh
                results = face_mesh.process(rgb_frame)

                # If faces are found
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Optionally draw landmarks for visualization
                        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                        # Get the face bounding box for recognition (using the first 4 landmarks as an approximation)
                        min_x, min_y = max_x, max_y = 0, 0
                        for landmark in face_landmarks.landmark:
                            min_x = min(min_x, landmark.x)
                            min_y = min(min_y, landmark.y)
                            max_x = max(max_x, landmark.x)
                            max_y = max(max_y, landmark.y)

                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        left = int(min_x * w)
                        top = int(min_y * h)
                        right = int(max_x * w)
                        bottom = int(max_y * h)

                        # Detect face encodings (same as before)
                        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])

                        if face_encoding:
                            face_features = np.array([face_encoding[0]])  # Reshape for prediction
                            name = "Unknown"
                            
                            try:
                                # Predict the label using PNN model
                                prediction = model.predict(face_features)
                                name = prediction[0]
                            except Exception as e:
                                print(f"Error in prediction: {e}")

                            # Draw a rectangle around the face and label it
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Display the frame with recognized faces
                cv2.imshow("Live Face Recognition with Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Create threads
capture_thread = threading.Thread(target=capture_frame)
process_thread = threading.Thread(target=process_frame)

# Start threads
capture_thread.start()
process_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
