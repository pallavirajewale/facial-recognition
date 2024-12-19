import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import os
import time
import tkinter as tk
from tkinter import filedialog

# Replace these with your Tapo camera credentials and IP
#camera_ip = "192.168.1.68"  # Replace with your camera's IP address
#username = "easyparkai"  # RTSP username from Tapo app
#password = "easypark@123"  # RTSP password from Tapo app

# RTSP URL format for Tapo cameras
#rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/stream1"
    
# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Store known face encodings and names
known_face_encodings = []
known_face_names = []

# Function to load dataset and register faces using file dialog
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names
    
    # Create a Tkinter window to use the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    print("Select one or more images from your dataset.")
    
    # Open file dialog to select multiple images
    dataset_images = filedialog.askopenfilenames(title="Select Dataset Images", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not dataset_images:
        print("No images selected. Exiting.")
        return

    # Register faces from selected images
    for image_path in dataset_images:
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face in dataset image using MTCNN
        faces = mtcnn.detect_faces(rgb_image)
        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])

            if face_encoding:
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(os.path.basename(image_path).split('.')[0])  # Use the image filename as the name

    print(f"Loaded {len(known_face_encodings)} faces from dataset.")

# Function to stream video from Tapo camera and recognize faces
def stream_and_recognize_faces():
    global known_face_encodings, known_face_names

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
                else:
                    name = "Unknown"

                # Draw rectangle and name around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Calculate FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time  # Reset start time for next frame

        # Display FPS on the video
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the video feed with face recognition
        cv2.imshow("Tapo Camera Live Stream", frame)

        # Exit the stream when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    load_dataset_and_register_faces()  # Load faces from selected images
    stream_and_recognize_faces()  # Start streaming and recognizing faces
