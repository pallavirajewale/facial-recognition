import cv2
import face_recognition
import dlib
import numpy as np

# Load the face detector and shape predictor from Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor(r"C:\facial project\shape_predictor_68_face_landmarks.dat")

def process_video_for_registration(video_path):
    """
    Process a video to register faces and extract encodings.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Convert frame to RGB format (face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("No faces detected in this frame.")
            continue

        # Detect landmarks for each face
        for face_location in face_locations:
            # Crop the face
            top, right, bottom, left = face_location
            face_crop = rgb_frame[top:bottom, left:right]

            # Convert face location to Dlib rectangle
            rect = dlib.rectangle(left, top, right, bottom)

            # Detect landmarks using the predictor
            landmarks = predictor(rgb_frame, rect)

            # Convert landmarks to the required format
            landmarks_array = [(p.x, p.y) for p in landmarks.parts()]

            # Compute face encodings
            try:
                face_encoding = face_recognition.face_encodings(rgb_frame, [landmarks_array])[0]
                print("Face encoding generated successfully:", face_encoding)
            except IndexError:
                print("Failed to compute face encoding. Skipping this face.")
                continue

    video_capture.release()
    print("Video processing completed.")

# Example function to test video registration
def load_dataset_and_register_faces():
    import tkinter as tk
    from tkinter import filedialog

    # Open a file dialog to select the video file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select the dataset video file")
    if not file_path:
        print("No file selected. Exiting...")
        return

    process_video_for_registration(file_path)

# Run the dataset loader
if __name__ == "__main__":
    load_dataset_and_register_faces()
