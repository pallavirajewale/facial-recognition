import cv2
from deepface import DeepFace
import numpy as np
from tkinter import Tk, filedialog
import os

# Accuracy calculation (Simple face recognition accuracy)
def calculate_accuracy(detected_face, ground_truth_faces):
    # Assuming ground_truth_faces contains list of known faces (e.g., image files)
    correct_matches = 0
    for gt_face in ground_truth_faces:
        try:
            # Compare the detected face with ground truth faces
            result = DeepFace.verify(detected_face, gt_face)
            if result["verified"]:
                correct_matches += 1
        except Exception as e:
            print(f"Error comparing faces: {e}")
    return correct_matches / len(ground_truth_faces) if ground_truth_faces else 0

# File dialog to select video
Tk().withdraw()  # Hide Tkinter root window
video_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Ground truth faces (list of known face images for accuracy calculation)
# In this example, we assume ground_truth_faces contains paths to known face images.
ground_truth_faces = ["path_to_known_face_image1.jpg", "path_to_known_face_image2.jpg"]

frame_count = 0
total_accuracy = 0
face_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    try:
        # DeepFace for face detection and recognition
        faces = DeepFace.detectFace(frame, detector_backend='opencv')  # You can use other backends as well

        for face in faces:
            detected_face_path = "temp_detected_face.jpg"
            cv2.imwrite(detected_face_path, face)
            
            # Calculate accuracy based on ground truth data
            accuracy = calculate_accuracy(detected_face_path, ground_truth_faces)
            total_accuracy += accuracy
            face_count += 1

            # Display face bounding box on the frame
            x, y, w, h = DeepFace.detectFace(frame, return_box=True)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error detecting faces: {e}")

    cv2.imshow("DeepFace Video Recognition", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print the average accuracy (mean accuracy of the faces detected)
if face_count > 0:
    average_accuracy = total_accuracy / face_count
    print(f"Average Recognition Accuracy: {average_accuracy:.2f}")
else:
    print("No faces detected in the video.")

cap.release()
cv2.destroyAllWindows()
