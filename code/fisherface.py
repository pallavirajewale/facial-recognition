import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog to select a video file
def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Hides the root window
    file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Ask user to select a video file
video_path = select_video_file()

if not video_path:
    print("No video file selected. Exiting...")
    exit()

# Open the selected video file
cap = cv2.VideoCapture(video_path)

# Initialize variables for face recognition
faces = []
labels = []
current_label = 0  # Start with label 0 for the first person
true_labels = []  # This will store true labels for accuracy calculation
predicted_labels = []  # This will store predicted labels for comparison

# Collect face data from the selected video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the detected face for training
        face = gray[y:y + h, x:x + w]
        faces.append(face)
        labels.append(current_label)  # You can assign different labels to different faces
        true_labels.append(current_label)  # Store the true label for this face

    # Display the current frame with faces drawn
    cv2.imshow("Video Feed", frame)

    # Stop when the user presses 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Once face data collection is done, train the recognizer
if faces:
    recognizer.train(faces, np.array(labels))  # Train with the collected faces and labels
    recognizer.save('trainer.yml')  # Save the trained model
    print("Training complete and model saved!")

# After data collection, use the trained recognizer to predict faces
cap.release()
cv2.destroyAllWindows()

# Load the trained recognizer for prediction
recognizer.read('trainer.yml')

# Reopen the video file to predict faces
cap = cv2.VideoCapture(video_path)

# Variables to calculate accuracy
correct_predictions = 0
total_predictions = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face
        face = gray[y:y + h, x:x + w]

        # Predict the label and confidence of the detected face
        label, confidence = recognizer.predict(face)

        # Compare the predicted label with the true label (for accuracy calculation)
        if label == true_labels[total_predictions]:  # Compare with true label
            correct_predictions += 1

        total_predictions += 1

        # Display the prediction on the frame
        cv2.putText(frame, f"Label: {label}, Conf: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow("Video Feed", frame)

    # Stop when the user presses 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Calculate accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No predictions made, unable to calculate accuracy.")
