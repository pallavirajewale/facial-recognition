import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tkinter as tk
from tkinter import filedialog


# Define a simple CNN model for face recognition (this is just an example, you can improve the model)
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification (e.g., face vs. no face)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess frames before feeding into the CNN
def preprocess_frame(frame):
    # Resize to 64x64 and normalize pixel values
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0  # Normalize pixel values
    return np.expand_dims(frame, axis=0)  # Add batch dimension

# Initialize tkinter for file dialog
root = tk.Tk()
root.withdraw()

# Open file dialog to select a video file
file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])

# Check if the user selected a file
if file_path:
    # Load the CNN model
    model = create_model()

    # Initialize video capture from selected file
    cap = cv2.VideoCapture(file_path)

    # Load OpenCV Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    total_frames = 0
    correct_predictions = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video ends

        total_frames += 1

        # Detect faces using OpenCV's Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create ground truth label: 1 for face detected, 0 for no face detected
        ground_truth = 1 if len(faces) > 0 else 0

        # Preprocess the frame for CNN prediction
        processed_frame = preprocess_frame(frame)

        # Perform face recognition using the CNN (dummy model here)
        prediction = model.predict(processed_frame)
        predicted_label = 1 if prediction[0] > 0.5 else 0

        # Compare the prediction with the ground truth
        if predicted_label == ground_truth:
            correct_predictions += 1

        # Display the frame with detected faces and the prediction label
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Pred: {'Face' if predicted_label == 1 else 'No Face'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Face Recognition", frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate accuracy
    accuracy = correct_predictions / total_frames * 100 if total_frames > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")

    cap.release()
    cv2.destroyAllWindows()
else:
    print("No file selected.")
