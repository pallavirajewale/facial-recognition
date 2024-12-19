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
    model.add(Dense(1, activation='sigmoid'))  # Fo binary classification (e.g., face vs. no face)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video ends

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Perform face recognition (this is just a dummy prediction, replace with actual face detection/recognition)
        prediction = model.predict(processed_frame)
        label = "Face" if prediction[0] > 0.5 else "No Face"

        # Display the frame with the prediction label
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Face Recognition", frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("No file selected.")
