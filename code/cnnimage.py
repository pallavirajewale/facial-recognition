import cv2
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained FaceNet model (you need to download this model first)
def load_facenets_model(model_path):
    model = load_model(model_path)
    return model

# Initialize MTCNN detector for face detection
detector = MTCNN()

# Function to preprocess the image for face recognition
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))  # Resize to the required size for FaceNet
    image = image.astype(np.float32)
    mean = np.array([127.5, 127.5, 127.5])  # Mean subtraction (standard for FaceNet)
    std = np.array([128.0, 128.0, 128.0])   # Standard deviation
    image = (image - mean) / std  # Normalize
    return np.expand_dims(image, axis=0)

# Initialize tkinter for file dialog
root = tk.Tk()
root.withdraw()

# Open file dialog to select an image file
image_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

# Check if the user selected an image file
if image_path:
    # Load FaceNet model (update the path to your model)
    model_path = "path_to_your_facenets_model.h5"  # Update with actual path to your pre-trained FaceNet model
    facenet_model = load_facenets_model(model_path)

    # Load the uploaded image
    image = cv2.imread(image_path)

    # Detect faces using MTCNN
    faces = detector.detect_faces(image)

    # Create ground truth label: 1 for face detected, 0 for no face detected
    ground_truth = 1 if len(faces) > 0 else 0

    # If faces are detected, crop and preprocess each face for recognition
    if len(faces) > 0:
        face = faces[0]  # Just take the first face detected (for simplicity)
        x, y, w, h = face['box']
        face_image = image[y:y+h, x:x+w]
        processed_image = preprocess_image(face_image)

        # Perform face recognition using FaceNet (dummy model here)
        prediction = facenet_model.predict(processed_image)
        predicted_label = 1 if prediction[0][0] > 0.5 else 0

        # Calculate accuracy (for simplicity, we're comparing face vs no face)
        accuracy = 100 if predicted_label == ground_truth else 0

        # Draw bounding box around detected face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the image with bounding box and predicted label
        cv2.putText(image, f"Pred: {'Face' if predicted_label == 1 else 'No Face'}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image Face Recognition", image)
        cv2.waitKey(0)  # Wait for any key press to close the window
        cv2.destroyAllWindows()

        # Print the accuracy
        print(f"Accuracy: {accuracy}%")
    else:
        print("No faces detected in the image.")
else:
    print("No image selected.")
