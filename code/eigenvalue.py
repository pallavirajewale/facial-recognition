import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog and select a video
def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path

# Function to preprocess and collect faces
def collect_faces(video_path, max_faces=10):
    # Initialize OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []  # List to hold the faces
    labels = []  # List to hold corresponding labels (just integers for simplicity)
    
    # Open video file or capture device
    cap = cv2.VideoCapture(video_path)

    while len(faces) < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"Detected faces in current frame: {len(detected_faces)}")  # Debug print
        
        for (x, y, w, h) in detected_faces:
            # Extract the face region
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))  # Resize to a fixed size (100x100)
            faces.append(face_resized.flatten())  # Flatten the face image and add to the list
            labels.append(len(faces))  # Assign a simple label (just index)

        if len(faces) >= max_faces:
            break

    cap.release()
    
    if len(faces) == 0:
        print("No faces were detected.")  # Debug message if no faces detected
    else:
        print(f"Total faces collected: {len(faces)}")  # Debug message showing collected faces
    
    return np.array(faces), np.array(labels)

# Function to train Eigenfaces model
def train_eigenfaces(faces, labels):
    # Standardizing the data before PCA
    scaler = StandardScaler()
    faces_scaled = scaler.fit_transform(faces)
    
    # Dynamically adjust the number of components based on the number of faces
    n_components = min(50, faces.shape[0] - 1)  # Set to 50 or number of samples - 1
    
    # PCA for Eigenfaces (dimensionality reduction)
    pca = PCA(n_components=n_components)  # Adjust number of components dynamically
    faces_pca = pca.fit_transform(faces_scaled)
    
    # Train a classifier (e.g., KNN) on the reduced face data
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(faces_pca, labels)
    
    return pca, scaler, knn

# Function to predict face using trained Eigenfaces model
def predict_face(pca, scaler, knn, face, labels):
    # Standardize the face image
    face_scaled = scaler.transform([face.flatten()])
    
    # Apply PCA transformation
    face_pca = pca.transform(face_scaled)
    
    # Predict the label
    predicted_label = knn.predict(face_pca)
    
    # Return the predicted label
    return predicted_label

# Function to compute accuracy
def compute_accuracy(pca, scaler, knn, faces, labels):
    predictions = []
    for face in faces:
        predicted_label = predict_face(pca, scaler, knn, face, labels)
        predictions.append(predicted_label[0])
    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Main script to run the Eigenfaces algorithm
def main():
    # Select the video file using the file dialog
    video_path = select_video_file()
    if not video_path:
        print("No video file selected. Exiting.")
        return
    
    # Collect faces and labels from the video
    faces, labels = collect_faces(video_path)
    
    # Check if no faces were detected
    if faces.size == 0:
        print("Error: No faces detected from the video.")
        return
    
    # Train the Eigenfaces model (PCA + KNN)
    pca, scaler, knn = train_eigenfaces(faces, labels)
    
    # Compute accuracy on the collected faces
    accuracy = compute_accuracy(pca, scaler, knn, faces, labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Run the script
if __name__ == "__main__":
    main()
