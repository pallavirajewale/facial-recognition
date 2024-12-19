import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace

# Initialize the MTCNN face detector
detector = MTCNN()

# Initialize the DeepFace model (using FaceNet by default)
model_name = 'Facenet'  # You can change this to other models like 'VGG-Face', 'OpenFace', etc.

# Open the webcam or video file
cap = cv2.VideoCapture(0)

# Initialize counters for accuracy calculation
total_frames = 0
successful_detections = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame using MTCNN
    results = detector.detect_faces(frame_rgb)

    # Process each detected face
    for result in results:
        total_frames += 1
        # Get the bounding box and keypoints
        x, y, width, height = result['box']
        keypoints = result['keypoints']

        # Crop and convert the face region to grayscale for feature extraction
        face_region = frame[y:y + height, x:x + width]
        
        # Perform face recognition using DeepFace (FaceNet)
        try:
            # Using DeepFace for face recognition (extract embeddings)
            face_embedding = DeepFace.represent(face_region, model_name=model_name, enforce_detection=False)[0]['embedding']

            # Here you could compare the embedding to a known database of embeddings for recognition.
            # For simplicity, we just show the embedding in the output (you can extend this for real-time recognition)
            print(f"Face Embedding: {face_embedding[:5]}...")  # Displaying a portion of the embedding vector

            # Check if detection meets certain conditions to be counted as successful (simplified example)
            if np.linalg.norm(face_embedding) > 1.0:  # Example threshold (you can adjust this)
                successful_detections += 1
        except Exception as e:
            print(f"Error in face recognition: {e}")

        # Draw bounding box and facial landmarks
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        for key, point in keypoints.items():
            cv2.circle(frame, point, 2, (255, 0, 0), 5)

    # Calculate accuracy rate
    if total_frames > 0:
        accuracy_rate = (successful_detections / total_frames) * 100
    else:
        accuracy_rate = 0.0

    # Display the accuracy rate on the frame
    cv2.putText(frame, f"Accuracy Rate: {accuracy_rate:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the final output with face detection and recognition overlay
    cv2.imshow('Face and Recognition with Accuracy Rate', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

