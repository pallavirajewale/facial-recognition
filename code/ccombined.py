import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
import pickle

# Store registered face encodings and names
known_face_encodings = []
known_face_names = []
face_embeddings = []

# Mediapipe face mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

# Function to register faces from webcam
def register_face_from_webcam():
    global known_face_encodings, known_face_names, face_embeddings
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to capture and save the face, or 'q' to quit registration.")
    name = input("Enter the name of the person to register: ")

    temp_encodings = []  # Store multiple encodings for robustness

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Register Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if face_encodings:
                temp_encodings.append(face_encodings[0])
                print("Frame captured for registration.")
                if len(temp_encodings) >= 10:  # Capture at least 10 frames
                    avg_encoding = np.mean(temp_encodings, axis=0)  # Average encoding
                    known_face_encodings.append(avg_encoding)
                    known_face_names.append(name)
                    face_embeddings.append(avg_encoding)
                    print(f"Face registered for {name}.")
                    break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Train the PNN on the registered faces
    print("Training PNN model...")
    face_embeddings_scaled = scaler.fit_transform(face_embeddings)
    pnn_model.fit(face_embeddings_scaled, known_face_names)
    print("PNN model trained successfully.")

    # Save the model and scaler
    with open('pnn_model.pkl', 'wb') as model_file:
        pickle.dump(pnn_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Model and scaler saved.")

# Function to recognize faces using webcam with face mesh and PNN
def recognize_faces():
    global known_face_encodings, known_face_names, face_embeddings, pnn_model, scaler
    video_capture = cv2.VideoCapture(0)
    fps_start_time = time.time()

    # Load the model and scaler from files
    with open('pnn_model.pkl', 'rb') as model_file:
        pnn_model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process face mesh
        results = face_mesh.process(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Predict the face identity using PNN
            face_encoding_scaled = scaler.transform([face_encoding])  # Normalize the input
            probabilities = pnn_model.predict_proba(face_encoding_scaled)[0]
            confidence = max(probabilities) * 100

            if confidence < 80:  # Reject predictions below 80% confidence
                name = "Unknown"
            else:
                name = pnn_model.predict(face_encoding_scaled)[0]

            # Compute accuracy
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            accuracy = (1 - face_distances[np.argmin(face_distances)]) * 100
            accuracy = min(accuracy, 100.0)  # Ensure accuracy doesn't exceed 100%

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw the name and confidence
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{name} ({confidence:.2f}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw face mesh
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # Display FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Display the result
        cv2.imshow('Face Recognition with Mesh', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    while True:
        print("1. Register a new face")
        print("2. Recognize faces")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            register_face_from_webcam()
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")