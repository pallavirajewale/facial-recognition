import cv2
from deepface import DeepFace
import numpy as np

# Initialize known faces and names
known_face_encodings = []
known_face_names = []

# Function to register a face
def register_face_from_webcam():
    global known_face_encodings, known_face_names
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to capture and save the face, or 'q' to quit registration.")
    name = input("Enter the name of the person to register: ")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Register Face', frame)

        # Press 's' to capture the face
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Analyze the captured image
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            print(f"Registered face for {name}. Attributes: {analysis}")
            
            # Store the encoding and name
            known_face_encodings.append(frame)
            known_face_names.append(name)
            break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to recognize faces
def recognize_faces():
    global known_face_encodings, known_face_names
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and compare faces
        for idx, known_face in enumerate(known_face_encodings):
            try:
                # Compare the current frame with the known face
                result = DeepFace.verify(frame, known_face, enforce_detection=False)
                if result['verified']:
                    # Get the matched name
                    name = known_face_names[idx]
                    confidence = result['distance'] * 100
                    # Display name and confidence on the frame
                    cv2.putText(frame, f"{name} ({confidence:.2f}%)", (50, 50 + idx * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error comparing face: {e}")
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to quit
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
