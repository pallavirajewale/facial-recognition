import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
 
# Store registered face encodings and names
known_face_encodings = []
known_face_names = []
 
# Mediapipe face mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
# Function to register faces from webcam
def register_face_from_webcam():
    global known_face_encodings, known_face_names
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to capture and save the face, or 'q' to quit registration.")
    name = input("Enter the name of the person to register: ")
 
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
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                print(f"Face registered for {name}.")
                break
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break
 
    video_capture.release()
    cv2.destroyAllWindows()
 
# Function to recognize faces using webcam with face mesh
def recognize_faces():
    global known_face_encodings, known_face_names
    video_capture = cv2.VideoCapture(0)
    fps_start_time = time.time()
 
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
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
 
            name = "Unknown"
            accuracy = 0.0
 
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    accuracy = (1 - face_distances[best_match_index]) * 100
                    accuracy = min(accuracy, 100.0)  # Ensure accuracy doesn't exceed 100%
 
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw the name and accuracy
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
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