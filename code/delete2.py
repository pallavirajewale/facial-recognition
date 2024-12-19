import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import threading
from tkinter import filedialog
from tkinter import Tk
from mtcnn import MTCNN
# Store registered face encodings and names
known_face_encodings = []
known_face_names = []
face_embeddings = []

# Mediapipe face mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mtcnn = MTCNN()
# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000)
scaler = StandardScaler()

# Function to register faces from webcam
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images
    print("Select the folder containing dataset images.")
    root = Tk()
    root.withdraw()
    #folder_path = filedialog.askdirectory(title="Select Folder")

    #print(f"Selected folder: {folder_path}")
    dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
    if not dataset_folder:
        print("No dataset folder selected. Exiting.")
        return
 
    for root_dir, subdirs, files in os.walk(dataset_folder):
        for image_name in files:
            if image_name.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root_dir, image_name)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
                faces = mtcnn.detect_faces(rgb_image)
                for face in faces:
                    x, y, w, h = face['box']
                    try:
                        face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(image_name.split('.')[0])
                        face_embeddings.append(np.array(face_encoding[0]))
                        # face_images.append(image)
                    except IndexError:
                        print(f"Face not detected in {image_name}. Skipping.")

    # Train the PNN on the registered faces
    print("Training PNN model...")
    face_embeddings_scaled = scaler.fit_transform(face_embeddings)  # Normalize the embeddings
    pnn_model.fit(face_embeddings_scaled, known_face_names)
    print("PNN model trained successfully.")

# Function to recognize faces using webcam with face mesh and PNN
def recognize_faces():
    global frame_queue
    rtsp_url = "rtsp://admin:admin@192.168.1.200:554/avstream/channel=1/stream=0.sdp"
 
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()
 
    fps_start_time = time.time()
    print("Starting face recognition...")
 
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
 
            processed_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Face recognition
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Process face mesh
            results = face_mesh.process(rgb_frame)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Predict the face identity using PNN
                face_encoding_scaled = scaler.transform([face_encoding])  # Normalize the input
                name = pnn_model.predict(face_encoding_scaled)[0]

                # Compute accuracy
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                accuracy = (1 - face_distances[np.argmin(face_distances)]) * 100
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

def main():
    print("Do you want to load the pre-trained dataset or add new data?")
    print("1. Load pre-trained dataset")
    print("2. Add new dataset")
    choice = input("Enter your choice: ")

    if choice == '1':
        load_dataset()
        # pnn_model, scaler = load_model_and_scaler()
    elif choice == '2':
        load_dataset_and_register_faces()
    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    main()
recognize_faces()