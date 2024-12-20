import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import os
import time
import threading
from tkinter import filedialog
from tkinter import Tk
from mtcnn import MTCNN
import queue
import pickle

frame_queue = queue.Queue(maxsize=2)

# Store registered face encodings and names
known_face_encodings = []
known_face_names = []
face_embeddings = []
face_images = []

mtcnn = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)


haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# File paths for saving and loading dataset and model
dataset_file = 'large_dataset3.pkl'   
dataset_names = 'largesetname2.pkl'
face_names = 'large_face_names2.pkl'

# dataset_file = 'face_dataset3.pkl'
# dataset_names = 'datasetname2.pkl'
# face_names = 'face_names2.pkl'
# model_file = 'pnn_model.pkl'
# scaler_file = 'scaler.pkl'

# Store the last recognized name and encoding for consistency
last_recognized_name = None
last_recognized_encoding = None

def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
 


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
                        face_images.append(image)
                    except IndexError:
                        print(f"Face not detected in {image_name}. Skipping.")
    with open(dataset_file, 'wb') as model_file:
        pickle.dump(known_face_encodings, model_file)
    with open(dataset_names, 'wb') as scaler_file:
        pickle.dump(known_face_names, scaler_file)
    with open(face_names, 'wb') as scaler_file:
        pickle.dump(face_images, scaler_file)
    print("pickle file saved.") 

def load_dataset():
    global known_face_encodings, known_face_names, face_images
    with open(dataset_file, 'rb') as file:
    # Load the data from the pickle file
       known_face_encodings = pickle.load(file)
    if isinstance(known_face_encodings, list):
        print("The dataset has been successfully converted to a list.")

    with open(dataset_names, 'rb') as file:
    # Load the data from the pickle file
         known_face_names = pickle.load(file)
    if isinstance(known_face_names, list):
        print("The names has been successfully converted to a list.")

    with open(face_names, 'rb') as file:
    # Load the data from the pickle file
         face_images = pickle.load(file)
    if isinstance(face_images, list):
        print("The images has been successfully converted to a list.")

def main():
    print("Do you want to load the pre-trained dataset or add new data?")
    print("1. Load pre-trained dataset")
    print("2. Add new dataset")
    choice = input("Enter your choice: ")

    if choice == '1':
        load_dataset()
    elif choice == '2':
        load_dataset_and_register_faces()
    else:
        print("Invalid choice. Exiting.")
        return
    
def capture_frames(rtsp_url, frame_queue):
    video_capture = cv2.VideoCapture(rtsp_url)
    if not video_capture.isOpened():
        print("Error opening RTSP stream. Please check the URL.")
        return
 
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
 
        if not frame_queue.full():
            frame_queue.put(frame)
 
    video_capture.release()
    

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # Face recognition
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            results = face_mesh.process(rgb_frame)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compute accuracy
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                accuracy = (1 - face_distances[np.argmin(face_distances)]) * 100
                accuracy = min(accuracy, 100.0)  # Ensure accuracy doesn't exceed 100%

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw the name and accuracy
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
            if results.multi_face_landmarks:
                combined_frame = processed_frame
                faces = mtcnn.detect_faces(rgb_frame)
                for face in faces:
                    x, y, w, h = face['box']
                    try:
                        face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]
                        similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
                        max_similarity = max(similarities)
                        best_match_index = similarities.index(max_similarity)
    
                        # Only update name if the match is strong enough
                        if max_similarity > 0.6:
                            name = known_face_names[best_match_index]
                            dataset_face = face_images[best_match_index] if best_match_index < len(face_images) else None
                            # Track the last recognized name and encoding
                            last_recognized_name = name
                            last_recognized_encoding = face_encoding
                        else:
                            name = "Unknown"
                            last_recognized_name = None  
                            dataset_face = None
                        if dataset_face is not None:
                            dataset_face_resized = cv2.resize(dataset_face, (320, 240))  # Resize to match the live stream frame
                            processed_frame_resized = cv2.resize(processed_frame, (320, 240))
                            combined_frame = np.hstack((dataset_face_resized, processed_frame_resized))  # Combine images horizontally

                        # Show matched image from the dataset (if any)
                        if last_recognized_name:
                            dataset_face = face_images[best_match_index] if best_match_index < len(face_images) else None
                            if dataset_face is not None:
                                dataset_face_resized = cv2.resize(dataset_face, (320, 240))
                                processed_frame_resized = cv2.resize(processed_frame, (320, 240))
                                combined_frame = np.hstack((dataset_face_resized, processed_frame_resized))

                        # Draw bounding box and name on the live video frame
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"{name} ({max_similarity * 100:.2f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        face_image = processed_frame[y:y + h, x:x + w]
                        
    
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"{name} ({max_similarity * 100:.2f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        dataset_face = face_images[best_match_index] if best_match_index < len(face_images) else None
                        if dataset_face is not None:
                            dataset_face_resized = cv2.resize(dataset_face, (320, 240))  
                            processed_frame_resized = cv2.resize(processed_frame, (320, 240))
                            combined_frame = np.hstack((dataset_face_resized, processed_frame_resized))  
                    except IndexError:
                        print("Error in encoding. Skipping this face.")
                fps_end_time = time.time()
                fps = 1 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face Recognition - Dataset vs Webcam", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
recognize_faces()
