import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import pickle
import queue
from deepface import DeepFace
from tkinter import filedialog, Tk

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import threading
import mediapipe as mp
from skimage import exposure

camera_urls = [
    "rtsp://admin:admin@192.168.1.200:554/avstream/channel=1/stream=0.sdp",  # Camera 1
    "rtsp://admin:admin@192.168.1.201:554/avstream/channel=1/stream=0.sdp",  # Camera 2
    # Add more camera URLs if needed
]
frame_queues = [queue.Queue(maxsize=2) for _ in range(len(camera_urls))]  # Create one queue per cameraknown_face_encodings = []
known_face_names = []
known_face_encodings = []
face_images = []
face_embeddings = []
face_detector = MTCNN()
mtcnn = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

# File paths for saving and loading dataset and model
dataset_file = 'face_dataset.pkl'
model_file = 'pnn_model.pkl'
scaler_file = 'scaler.pkl'

def save_dataset():
    """Save the dataset to a pickle file."""
    with open(dataset_file, 'wb') as file:
        pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, file)
    print("Dataset saved successfully.")

def load_dataset():
    """Load the dataset from a pickle file."""
    global known_face_encodings, known_face_names
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as file:
            data = pickle.load(file)
            known_face_encodings = data.get('encodings', [])
            known_face_names = data.get('names', [])
        print("Dataset loaded successfully.")
    else:
        print("No pre-trained dataset found. Starting from scratch.")

def save_model():
    """Save the trained model and scaler."""
    with open(model_file, 'wb') as pnn_model_file:  # Rename this variable to avoid conflict
        pickle.dump(pnn_model, pnn_model_file)
    with open(scaler_file, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Model and scaler saved.")



def capture_frames(camera_id, rtsp_url, frame_queue):
    video_capture = cv2.VideoCapture(rtsp_url)
    if not video_capture.isOpened():
        print(f"Error opening RTSP stream for Camera {camera_id}. Please check the URL.")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Failed to grab frame from Camera {camera_id}. Skipping this frame.")
            continue  # Skip to the next frame if failed to capture
        
        if frame is not None:
            processed_frame = cv2.resize(frame, (640, 480))  # Resize frame to (640, 480)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

            if not frame_queue.full():
                frame_queue.put(processed_frame)  # Add frame to the queue for further processing
        else:
            print(f"Error: Received an empty frame from Camera {camera_id}. Skipping this frame.")
            continue  # Skip processing if the frame is None

    video_capture.release()


 
def detect_faces(image):
    """Detect faces in the image using MTCNN."""
    results = face_detector.detect_faces(image)
    faces = []
    for result in results:
        box = result['box']
        x, y, width, height = box
        face = image[y:y+height, x:x+width]
        faces.append((face, box))
    return faces

def load_model_and_scaler():
    """Load the trained model and scaler from pickle files."""
    try:
        with open(model_file, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_file, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found. Please ensure you have trained the model.")
        return None, None
    
def generate_face_embedding(face):
    """Generate face embedding using DeepFace."""
    try:
        embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except:
        return None
 
def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
 
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images
    print("Select the folder containing dataset images.")
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder")

    print(f"Selected folder: {folder_path}")
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
                   
    print("Dataset loaded and faces registered.")
    print("Training PNN model...")
 
    face_embeddings = np.vstack(known_face_encodings)  
    face_embeddings_scaled = scaler.fit_transform(face_embeddings)
    pnn_model.fit(face_embeddings_scaled, known_face_names)
    print("PNN model trained successfully.")
    save_model()
    save_dataset()
    with open('pnn_model.pkl', 'wb') as model_file:
        pickle.dump(pnn_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Model and scaler saved.")
 
def load_model_and_scaler():
    try:
        with open('pnn_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found. Please ensure you have trained the model.")
        exit()
 
pnn_model, scaler = load_model_and_scaler()
 
def extract_color_features(face):
    """Extract color histogram features from the face image."""
    hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv_face], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_face], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_face], [2], None, [256], [0, 256])
    return hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()
 
def enhance_contrast(face):
    """Enhance the contrast of the face using adaptive histogram equalization."""
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    enhanced_face = exposure.equalize_adapthist(gray_face, clip_limit=0.03)
    return enhanced_face
 
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
    #global frame_queue
 
    #capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, frame_queue))
    #capture_thread.daemon = True
    #capture_thread.start()
 
    fps_start_time = time.time()
    print("Starting face recognition...")
    threads = []
    for i, rtsp_url in enumerate(camera_urls):
        capture_thread = threading.Thread(target=capture_frames, args=(i, rtsp_url, frame_queues[i]))
        capture_thread.daemon = True  # Ensure the thread exits when the main program exits
        capture_thread.start()
        threads.append(capture_thread)
 
    while True:
        for i, frame_queue in enumerate(frame_queues):
            if not frame_queue.empty():
                frame = frame_queue.get()
 
                processed_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
 
                faces = mtcnn.detect_faces(rgb_frame)
 
                combined_frame = processed_frame
 
            for face in faces:
                x, y, w, h = face['box']
                try:
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]
                    face_encoding_scaled = scaler.transform(np.reshape(face_encoding, (1, -1)))
                    similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
                    max_similarity = max(similarities)
                    best_match_index = similarities.index(max_similarity)
 
                    if max_similarity > 0.6:
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
           
                    face_image = processed_frame[y:y + h, x:x + w]
                    color_features = extract_color_features(face_image)
                    enhanced_contrast = enhance_contrast(face_image)
 
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
def detect_faces(image):
    """Detect faces in the image using MTCNN or any other method."""
    # This function should return detected faces from the image
    # Dummy implementation for the purpose of the example:
    faces = [(None, (100, 100, 200, 200))]  # Just an example: x, y, width, height of detected face
    return faces

def main():
    print("Do you want to load the pre-trained dataset or add new data?")
    print("1. Load pre-trained dataset")
    print("2. Add new dataset")
    choice = input("Enter your choice: ")

    if choice == '1':
        load_dataset()
        pnn_model, scaler = load_model_and_scaler()
    elif choice == '2':
        load_dataset_and_register_faces()
    else:
        print("Invalid choice. Exiting.")
        return

   

if __name__ == "__main__":
    main()
recognize_faces()