import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import pickle
import queue
from deepface import DeepFace
from tkinter import filedialog
from tkinter import Tk
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import threading
import mediapipe as mp
from skimage import exposure

frame_queue = queue.Queue(maxsize=2)
known_face_encodings = []
known_face_names = []
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
    
    with open('pnn_model1.pkl', 'wb') as model_file:
        pickle.dump(pnn_model, model_file)
    with open('scaler1.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Model and scaler saved.")

    with open('data_set1.pkl', 'wb') as model_file:
        pickle.dump(known_face_encodings, model_file)
    with open('names1.pkl', 'wb') as scaler_file:
        pickle.dump(known_face_names, scaler_file)
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

def load_dataset():
    global known_face_encodings, known_face_names
    try:
        with open('data_set1.pkl', 'rb') as model_file:
            known_face_encodings = pickle.load(model_file)
        if isinstance(known_face_encodings, list):
            print("The dataset has been successfully converted to a list.")
        with open('names1.pkl', 'rb') as scaler_file:
            known_face_names = pickle.load(scaler_file)
        if isinstance(known_face_names, list):
            print("The scaler has been successfully converted to a list.")
        print("Model2 and scaler loaded successfully.")

        face_embeddings = np.vstack(known_face_encodings)  
        face_embeddings_scaled = scaler.fit_transform(face_embeddings)
        pnn_model.fit(face_embeddings_scaled, known_face_names)
        print("PNN model trained successfully.")

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

            faces = mtcnn.detect_faces(rgb_frame)

            combined_frame = processed_frame

            for face in faces:
                x, y, w, h = face['box']
                try:
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]
                    # face_encoding_scaled = scaler.transform(np.reshape(face_encoding, (1, -1)))
                    similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
                    max_similarity = max(similarities)
                    best_match_index = similarities.index(max_similarity)

                    if max_similarity > 0.9: 
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
            
                    face_image = processed_frame[y:y + h, x:x + w]
                    # color_features = extract_color_features(face_image)
                    # enhanced_contrast = enhance_contrast(face_image)

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
# load_dataset_and_register_faces()
# recognize_faces()
def main():
    print("Do you want to load the pre-trained dataset or add new data?")
    print("1. Load pre-trained dataset")
    print("2. Add new dataset")
    choice = input("Enter your choice: ")

    if choice == '1':
        load_dataset()
        recognize_faces()
    elif choice == '2':
        load_dataset_and_register_faces()
        recognize_faces()
    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    main()
