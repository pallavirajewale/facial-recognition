#FINAL CODE WITH MATCHING(final)
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
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import os
import time
import threading
 
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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
 
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
 
def crop_visible_features(face, box):
    """Focus on visible regions (eyes and forehead) for robustness."""
    x, y, w, h = box
    forehead = face[:h//3, :]  
    eyes = face[h//3:h//2, :]  
    visible_features = np.concatenate((forehead.flatten(), eyes.flatten()))  
    return visible_features
 
def generate_face_embedding(face):
    """Generate face embedding using DeepFace."""
    try:
        embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except:
        return None
   
def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
 
def extract_facial_features(image, face_landmarks):
    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
    lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
 
    h, w, _ = image.shape
 
    def extract_points(indices):
        return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
 
    left_eye_points = extract_points(left_eye_indices)
    right_eye_points = extract_points(right_eye_indices)
    lips_points = extract_points(lips_indices)
 
    return left_eye_points, right_eye_points, lips_points
 
def compute_glcm_features(gray_frame):
    glcm = np.zeros((256, 256), dtype=np.float64)
    rows, cols = gray_frame.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            current_pixel = gray_frame[i, j]
            right_pixel = gray_frame[i, j + 1]
            glcm[current_pixel, right_pixel] += 1
    glcm /= glcm.sum()
    contrast = np.sum([(i - j) ** 2 * glcm[i, j] for i in range(256) for j in range(256)])
    dissimilarity = np.sum([abs(i - j) * glcm[i, j] for i in range(256) for j in range(256)])
    homogeneity = np.sum([glcm[i, j] / (1.0 + abs(i - j)) for i in range(256) for j in range(256)])
    return contrast, dissimilarity, homogeneity
 
def apply_crystal_contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image
 
def draw_facial_features(image, left_eye, right_eye, lips):
    for point in left_eye:
        cv2.circle(image, point, 2, (0, 255, 0), -1)
    for point in right_eye:
        cv2.circle(image, point, 2, (0, 255, 255), -1)
    for point in lips:
        cv2.circle(image, point, 2, (255, 0, 0), -1)
 
def calculate_color_features(frame):
    resized_frame = cv2.resize(frame, (320, 240))
 
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
 
    mean_bgr = cv2.mean(resized_frame)[:3]
    mean_hsv = cv2.mean(hsv_frame)[:3]
 
    h_hist = cv2.calcHist([hsv_frame], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv_frame], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv_frame], [2], None, [16], [0, 256])
 
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
 
    color_hist_features = np.concatenate([h_hist, s_hist, v_hist])
 
    return mean_bgr, mean_hsv, color_hist_features
 
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
    with open('pnn_model.pkl', 'wb') as model_file:
        pickle.dump(pnn_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Model and scaler saved.")
 
def apply_crystal_contrast_enhancement(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   
    enhanced_frame = clahe.apply(gray_frame)
   
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
   
    return enhanced_frame
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
           
            enhanced_frame = apply_crystal_contrast_enhancement(processed_frame)
           
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            faces = mtcnn.detect_faces(rgb_frame)
 
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
                        matched_image = face_images[best_match_index]
                    else:
                        name = "Unknown"
                        matched_image = None
 
                    cv2.rectangle(enhanced_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(enhanced_frame, f"{name} ({max_similarity * 100:.2f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
                    if matched_image is not None:
                        matched_image_resized = cv2.resize(matched_image, (100, 100))
                        enhanced_frame[10:110, 10:110] = matched_image_resized
 
                except IndexError:
                    print("Error in encoding. Skipping this face.")
 
            
            fps_end_time = time.time()
            fps = 1 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            cv2.putText(enhanced_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
            cv2.imshow("Face Recognition", enhanced_frame)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
    cv2.destroyAllWindows()
 
# Main
load_dataset_and_register_faces()
recognize_faces()
 