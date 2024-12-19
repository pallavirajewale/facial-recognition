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
from concurrent.futures import ThreadPoolExecutor
from skimage.feature import graycomatrix, graycoprops
 
 
 
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
pnn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, alpha=0.001, random_state=42)
scaler = StandardScaler()
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 
 
 
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

# Example of multi-scale face detection (with parameters)
def multi_scale_face_detection(image):
    scales = [1.0, 0.75, 0.5]  # Original, 75%, and 50% of the original size
    all_faces = []
    for scale in scales:
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)
        detected_faces = face_detector.detect_faces(resized_image)
        for face in detected_faces:
            x, y, w, h = [int(coord / scale) for coord in face['box']]
            all_faces.append((image[y:y + h, x:x + w], (x, y, w, h)))
    return all_faces

# Using custom parameters when calling detect_faces
def detect_faces_with_custom_parameters(image):
    faces = face_detector.detect_faces(image, min_face_size=20, scale_factor=0.709, steps_threshold=[0.6, 0.7, 0.7])
    return faces

def detect_faces_with_haar(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(image[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in haar_faces]

def generate_face_embedding(face):
    try:
        embedding = DeepFace.represent(face, model_name="ArcFace", enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except:
        return None

def detect_moving_objects(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def augment_face(face):
    augmented_faces = [face]
    augmented_faces.append(cv2.flip(face, 1))  # Horizontal flip
    augmented_faces.append(cv2.GaussianBlur(face, (5, 5), 0))  # Gaussian blur
    return augmented_faces

def extract_glcm_features(face_gray):
    glcm = graycomatrix(face_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def analyze_facial_features(face_image):
    import face_recognition
    landmarks = face_recognition.face_landmarks(face_image)
    
    if not landmarks:
        return None
    
    # Example: Calculate the distance between the eyes
    left_eye = landmarks[0]['left_eye']
    right_eye = landmarks[0]['right_eye']
    eye_distance = np.linalg.norm(
        np.array(left_eye[0]) - np.array(right_eye[0])
    )
    
    # Example: Calculate jawline proportion
    jawline = landmarks[0]['chin']
    jaw_width = np.linalg.norm(
        np.array(jawline[0]) - np.array(jawline[-1])
    )
    
    features = {
        "eye_distance": eye_distance,
        "jaw_width": jaw_width,
    }
    return features

def analyze_glcm_features(face_image):
    # Convert to grayscale for texture analysis
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Create GLCM matrix with distance 1 and angle 0 (you can tweak these parameters)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract features like contrast, correlation, energy, and homogeneity
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    features = {
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity,
    }
    return features

def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images
    print("Select the folder containing dataset images.")
    root = Tk()
    root.withdraw()
    dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
    if not dataset_folder:
        print("No dataset folder selected. Exiting.")
        return
 
    def process_image(image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}. Skipping.")
                return None
 
            image = cv2.resize(image, (800, 800))  # Resize to a smaller resolution
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
            faces = mtcnn.detect_faces(rgb_image)
            for face in faces:
                x, y, w, h = face['box']
                try:
                    face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])[0]
                    return face_encoding, image_path.split(os.sep)[-1].split('.')[0], image
                except IndexError:
                    print(f"No face detected in {image_path}. Skipping.")
                    return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return None
 
    # Collect image paths
    image_paths = [os.path.join(root_dir, file)
                   for root_dir, _, files in os.walk(dataset_folder)
                   for file in files if file.lower().endswith(('jpg', 'jpeg', 'png'))]
 
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image, image_paths))
 
    # Filter out None results and populate the global lists
    for result in results:
        if result:
            encoding, name, image = result
            known_face_encodings.append(encoding)  #list1
            known_face_names.append(name)
            face_images.append(image)
 
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
    global frame_queue
    rtsp_url = "rtsp://admin:admin@192.168.1.200:554/avstream/channel=1/stream=0.sdp"
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()
 
    fps_start_time = time.time()
    print("Starting face recognition...")
 
    # Initialize frame_count here
    frame_count = 0
   
 
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1
 
            # Process every 5th frame to reduce load
            if frame_count % 5 == 0:
                processed_frame = cv2.resize(frame, (320, 240))  # Reduce resolution to improve speed
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
 
                # Initialize faces to an empty list before detection
                faces = []
 
                faces = mtcnn.detect_faces(rgb_frame)
 
                face_detected = False
                if faces:  # If faces are detected, save the full frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    frame_filename = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, processed_frame)
                    print(f"Saved frame: {frame_filename}")
 
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

                        facial_features = analyze_facial_features(face_image)

                        glcm_features = analyze_glcm_features(face_image)

                        face_filename = os.path.join(output_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(face_filename, face_image)
                        print(f"Saved face: {face_filename}")
 
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"{name} ({max_similarity * 100:.2f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        dataset_face = face_images[best_match_index] if best_match_index < len(face_images) else None
                        if dataset_face is not None:
                            dataset_face_resized = cv2.resize(dataset_face, (320, 240))  
                            processed_frame_resized = cv2.resize(processed_frame, (320, 240))  
                            combined_frame = np.hstack((dataset_face_resized, processed_frame_resized))
                        face_detected = True  
                    except IndexError:
                        print("Error in encoding. Skipping this face.")
 
                if face_detected:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    frame_filename = os.path.join(output_dir, f"annotated_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, processed_frame)
                    print(f"Saved annotated frame: {frame_filename}")
                fps_end_time = time.time()
                fps = 1 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face Recognition - Dataset vs Webcam", combined_frame)
 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
 
    cv2.destroyAllWindows()
load_dataset_and_register_faces()
recognize_faces()
 
 

