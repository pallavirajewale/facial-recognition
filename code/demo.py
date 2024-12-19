import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN
import pickle
from retinaface import RetinaFace
from tkinter import filedialog                                                               
from tkinter import Tk
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mediapipe import solutions as mp
from threading import Thread
import pickle
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import os
import time
import dlib

# Load the face detector and shape predictor from Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Initialize the video capture and MTCNN detector
video_capture = cv2.VideoCapture(0)  # 0 for webcam, replace with video path for file
detector = MTCNN()

# Resize video for performance
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables to manage frame skipping
frame_count = 0
frame_skip = 5  # Process every 5th frame

# Store registered face encodings, names, and images
known_face_encodings = []
known_face_names = []
face_images = []
face_embeddings = []

# Function to capture frames
frame = None
stop_capture = False

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Function to extract frames from a video
def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    while success:
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
        success, frame = video.read()

    video.release()
    print(f"Extracted {count} frames from {video_path}.")
    return count

def capture_frames():
    global frame, stop_capture
    while not stop_capture:
        ret, temp_frame = video_capture.read()
        if ret:
            frame = cv2.resize(temp_frame, (640, 480))
        else:
            break


# Start frame capture thread
capture_thread = Thread(target=capture_frames, daemon=True)
capture_thread.start()


# Mediapipe face mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

# Function to calculate cosine similarity between two face encodings
def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

def extract_facial_features(image, face_landmarks):
    # Define landmark indices for the left eye, right eye, and lips
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

# Manual GLCM computation function
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

# Crystal Contrast Enhancement Function (Assumed)
def apply_crystal_contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def draw_facial_features(image, left_eye, right_eye, lips):
    # Draw left eye
    for point in left_eye:
        cv2.circle(image, point, 2, (0, 255, 0), -1)
    # Draw right eye
    for point in right_eye:
        cv2.circle(image, point, 2, (0, 255, 255), -1)
    # Draw lips
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

# Function to load dataset and register faces
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images, face_embeddings
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
                    face_encoding = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])[0]
                    
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(image_name.split('.')[0])
                    face_images.append(image)
                    face_embeddings.append(face_encoding)
                    

    print("Dataset loaded and faces registered.")
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

# Function to recognize faces in uploaded video
def recognize_faces_from_video():
    global known_face_encodings, known_face_names, face_images
    video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

    if not video_file:
        print("No video file selected. Exiting.")
        return

    video_capture = cv2.VideoCapture(video_file)

    with open('pnn_model.pkl', 'rb') as model_file:
        pnn_model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgb_frame)

        enhanced_frame = apply_crystal_contrast_enhancement(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]
            
            similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)

            if max_similarity > 0.6:
                name = known_face_names[best_match_index]
                matched_image = face_images[best_match_index]
            else:
                name = "Unknown"
                matched_image = None

        results = face_mesh.process(rgb_frame)
        #face_locations = face_recognition.face_locations(rgbTo adapt your code for uploading a video dataset and performing face recognition, comparison, and displaying results using video streaming and dataset images simultaneously, I’ve restructured the core logic and added functionality to load and process videos directly from the dataset. Below is a breakdown of what needs to be done to integrate your system with a video dataset upload and display functionality).

### Key Steps:
#1. **Upload Video Dataset**: Implement a function to load videos and compare frames with the database of registered faces.
#2. **Compare Video Frames with Database**: For each frame in the video, extract features and compare them with the known faces from the dataset.
#3. **Display Results Simultaneously**: Display the live camera stream alongside a matched image from the dataset.

#Here’s the modified version of your code to include these functionalities
def load_video_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_images, face_embeddings
    print("Select the folder containing video dataset.")
    root = Tk()
    root.withdraw()  # Hide Tkinter window
    dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
    if not dataset_folder:
        print("No dataset folder selected. Exiting.")
        return

    # Loop through all video files in the folder
    for video_name in os.listdir(dataset_folder):
        if video_name.endswith(('.mp4', '.avi', '.mov')):  # Check for video files
            video_path = os.path.join(dataset_folder, video_name)
            capture_video(video_path)  # Capture frames from the video and register faces
            process_video_for_registration(video_path)

def process_video_for_registration(video_path):
    """Processes a video to register faces into the database."""
    global known_face_encodings, known_face_names, face_images, face_embeddings
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(video_path.split('/')[-1].split('.')[0])
            face_images.append(frame)
            face_embeddings.append(face_encoding)

    video_capture.release()
    print(f"Faces from {video_path} registered.")

def calculate_similarity(encoding1, encoding2):
    """Calculates cosine similarity between two encodings."""
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

def recognize_faces_from_video_stream():
    """Recognizes faces from a live webcam stream."""
    global known_face_encodings, known_face_names, face_images
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

            similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)

            if max_similarity > 0.6:
                name = known_face_names[best_match_index]
                matched_image = face_images[best_match_index]
            else:
                name = "Unknown"
                matched_image = None

            # Draw rectangle and name on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({max_similarity*100:.2f}%)", (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if matched_image is not None:
                aspect_ratio = matched_image.shape[1] / matched_image.shape[0]
                new_height = frame.shape[0]
                new_width = int(aspect_ratio * new_height)
                matched_image_resized = cv2.resize(matched_image, (new_width, new_height))
                combined_frame = np.hstack((frame, matched_image_resized))
                cv2.imshow("Face Recognition - Webcam & Dataset Image", combined_frame)
            else:
                cv2.imshow("Face Recognition - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
         
def process_video_for_registration(video_path):
    """Processes a video to register faces into the database."""
    global known_face_encodings, known_face_names, face_images, face_embeddings
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(video_path.split('/')[-1].split('.')[0])
            face_images.append(frame)
            face_embeddings.append(face_encoding)

    video_capture.release()
    print(f"Faces from {video_path} registered.")

def calculate_similarity(encoding1, encoding2):
    """Calculates cosine similarity between two encodings."""
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

def capture_video(video_path):
    global known_face_encodings, known_face_names, face_images, face_embeddings

    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = mtcnn.detect_faces(rgb_frame)

        # Extract facial features
        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]
            
            known_face_encodings.append(face_encoding)
            known_face_names.append(video_path.split('/')[-1])  # Use video file name as identity
            face_images.append(frame)  # Store the full image
            face_embeddings.append(face_encoding)

    video_capture.release()
    print(f"Faces from {video_path} registered.")

def recognize_faces_from_video_stream():
    global known_face_encodings, known_face_names, face_images
    video_capture = cv2.VideoCapture(0)  # Use webcam for live stream

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        faces = mtcnn.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

            # Calculate cosine similarity
            similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)

            if max_similarity > 0.6:  # If similarity is high enough
                name = known_face_names[best_match_index]
                matched_image = face_images[best_match_index]
            else:
                name = "Unknown"
                matched_image = None

            # Draw rectangle and name on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({max_similarity*100:.2f}%)", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show matching image from dataset alongside webcam stream
            if matched_image is not None:
                aspect_ratio = matched_image.shape[1] / matched_image.shape[0]
                new_height = frame.shape[0]
                new_width = int(aspect_ratio * new_height)
                matched_image_resized = cv2.resize(matched_image, (new_width, new_height))
                combined_frame = np.hstack((frame, matched_image_resized))
                cv2.imshow("Face Recognition - Webcam & Dataset Image", combined_frame)
            else:
                cv2.imshow("Face Recognition - Webcam", frame)

        # Wait for user to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def calculate_similarity(encoding1, encoding2):
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

# Main execution
load_video_dataset_and_register_faces()  # Load dataset and register faces
recognize_faces_from_video_stream()  # Start face recognition on live webcam feed
