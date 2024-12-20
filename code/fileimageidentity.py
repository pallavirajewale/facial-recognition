++++++++++++++++++++++++++++import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
from mtcnn import MTCNN
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import filedialog
from tkinter import Tk

# Store registered face encodings and names
known_face_encodings = []
known_face_names = []
face_embeddings = []

# Mediapipe face mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

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

# Function to calculate cosine similarity between two face encodings
def calculate_similarity(encoding1, encoding2):
    return cosine_similarity([encoding1], [encoding2])[0][0]

def calculate_color_features(frame):
    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (320, 240))

    # Convert the frame to the HSV color space for color features
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # Calculate mean color in RGB and HSV
    mean_bgr = cv2.mean(resized_frame)[:3]
    mean_hsv = cv2.mean(hsv_frame)[:3]

    # Calculate color histogram (using 16 bins for each channel in HSV)
    h_hist = cv2.calcHist([hsv_frame], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv_frame], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv_frame], [2], None, [16], [0, 256])

    # Normalize the histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    # Combine histograms into one feature vector
    color_hist_features = np.concatenate([h_hist, s_hist, v_hist])

    return mean_bgr, mean_hsv, color_hist_features

# Function to register faces from dataset folder
def load_dataset_and_register_faces():
    global known_face_encodings, known_face_names, face_embeddings
    print("Select the folder containing dataset images.")
    
    # Open file dialog to select dataset folder
    root = Tk()
    root.withdraw()  # Hide the Tkinter window
    dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
    
    if not dataset_folder:
        print("No dataset folder selected. Exiting.")
        return
    
    # Loop through dataset folder and load images from subfolders
    for root_dir, subdirs, files in os.walk(dataset_folder):
        for image_name in files:
            if image_name.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root_dir, image_name)
        
            # Load and process each image in the dataset
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            # Calculate color features for the image
            mean_bgr, mean_hsv, color_hist_features = calculate_color_features(image)

            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(image_name.split('.')[0])  # Use image name as the person's name
                face_embeddings.append(face_encoding)

    print("Dataset loaded and faces registered.")

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

def apply_crystal_contrast_enhancement(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object with parameters (clip limit, tile grid size)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to the grayscale image
    enhanced_frame = clahe.apply(gray_frame)
    
    # Convert back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
    
    return enhanced_frame

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

        # Apply Crystal Contrast Enhancement
        enhanced_frame = apply_crystal_contrast_enhancement(frame)

        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        # Calculate color features for the frame
        mean_bgr, mean_hsv, color_hist_features = calculate_color_features(enhanced_frame)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process face mesh
        results = face_mesh.process(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Reshape face encoding to 2D array before scaling
            face_encoding_scaled = scaler.transform(np.reshape(face_encoding, (1, -1)))

            # Calculate cosine similarity for better face matching
            similarities = [calculate_similarity(face_encoding, known_encoding) for known_encoding in known_face_encodings]
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)

            if max_similarity > 0.6:  # Adjust the threshold based on experimentation
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            # Draw a rectangle around the face and display results
            cv2.rectangle(enhanced_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(enhanced_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(enhanced_frame, f"{name} ({max_similarity*100:.2f}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw face mesh
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(enhanced_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                          mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Display the resulting frame
        cv2.imshow("Face Recognition with Webcam", enhanced_frame)

        # FPS counter
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        cv2.putText(enhanced_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    video_capture.release()
    cv2.destroyAllWindows()

# Main execution
load_dataset_and_register_faces()  # Load dataset and register faces
recognize_faces()  # Recognize faces using webcam


