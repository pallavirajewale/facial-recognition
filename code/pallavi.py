import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
from mtcnn import MTCNN
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

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
detector=MTCNN()

# Initialize the PNN classifier
pnn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, alpha=0.01, random_state=42)
scaler = StandardScaler()

def extract_critical_features(image, face_landmarks):
    h, w, _ = image.shape
    left_eye_indices = [33, 133, 159, 158, 144, 145, 153]
    right_eye_indices = [362, 263, 387, 386, 373, 374, 380]
    jawline_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Just an example, you can expand this for better jawline detail

    def get_points(indices):
        return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

    left_eye_points = get_points(left_eye_indices)
    right_eye_points = get_points(right_eye_indices)
    jawline_points = get_points(jawline_indices)

    return left_eye_points, right_eye_points, jawline_points

def extract_critical_features(image, face_landmarks):
    # Define the critical landmarks for face recognition
    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
    lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    nose_indices = [1, 2, 4, 5]  # Nose landmarks can be expanded

    h, w, _ = image.shape
    
    def extract_points(indices):
        return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

    # Extract points for eyes, lips, and nose
    left_eye_points = extract_points(left_eye_indices)
    right_eye_points = extract_points(right_eye_indices)
    lips_points = extract_points(lips_indices)
    nose_points = extract_points(nose_indices)
    
    # Combine all extracted points into one feature vector (optional)
    critical_features = {
        "left_eye": left_eye_points,
        "right_eye": right_eye_points,
        "lips": lips_points,
        "nose": nose_points
    }
    
    return critical_features

def extract_eye_features(face_landmarks, image):
    # Define the indices for the left and right eyes
    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
    
    # Extract points for both eyes
    h, w, _ = image.shape
    def extract_points(indices):
        return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
    
    left_eye_points = extract_points(left_eye_indices)
    right_eye_points = extract_points(right_eye_indices)
    
    return left_eye_points, right_eye_points

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

# Enhanced face detection function to handle obstacles (e.g., masks, caps, helmets)
def detect_faces_with_obstacles(frame):
    # Use MTCNN for face detection with better handling for masks and caps
    detector = MTCNN()
    faces = detector.detect_faces(frame)

    # If no faces are found by MTCNN, use face_recognition as fallback
    if len(faces) == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        return face_locations, None

    # If faces are detected by MTCNN, extract bounding boxes
    face_locations = []
    for face in faces:
        x, y, w, h = face['box']
        face_locations.append((y, x + w, y + h, x))

    return face_locations, faces

def process_webcam_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Detect faces using the enhanced face detection
        face_locations, faces = detect_faces_with_obstacles(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = face['box']
                # Optionally, draw landmarks or bounding boxes for faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Facial Feature Extraction', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def apply_crystal_contrast_enhancement(frame):
    """
    Apply Crystal Contrast Enhancement to the given frame.
    The method adjusts the contrast of the image to improve the clarity of features.
    """
    # Convert to YUV color space
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization on the Y channel (luminance)
    yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
    
    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
    
    return enhanced_frame

# Enhanced face detection function with MTCNN
def detect_faces(frame):
    faces = detector.detect_faces(frame)
    face_locations = []
    for face in faces:
        x, y, w, h = face['box']
        face_locations.append((y, x + w, y + h, x))
    return face_locations

# Function to register faces from webcam
def register_face_from_webcam():
    global known_face_encodings, known_face_names, face_embeddings
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to capture and save the face, or 'q' to quit registration.")
    name = input("Enter the name of the person to register: ")

    temp_encodings = []  # Store multiple encodings for robustness

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces using the enhanced detection function
        face_locations, faces = detect_faces_with_obstacles(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Register Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Capture face encodings when the user presses 's'
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                temp_encodings.append(face_encodings[0])
                print("Frame captured for registration.")
                if len(temp_encodings) >= 10:  # Capture at least 10 frames
                    avg_encoding = np.mean(temp_encodings, axis=0)  # Average encoding
                    known_face_encodings.append(avg_encoding)
                    known_face_names.append(name)
                    face_embeddings.append(avg_encoding)
                    print(f"Face registered for {name}.")
                    break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

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

def calculate_color_features(frame):
    """
    Calculate the color features from the frame, including:
    - Mean color in BGR and HSV
    - Color histograms in HSV space
    """
    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (320, 240))

    # Convert the frame to the HSV color space for color features
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # Calculate mean color in RGB and HSV
    mean_bgr = cv2.mean(resized_frame)[:3]  # BGR mean color
    mean_hsv = cv2.mean(hsv_frame)[:3]  # HSV mean color

    # Calculate color histograms (using 16 bins for each channel in HSV)
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


def calculate_similarity(encoding1, encoding2):
    return cosine_similarity([encoding1], [encoding2])[0][0]

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
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # This line should be added

        # Process face mesh
        results = face_mesh.process(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Reshape face encoding to 2D array before scaling
            face_encoding_scaled = scaler.transform(np.reshape(face_encoding, (1, -1)))

            # Extract critical features using face mesh
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye, right_eye = extract_eye_features(face_landmarks, enhanced_frame)
                    critical_features = extract_critical_features(enhanced_frame, face_landmarks)

                    # You can process these features here (e.g., check if eyes, lips, or nose are covered)
                    print("Critical features extracted:", critical_features)
                    

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
                mp_drawing.draw_landmarks(
                    image=enhanced_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=enhanced_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # Display FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        cv2.putText(enhanced_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', enhanced_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Call register function to register a new face
register_face_from_webcam()

# Call recognize function to recognize faces using webcam
recognize_faces()