import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import mediapipe as mp
 
# Initialize MTCNN face detector and DeepFace model (FaceNet by default)
detector = MTCNN()
model_name = 'Facenet'
 
# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
 
# Initialize drawing utilities for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
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
 
def calculate_glcm_features(gray_frame, distances=[1], angles=[0]):
    # Initialize GLCM matrix (256x256 for grayscale images)
 
    glcm = np.zeros((256, 256), dtype=np.float32)
    rows, cols = gray_frame.shape
   
    # Calculate GLCM for given distances and angles
    for dist in distances:
        for angle in angles:
            for i in range(rows - dist):
                for j in range(cols - dist):
                    pixel_val = gray_frame[i, j]
                    if angle == 0:
                        neighbor_val = gray_frame[i + dist, j]
                    elif angle == np.pi / 2:
                        neighbor_val = gray_frame[i, j + dist]
                    else:
                        continue  # Only handle horizontal and vertical directions
 
                    glcm[pixel_val, neighbor_val] += 1
   
    # Normalize GLCM matrix
    glcm /= np.sum(glcm)
 
    # Calculate texture properties from GLCM
    contrast = 0
    dissimilarity = 0
    homogeneity = 0
    energy = 0
    correlation = 0
   
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]
            dissimilarity += abs(i - j) * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
            energy += glcm[i, j] ** 2
            correlation += (i - np.mean(gray_frame)) * (j - np.mean(gray_frame)) * glcm[i, j]
 
    return contrast, dissimilarity, homogeneity, energy, correlation
 
# Open video capture (0 for default camera)
cap = cv2.VideoCapture(0)
 
# Initialize counters for accuracy calculation
total_frames = 0
successful_detections = 0
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Calculate color features
    mean_bgr, mean_hsv, color_hist_features = calculate_color_features(frame)
 
    # Convert frame to grayscale for GLCM calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast, dissimilarity, homogeneity, energy, correlation = calculate_glcm_features(gray_frame)
 
    # Display mean color and GLCM information on the frame
    color_info = f"Mean BGR: {mean_bgr}, Mean HSV: {mean_hsv}"
    glcm_info = f"GLCM -> Contrast: {contrast:.2f}, Dissimilarity: {dissimilarity:.2f}, Homogeneity: {homogeneity:.2f}"
    cv2.putText(frame, color_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, glcm_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
    # Convert frame to RGB for MediaPipe face mesh and MTCNN face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Detect faces using MTCNN
    results = detector.detect_faces(rgb_frame)
 
    for result in results:
        total_frames += 1
 
        # Get bounding box and keypoints
        x, y, width, height = result['box']
        keypoints = result['keypoints']
 
        # Extract face region
        face_region = frame[y:y + height, x:x + width]
 
        try:
            # Face recognition with DeepFace
            face_embedding = DeepFace.represent(face_region, model_name=model_name, enforce_detection=False)[0]['embedding']
            print(f"Face Embedding: {face_embedding[:5]}...")
 
            # Example condition for successful detection (adjust threshold as needed)
            if np.linalg.norm(face_embedding) > 1.0:
                successful_detections += 1
        except Exception as e:
            print(f"Error in face recognition: {e}")
 
        # Draw bounding box and landmarks
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        for key, point in keypoints.items():
            cv2.circle(frame, point, 2, (255, 0, 0), 5)
 
    # MediaPipe Face Mesh processing
    face_mesh_results = face_mesh.process(rgb_frame)
 
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # Draw face landmarks on the frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
 
    # Calculate accuracy rate
    accuracy_rate = (successful_detections / total_frames) * 100 if total_frames > 0 else 0.0
 
    # Display the accuracy rate
    cv2.putText(frame, f"Accuracy Rate: {accuracy_rate:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    # Show the frame with overlays
    cv2.imshow('Face and Recognition with Accuracy Rate', frame)
 
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()