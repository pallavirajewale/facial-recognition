import cv2
import numpy as np
from mtcnn import MTCNN
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

# Initialize the MTCNN face detector
detector = MTCNN()

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Open the webcam
cap = cv2.VideoCapture(0)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and landmarks
    results = detector.detect_faces(frame_rgb)

    # Process each face detected
    for result in results:
        keypoints = result['keypoints']
        left_eye = np.array(keypoints['left_eye'])
        right_eye = np.array(keypoints['right_eye'])
        nose = np.array(keypoints['nose'])
        mouth_left = np.array(keypoints['mouth_left'])
        mouth_right = np.array(keypoints['mouth_right'])

        # Calculate distances (e.g., interocular distance)
        eye_distance = euclidean_distance(left_eye, right_eye)
        mouth_width = euclidean_distance(mouth_left, mouth_right)
        
        print(f"Interocular Distance: {eye_distance}")
        print(f"Mouth Width: {mouth_width}")
        
        # Draw landmarks and lines for visualization
        for key, point in keypoints.items():
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), 5)
        
        cv2.line(frame, tuple(left_eye), tuple(right_eye), (0, 255, 0), 2)
        cv2.line(frame, tuple(mouth_left), tuple(mouth_right), (0, 255, 0), 2)

    # Convert to grayscale for GLCM analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert to 8-bit unsigned byte format
    gray_frame_ubyte = img_as_ubyte(gray_frame)
    
    # Calculate GLCM matrix
    glcm = greycomatrix(gray_frame_ubyte, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = greycoprops(glcm, 'contrast')
    correlation = greycoprops(glcm, 'correlation')
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')

    # Print GLCM features for the frame
    print(f"Frame {frame_count}:")
    print("Contrast:", contrast.mean())
    print("Correlation:", correlation.mean())
    print("Energy:", energy.mean())
    print("Homogeneity:", homogeneity.mean())

    # Display the frame with GLCM values as overlay text (optional)
    cv2.putText(frame, f"Contrast: {contrast.mean():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f"Correlation: {correlation.mean():.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f"Energy: {energy.mean():.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f"Homogeneity: {homogeneity.mean():.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the frame with facial landmarks and GLCM features
    cv2.imshow('Facial Features and GLCM on Webcam Frame', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
