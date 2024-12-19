import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance
 
# Load pre-trained CNN model for face embeddings (e.g., FaceNet)
model = load_model('path_to_your_pretrained_facenet_model.h5')  # Load your pre-trained CNN face model
 
# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Load the dlib face landmark model
 
def get_face_embedding(face_img):
    """Get embedding vector for a detected face using the CNN model."""
    face_img = cv2.resize(face_img, (160, 160))  # Resize to the model input size
    face_img = face_img.astype('float32') / 255.0  # Normalize pixel values
    face_img = np.expand_dims(face_img, axis=0)
    embedding = model.predict(face_img)[0]  # Get embedding
    return embedding
 
def extract_landmarks(face, gray_frame):
    """Get facial landmarks for the face outline."""
    landmarks = predictor(gray_frame, face)
    outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)])  # Jawline points (face outline)
    return outline
 
def optical_flow_tracking(previous_gray, current_gray, points):
    """Track movement using optical flow."""
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **lk_params)
    motion_vectors = new_points - points
    return new_points, motion_vectors
 
# Load video and initialize variables
cap = cv2.VideoCapture('video.mp4')
previous_gray = None
face_embeddings = []
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detect faces
    faces = detector(gray_frame)
   
    for face in faces:
        # Extract the face ROI and get embedding
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face_img)
       
        # Store embedding for later temporal analysis
        face_embeddings.append(embedding)
       
        # Extract landmarks for outline
        outline = extract_landmarks(face, gray_frame)
       
        # Optical flow for motion tracking (initialize if first frame)
        if previous_gray is not None:
            outline_points = np.float32(outline).reshape(-1, 1, 2)
            new_outline, motion_vectors = optical_flow_tracking(previous_gray, gray_frame, outline_points)
            head_movement_vector = np.mean(motion_vectors, axis=0)  # Average motion direction for head movement
 
            # Draw outline and motion vectors
            for (x1, y1), (x2, y2) in zip(outline, new_outline.reshape(-1, 2)):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)  # Draw motion line
           
            # Indicate head movement
            cv2.arrowedLine(frame, (int(x+w/2), int(y+h/2)),
                            (int(x+w/2 + head_movement_vector[0]), int(y+h/2 + head_movement_vector[1])),
                            (255, 0, 0), 2)
 
        # Update the previous frame and outline points
        previous_gray = gray_frame.copy()
 
    # Display the frame with face outline, motion, and head movement tracking
    cv2.imshow("Face Recognition and Temporal Features", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()