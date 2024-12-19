import cv2
import dlib
import numpy as np

# Load the dlib pre-trained shape predictor model for 68 landmarks
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get the landmarks for each face
        landmarks = predictor(gray, face)

        # Loop through all 68 landmarks and draw them on the frame
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow("Webcam - Facial Landmarks", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
