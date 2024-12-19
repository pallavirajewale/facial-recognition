import cv2
from deepface import DeepFace
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Set up the Tkinter window and hide it
root = Tk()
root.withdraw()  # Hide the Tkinter root window

# Open a file dialog to let the user select a video file
video_path = askopenfilename(title="Select a Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

if not video_path:
    print("No video selected. Exiting...")
    exit()

# Initialize video capture with the selected video file
cap = cv2.VideoCapture(video_path)

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize variables for accuracy calculation
recognized_faces = {}  # Dictionary to store recognized face counts
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using OpenCV Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected in this frame.")
            continue  # Skip this frame if no faces are detected

        # Optionally resize the frame for better face detection (e.g., for smaller faces)
        frame_resized = cv2.resize(frame, (640, 480))

        # Analyze each detected face using DeepFace with enforce_detection=False for flexibility
        analysis = DeepFace.analyze(frame_resized, actions=["emotion", "age", "gender"], enforce_detection=False, model_name="VGG-Face")

        # Process each face found in the analysis
        for face_analysis in analysis:
            # Use "dominant_emotion" as a placeholder identifier for face recognition
            face_identity = face_analysis.get("dominant_emotion", "Unknown")
            recognized_faces[face_identity] = recognized_faces.get(face_identity, 0) + 1

        # Increment the frame count
        total_frames += 1

        # Display results
        for i, (face, count) in enumerate(recognized_faces.items()):
            label = f"{face}: {count} times"
            cv2.putText(frame_resized, label, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Facial Recognition", frame_resized)

    except ValueError as e:
        print(f"Error processing frame: {e}")

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate some basic statistics
most_frequent_face = max(recognized_faces, key=recognized_faces.get, default="No faces detected")
print(f"Most recognized face: {most_frequent_face}")
print(f"Recognition counts: {recognized_faces}")

# Release resources
cap.release()
cv2.destroyAllWindows()
