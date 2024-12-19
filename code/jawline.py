import cv2
import numpy as np
from mtcnn import MTCNN
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize Tkinter to open file dialog
Tk().withdraw()  # Hide the root window
image_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Check if a file was selected
if not image_path:
    print("No file selected. Exiting program.")
else:
    # Initialize the MTCNN model
    detector = MTCNN()

    # Function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Load the selected image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces and landmarks
        results = detector.detect_faces(image_rgb)

        # Process each face detected
        for result in results:
            # Extract facial landmarks
            keypoints = result['keypoints']
            left_eye = np.array(keypoints['left_eye'])
            right_eye = np.array(keypoints['right_eye'])
            nose = np.array(keypoints['nose'])
            mouth_left = np.array(keypoints['mouth_left'])
            mouth_right = np.array(keypoints['mouth_right'])

            # Calculate distances (e.g., interocular distance and mouth width)
            eye_distance = euclidean_distance(left_eye, right_eye)
            mouth_width = euclidean_distance(mouth_left, mouth_right)
            print(f"Interocular Distance: {eye_distance}")
            print(f"Mouth Width: {mouth_width}")

            # Jawline feature (estimated using the distance from the mouth corners to the chin)
            chin_point = (keypoints['mouth_left'][0], keypoints['mouth_left'][1] + 60)  # Estimate chin position
            jaw_left = euclidean_distance(np.array(chin_point), mouth_left)
            jaw_right = euclidean_distance(np.array(chin_point), mouth_right)
            jaw_width = jaw_left + jaw_right
            print(f"Jaw Width (approximate): {jaw_width}")

            # Cheekbone contour (estimated using eye to mouth corner distances)
            cheek_left = euclidean_distance(left_eye, mouth_left)
            cheek_right = euclidean_distance(right_eye, mouth_right)
            cheekbone_contour = cheek_left + cheek_right
            print(f"Cheekbone Contour (approximate): {cheekbone_contour}")

            # Draw landmarks and lines for visualization
            for key, point in keypoints.items():
                cv2.circle(image, tuple(point), 2, (255, 0, 0), 5)
            
            # Draw lines between key points
            cv2.line(image, tuple(left_eye), tuple(right_eye), (0, 255, 0), 2)
            cv2.line(image, tuple(mouth_left), tuple(mouth_right), (0, 255, 0), 2)
            cv2.line(image, mouth_left, chin_point, (255, 0, 255), 2)
            cv2.line(image, mouth_right, chin_point, (255, 0, 255), 2)

        # Show the image with landmarks
        cv2.imshow('Facial Features', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
