import cv2
from retinaface import RetinaFace
import tkinter as tk
from tkinter import filedialog

# Function to calculate Intersection over Union (IoU) for accuracy
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0
    
    # Compute the areas of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Initialize tkinter for file dialog
root = tk.Tk()
root.withdraw()

# Select a video file
file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])

if file_path:
    # Initialize video capture
    video_capture = cv2.VideoCapture(file_path)

    # Variables for calculating average IoU (accuracy)
    total_iou = 0
    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces using RetinaFace
        detections = RetinaFace.detect_faces(frame)

        # Example ground truth for testing purposes (replace with actual ground-truth data for real evaluation)
        ground_truth_boxes = [(50, 50, 200, 200)]  # Replace with actual bounding boxes per frame

        for key, face_data in detections.items():
            # Get the bounding box for each detected face
            x1, y1, x2, y2 = face_data['facial_area']
            detected_box = [x1, y1, x2, y2]

            # Calculate IoU with ground-truth boxes
            for gt_box in ground_truth_boxes:
                iou = calculate_iou(detected_box, gt_box)
                total_iou += iou
                detection_count += 1

        frame_count += 1

        # Optional: Display the frame with detected bounding boxes
        for key, face_data in detections.items():
            x1, y1, x2, y2 = face_data['facial_area']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate average IoU as an accuracy metric
    if detection_count > 0:
        average_iou = total_iou / detection_count
        print(f"Average IoU (Accuracy): {average_iou:.2f}")
    else:
        print("No detections made.")

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
else:
    print("No file selected.")
