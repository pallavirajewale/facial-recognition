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

# Function to calculate precision, recall, and F1-score
def calculate_metrics(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

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
    detection_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Example: Ground truth (replace with actual data if available)
    ground_truth_boxes = [(50, 50, 200, 200)]  # Replace with actual ground-truth bounding boxes

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces using RetinaFace
        detections = RetinaFace.detect_faces(frame)

        # Debug: Print detections to verify if faces are being detected
        print(f"Detections: {detections}")

        # Apply Non-Maximum Suppression (NMS) to remove redundant detections
        boxes = []
        confidences = []

        for key, face_data in detections.items():
            x1, y1, x2, y2 = face_data['facial_area']
            
            # Check if 'confidence' exists in the detected face data
            confidence = face_data.get('confidence', 1.0)  # Default confidence to 1.0 if not present
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to [x, y, width, height]
            confidences.append(confidence)

        # Use Non-Maximum Suppression (NMS) to remove redundant detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.3)  # Lowered thresholds

        # Debug: Print NMS indices
        print(f"Indices after NMS: {indices}")

        # Make sure indices is not empty
        if len(indices) > 0:
            # Track matched faces
            detected_face_boxes = []
            
            # Iterate over the indices returned by NMS
            for i in indices.flatten():  # Flatten the result to access it
                detected_box = boxes[i]
                x1, y1, w, h = detected_box
                detected_box = [x1, y1, x1 + w, y1 + h]  # Convert to [x1, y1, x2, y2]

                matched = False
                for gt_box in ground_truth_boxes:
                    iou = calculate_iou(detected_box, gt_box)
                    if iou >= 0.5:  # IoU threshold
                        total_iou += iou
                        detection_count += 1
                        true_positive += 1
                        matched = True
                        detected_face_boxes.append(detected_box)
                        break

                if not matched:
                    false_positive += 1

            # Count false negatives (ground truth boxes not detected)
            false_negative += len(ground_truth_boxes) - len(detected_face_boxes)

        else:
            print("No detections after NMS")

        # Optional: Display the frame with detected bounding boxes
        for idx in indices.flatten() if len(indices) > 0 else []:
            x1, y1, w, h = boxes[idx]
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

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

    # Calculate additional metrics: Precision, Recall, F1-score
    precision, recall, f1_score = calculate_metrics(true_positive, false_positive, false_negative)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}")

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
else:
    print("No file selected.")



