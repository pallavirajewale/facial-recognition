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

# Select an image file
file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

if file_path:
    # Load the image
    image = cv2.imread(file_path)

    # Detect faces using RetinaFace
    detections = RetinaFace.detect_faces(image)

    if not detections:
        print("No faces detected by RetinaFace.")
    else:
        # Example ground truth for testing purposes (replace with actual ground-truth data for real evaluation)
        ground_truth_boxes = [(50, 50, 200, 200)]  # Replace with actual bounding boxes for the image

        total_iou = 0
        detection_count = 0

        # Process detections for the image
        for key, face_data in detections.items():
            # Get the bounding box for each detected face
            x1, y1, x2, y2 = face_data['facial_area']
            detected_box = [x1, y1, x2, y2]

            # Print detected box coordinates
            print("Detected Box:", detected_box)

            # Calculate IoU with ground-truth boxes
            for gt_box in ground_truth_boxes:
                # Print ground truth box coordinates
                print("Ground Truth Box:", gt_box)

                iou = calculate_iou(detected_box, gt_box)
                print(f"IoU between Detected and Ground Truth Box: {iou:.2f}")
                total_iou += iou
                detection_count += 1

            # Draw the detected bounding box in green
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the ground truth bounding boxes in red
        for gt_box in ground_truth_boxes:
            cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)

        # Calculate and display average IoU for image
        if detection_count > 0:
            average_iou = total_iou / detection_count
            print(f"Average IoU (Accuracy) for Image: {average_iou:.2f}")
        else:
            print("No detections made.")

        # Display the image with both detected and ground-truth bounding boxes
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("No file selected.")
