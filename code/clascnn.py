import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Create a folder for storing images if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Starting image capture... Press 'Esc' to stop.")
face_images = []
face_labels = []
img_counter = 0
label_map = {}
name_to_id = {}
unknown_label = None  # Placeholder for unknown class

# Ask the user how many people they want to add
num_people = int(input("Enter the number of people to capture faces for (include 1 for 'Unknown' class): "))

for i in range(num_people):
    print(f"Capturing faces for person {i + 1} of {num_people}")
    name = input("Enter the name of the person (type 'Unknown' for unknown faces): ")

    # Assign a unique label to the person
    if name not in name_to_id:
        label = len(name_to_id)
        name_to_id[name] = label
        label_map[label] = name
        if name.lower() == "unknown":
            unknown_label = label
    else:
        label = name_to_id[name]

    print(f"Starting to capture images for {name}. Press 'Esc' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(roi_gray, (32, 32))

            face_images.append(face_resized)
            face_labels.append(label)
            img_counter += 1

        cv2.imshow("Capturing Face", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' key to break the loop
            break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {img_counter} images.")

# Convert face_images to the correct shape and normalize
face_images = np.array(face_images)
face_images = face_images.reshape(-1, 32, 32, 1)
face_images = face_images.astype('float32') / 255.0

face_labels = to_categorical(face_labels, num_classes=len(name_to_id))

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(name_to_id), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(face_images, face_labels, epochs=10, batch_size=32)

model.save('face_recognition_model.h5')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Starting face recognition... Press 'Esc' to stop.")

threshold = 0.6  # Set a confidence threshold for "unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(roi_gray, (32, 32))
        face_resized = face_resized.reshape(1, 32, 32, 1)

        prediction = model.predict(face_resized)
        max_confidence = np.max(prediction)
        label = np.argmax(prediction)

        if max_confidence < threshold:  # If confidence is below threshold, classify as "Unknown"
            predicted_name = "Unknown"
        else:
            predicted_name = label_map[label]

        cv2.putText(frame, f"Person: {predicted_name} ({max_confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' key to break
        break

cap.release()
cv2.destroyAllWindows()
