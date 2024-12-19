import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import mediapipe as mp
import pickle
 
# Paths for saving data and models
DATASET_PATH = "dataset"
MODEL_PATH = "face_recognition_cnn_model.h5"
 
# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
# Create dataset directories if not exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
 
# Helper function to collect face data
def collect_face_data():
    video_capture = cv2.VideoCapture(0)
    name = input("Enter the name of the person to register: ")
    person_dir = os.path.join(DATASET_PATH, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
 
    print("Press 's' to save the frame, or 'q' to quit.")
    count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
 
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
        cv2.imshow('Collect Face Data', frame)
 
        if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
            face_image = frame[y:y+h, x:x+w]
            face_image_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(face_image_path, face_image)
            print(f"Saved: {face_image_path}")
            count += 1
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    video_capture.release()
    cv2.destroyAllWindows()
 
# CNN model definition
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
 
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
 
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# Function to train CNN model
def train_cnn_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
 
    input_shape = (64, 64, 3)
    num_classes = len(train_generator.class_indices)
 
    model = build_cnn_model(input_shape, num_classes)
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
 
    model.save(MODEL_PATH)
    print("Model training completed and saved.")
 
# Load the trained CNN model
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)
 
# Real-time face recognition
def recognize_faces():
    model = load_cnn_model()
    class_indices = {v: k for k, v in ImageDataGenerator().flow_from_directory(DATASET_PATH).class_indices.items()}
 
    video_capture = cv2.VideoCapture(0)
 
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)
 
        results = face_mesh.process(rgb_frame)
 
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_image_resized = cv2.resize(face_image, (64, 64))
            face_image_normalized = face_image_resized / 255.0
            face_image_expanded = np.expand_dims(face_image_normalized, axis=0)
 
            predictions = model.predict(face_image_expanded)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
 
            name = class_indices[predicted_class] if confidence > 0.8 else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
 
        cv2.imshow('Face Recognition with CNN', frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    video_capture.release()
    cv2.destroyAllWindows()
 
# Main program
if __name__ == "__main__":
    while True:
        print("1. Collect face data")
        print("2. Train CNN model")
        print("3. Recognize faces")
        print("4. Exit")
        choice = input("Enter your choice: ")
 
        if choice == '1':
            collect_face_data()
        elif choice == '2':
            train_cnn_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

     