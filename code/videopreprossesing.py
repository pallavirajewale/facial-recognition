import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2  # For video processing
from tkinter import Tk, filedialog  # For interactive file upload

# Define the encoder model
def build_encoder():
    model = models.Sequential([
        layers.InputLayer(input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])
    return model

# Define the decoder model
def build_decoder():
    model = models.Sequential([
        layers.InputLayer(input_shape=(256,)),
        layers.Dense(4 * 4 * 128, activation='relu'),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

# Build the full autoencoder model
def build_autoencoder():
    encoder = build_encoder()
    decoder = build_decoder()
    autoencoder = models.Sequential([encoder, decoder])
    return autoencoder

# Compile the autoencoder model
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Function to preprocess a video frame
def preprocess_frame(frame):
    # Resize the frame to 32x32
    frame = cv2.resize(frame, (32, 32))
    # Normalize pixel values to [0, 1]
    frame = frame.astype('float32') / 255.0
    return frame

# Function to upload a video interactively
def upload_video():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    return file_path

# Upload the video
print("Please select a video file:")
video_path = upload_video()
if not video_path:
    print("No file selected.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Prepare to process video frames
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess each frame
    processed_frame = preprocess_frame(frame)
    frames.append(processed_frame)

# Release the video capture
cap.release()
frames = np.array(frames)

# Train the autoencoder model (use dummy data or pre-train with CIFAR-10 if necessary)
from tensorflow.keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
autoencoder.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test))

# Compress and decompress the video frames
encoded_frames = autoencoder.layers[0].predict(frames)
decoded_frames = autoencoder.layers[1].predict(encoded_frames)

# Display the first few original and reconstructed frames
n = 5  # Number of frames to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display original frame
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(frames[i])
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed frame
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_frames[i])
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
