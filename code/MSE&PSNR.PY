import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog  # For interactive file upload
import math

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

# Function to preprocess an input image
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)
    # Resize the image to 32x32
    img = img.resize((32, 32))
    # Normalize pixel values to [0, 1]
    img = np.array(img).astype('float32') / 255.0
    # Ensure the image has the correct shape (32, 32, 3)
    if img.shape[-1] != 3:
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB if needed
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to upload an image interactively
def upload_image():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

# Upload the image and preprocess it
print("Please select an image file:")
image_path = upload_image()
if image_path:
    input_image = preprocess_image(image_path)
else:
    print("No file selected.")
    exit()

# Train the autoencoder model (use dummy data or pre-train with CIFAR-10 if necessary)
from tensorflow.keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
autoencoder.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test))

# Compress and decompress the input image
encoded_img = autoencoder.layers[0].predict(input_image)
decoded_img = autoencoder.layers[1].predict(encoded_img)

# Calculate MSE
mse = np.mean((input_image - decoded_img) ** 2)

# Calculate PSNR
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))

# Display the original and reconstructed image
plt.figure(figsize=(8, 4))
# Display original image
ax = plt.subplot(1, 2, 1)
plt.imshow(input_image[0])
plt.title("Original Image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Display reconstructed image
ax = plt.subplot(1, 2, 2)
plt.imshow(decoded_img[0])
plt.title("Reconstructed Image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

# Print MSE and PSNR
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
