import tensorflow as tf 

from tensorflow.keras import layers, models 

import numpy as np 

import matplotlib.pyplot as plt 

from tensorflow.keras.datasets import cifar10 

 

# Load dataset (CIFAR-10 as an example) 

(x_train, _), (x_test, _) = cifar10.load_data() 

 

# Normalize the data 

x_train = x_train.astype('float32') / 255.0 

x_test = x_test.astype('float32') / 255.0 

 

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

 

# Build the full autoencoder model by combining the encoder and decoder 

def build_autoencoder(): 

    encoder = build_encoder() 

    decoder = build_decoder() 

    autoencoder = models.Sequential([encoder, decoder]) 

    return autoencoder 

 

# Compile the autoencoder model 

autoencoder = build_autoencoder() 

autoencoder.compile(optimizer='adam', loss='mse') 

 

# Train the autoencoder model 

autoencoder.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test)) 

 

# Compress and decompress an image 

encoded_imgs = autoencoder.layers[0].predict(x_test) 

decoded_imgs = autoencoder.layers[1].predict(encoded_imgs) 

 

# Display the results 

n = 10  # how many digits we will display 

plt.figure(figsize=(20, 4)) 

for i in range(n): 

    # Display original 

    ax = plt.subplot(2, n, i + 1) 

    plt.imshow(x_test[i]) 

    plt.gray() 

    ax.get_xaxis().set_visible(False) 

    ax.get_yaxis().set_visible(False) 

     

    # Display reconstruction 

    ax = plt.subplot(2, n, i + 1 + n) 

    plt.imshow(decoded_imgs[i]) 

    plt.gray() 

    ax.get_xaxis().set_visible(False) 

    ax.get_yaxis().set_visible(False) 

plt.show()