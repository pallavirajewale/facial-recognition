import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.datasets import cifar10 

from tensorflow.keras.utils import to_categorical 

import matplotlib.pyplot as plt 

 

# Load CIFAR-10 dataset 

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

 

# Normalize the data 

x_train = x_train.astype('float32') / 255.0 

x_test = x_test.astype('float32') / 255.0 

 

# One-hot encode the labels 

y_train = to_categorical(y_train, 10) 

y_test = to_categorical(y_test, 10) 

 

# Define the CNN model 

model = models.Sequential([ 

    # Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation 

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), 

    layers.MaxPooling2D((2, 2)),  # MaxPooling layer with pool size 2x2 

    layers.Conv2D(64, (3, 3), activation='relu'), 

    layers.MaxPooling2D((2, 2)), 

    layers.Conv2D(64, (3, 3), activation='relu'), 

     

    layers.Flatten(),  # Flatten the feature map to a 1D vector 

    layers.Dense(64, activation='relu'),  # Fully connected layer 

    layers.Dense(10, activation='softmax')  # Output layer with 10 classes 

]) 

 

# Compile the model 

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy']) 

 

# Train the model 

history = model.fit(x_train, y_train, epochs=10, batch_size=64, 

                    validation_data=(x_test, y_test)) 

 

# Evaluate the model 

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) 

print(f"Test accuracy: {test_acc}") 

 

# Plot training and validation accuracy/loss 

plt.figure(figsize=(12, 4)) 

plt.subplot(1, 2, 1) 

plt.plot(history.history['accuracy'], label='Training Accuracy') 

plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 

plt.legend() 

plt.title('Accuracy') 

 

plt.subplot(1, 2, 2) 

plt.plot(history.history['loss'], label='Training Loss') 

plt.plot(history.history['val_loss'], label='Validation Loss') 

plt.legend() 

plt.title('Loss') 

 

plt.show() 