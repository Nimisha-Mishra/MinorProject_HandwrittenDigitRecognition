import tensorflow as tf
from tensorflow.keras import datasets, layers, models

print("1. Downloading dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print("2. Preprocessing data...")
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

print("3. Building the Neural Network...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("4. Training the model (this will take a minute)...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

print("5. Saving the model's brain...")
# THIS IS THE CRUCIAL LINE
model.save('mnist_digit_model.h5')

print("SUCCESS: mnist_digit_model.h5 has been created!")