# Image Classification with TensorFlow/Keras

## Introduction
This project demonstrates image classification using TensorFlow and Keras. It employs a Convolutional Neural Network (CNN) to classify flower images into five categories: roses, daisy, tulips, dandelion, and sunflowers. The model is trained and evaluated on a dataset of flower images with data augmentation to improve generalization.

## Dependencies
- TensorFlow (>=2.0)
- NumPy
- OpenCV
- Matplotlib
- PIL (Pillow)
- scikit-learn

## Installation
To install the required dependencies, run the following command:
```bash
pip install tensorflow numpy opencv-python matplotlib pillow scikit-learn
```
## Data Preparation
The dataset is downloaded from TensorFlow's storage and consists of images of flowers. It is then organized and prepared for training and testing.
```bash
import tensorflow as tf
import pathlib
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Download and extract dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)

# Create a dictionary for images and labels
flower_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'tulips': list(data_dir.glob('tulips/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*'))
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4
}

# Load and preprocess images
X, y = [], []
for flower_name, images in flower_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
```
## Model Architecture
The CNN model is defined with the following layers:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with ReLU activation
- Output layer with softmax activation
```bash
from tensorflow.keras import layers, Sequential

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
## Training
The model is trained on the scaled training data for 10 epochs.
```bash
model.fit(X_train_scaled, y_train, epochs=10)
```
## Evaluation
The model's performance is evaluated on the test set.
```bash
model.evaluate(X_test_scaled, y_test)
```
## Predictions
The model makes predictions on the test set and demonstrates how data augmentation can improve the model's robustness.
```bash
import matplotlib.pyplot as plt

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(180, 180, 3)),
    layers.experimental.preprocessing.RandomZoom(0.3),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

# Visualize augmented images
plt.axis('off')
plt.imshow(X[0])
plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype('uint8'))
```
## Further Exploration
- Experiment with different data augmentation techniques to improve model performance.
- Adjust hyperparameters and model architecture for better results.



 
