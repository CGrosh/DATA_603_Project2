import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to view an inputted image 
def viewImage(x):
    plt.figure(figsize=(2,2))
    plt.imshow(x, interpolation='nearest', cmap='gray')
    plt.show()

# Read in the MNIST data from tensorflow/keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the train images 
train_images = train_images.reshape((60000, 28, 28, 1))
# Normalize the images, this is possibly a reason why Batch Normalization did show to be effective 
train_images = train_images.astype('float32') / 255

# Reshape and normalize the test images 
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Prepare the Data Labels for the model 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# LeNet-5 Build attempt 
le_net = Sequential([
    layers.Conv2D(6, kernel_size=(3,3), activation='relu', \
        input_shape=(28,28,1)), 
    layers.AveragePooling2D(), 
    layers.Conv2D(16, kernel_size=(3,3), activation='relu'), 
    layers.AveragePooling2D(),
    layers.Flatten(), 
    layers.Dense(120, activation='relu'), 
    layers.Dense(84, activation='relu'), 
    layers.Dense(10, activation='softmax')
])


# AlexNet Attempt 
alex_net_model = Sequential([
    layers.Conv2D(filters=96, kernel_size=(5,5), \
        activation='relu', strides=(2,2), input_shape=(28,28, 1)),
    # layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(1,1)),
    layers.Conv2D(filters=256, kernel_size=(3,3), \
        activation='relu', padding="same"),
    # layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3)),
    layers.Conv2D(filters=384, kernel_size=(3,3), \
        activation='relu', padding="same"),
    # layers.BatchNormalization(),
    layers.Conv2D(filters=384, kernel_size=(3,3), \
        activation='relu', padding="same"),
    # layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(1,1), \
        activation='relu', padding="same"),
    # layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Custom method taken from Previous code that I had used for 
# this task in an Undergraduate Class, very similar to Le-Net 5  
cust_model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', \
        input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Same feed forward model tested
feed_forward = Sequential([
    layers.Flatten(input_shape=(28,28,1)), 
    layers.Dense(32, activation='relu'), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(10, activation='softmax')
])

# Optimizers that were tested for the models 
rmsprop = optimizers.RMSprop(lr=0.0001)
sgd = optimizers.SGD(learning_rate=0.0001)
adam = optimizers.Adam(learning_rate=0.0001)

# Compile functions for the 3 methods tested 
cust_model.compile(optimizer=rmsprop, 
            loss=losses.CategoricalCrossentropy(), 
             metrics=['acc'])
alex_net_model.compile(optimizer=rmsprop, 
            loss=losses.CategoricalCrossentropy(), 
             metrics=['acc'])
le_net.compile(optimizer=rmsprop, 
            loss=losses.CategoricalCrossentropy(), 
             metrics=['acc'])
feed_forward.compile(optimizer=adam, 
            loss=losses.CategoricalCrossentropy(), 
             metrics=['acc'])

# Fit the differnt models that were tested 
le_net.fit(train_images, train_labels, epochs=20,  \
    batch_size=32)
alex_net_model.fit(train_images, train_labels, epochs=20, steps_per_epoch=100, \
    batch_size=32)

feed_forward.fit(train_images, train_labels, epochs=20, steps_per_epoch=100, \
    batch_size=32)

# Evaluate on the testing data with the differnt models
evals = feed_forward.evaluate(x=test_images, y=test_labels)
# # evals = le_net.evaluate(x=test_images, y=test_labels)
# evals = alex_net_model.evaluate(x=test_images, y=test_labels)
print(evals)