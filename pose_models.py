import pandas as pd 
import numpy as np 
import scipy.io as io 
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

# Load in the data 
pose_data = io.loadmat('pose.mat')['pose']

# Reformat the data to a cleaner format 
full_data = []
for i in range(pose_data.shape[3]):
    person_imgs = pose_data[:,:,:,i] 
    person_list = [person_imgs[:,:,img] for img in range(13)]
    full_data.append(person_list)

full_data = np.array(full_data)
labels = np.array([[img for i in range(13)] for img in range(68)])
labels = labels.reshape(884,)
print(full_data.shape)

test_data, test_labels = [], []
train_data, train_labels = [], []
for per in range(len(full_data)):
    test_data.append(full_data[per][-3:])
    test_labels.append([per for i in range(3)])
    train_data.append(full_data[per][:10])
    train_labels.append([per for i in range(10)])

test_data, test_labels = np.array(test_data).reshape(204, 48,40,1), \
    to_categorical(np.array(test_labels).reshape(204,))
train_data, train_labels = np.array(train_data).reshape(680, 48,40,1), \
    to_categorical(np.array(train_labels).reshape(680,))

# LeNet-5 Build attempt 
le_net = Sequential([
    layers.Conv2D(6, kernel_size=(3,3), activation='relu', \
        input_shape=(48,40,1)), 
    layers.AveragePooling2D(), 
    layers.Conv2D(16, kernel_size=(3,3), activation='relu'), 
    layers.AveragePooling2D(),
    layers.Flatten(), 
    layers.Dense(120, activation='relu'), 
    layers.Dense(84, activation='relu'), 
    layers.Dense(68, activation='softmax')
])


# AlexNet Attempt 
alex_net_model = Sequential([
    layers.Conv2D(filters=96, kernel_size=(5,5), \
        activation='relu', strides=(2,2), input_shape=(48,40, 1)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3), strides=(1,1)),
    layers.Conv2D(filters=256, kernel_size=(3,3), \
        activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3)),
    layers.Conv2D(filters=384, kernel_size=(3,3), \
        activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=384, kernel_size=(3,3), \
        activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=(1,1), \
        activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3,3)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(68, activation='softmax')
])

# Custom method taken from Previous code used for this task 
cust_model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', \
        input_shape=(48, 40, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(68, activation='softmax')
])

# AlexNet Attempt 
# model = Sequential([
#     layers.Conv2D(filters=96, kernel_size=(5,5), \
#         activation='relu', strides=(2,2), input_shape=(48,40, 1)),
#     # layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3), strides=(1,1)),
#     layers.Conv2D(filters=256, kernel_size=(3,3), \
#         activation='relu', padding="same"),
#     # layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3)),
#     layers.Conv2D(filters=384, kernel_size=(3,3), \
#         activation='relu', padding="same"),
#     # layers.BatchNormalization(),
#     layers.Conv2D(filters=384, kernel_size=(3,3), \
#         activation='relu', padding="same"),
#     # layers.BatchNormalization(),
#     layers.Conv2D(filters=256, kernel_size=(1,1), \
#         activation='relu', padding="same"),
#     # layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3)),
#     layers.Flatten(),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(68, activation='softmax')
# ])

# model = Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', \
#     input_shape=(48, 40, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# #model.summary()
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(68, activation='softmax'))

rmsprop = optimizers.RMSprop(lr=0.001)
sgd = optimizers.SGD(learning_rate=0.0001)
adam = optimizers.Adam(learning_rate=0.0001)

alex_net_model.compile(optimizer=adam, 
            loss=losses.CategoricalCrossentropy(), 
             metrics=['acc'])
hist = alex_net_model.fit(train_data, train_labels, \
    epochs=60, batch_size=32)
evals = alex_net_model.evaluate(x=test_data, y=test_labels)
print(train_data.shape)
# le_net.compile(optimizer=sgd, 
#             loss=losses.CategoricalCrossentropy(), 
#              metrics=['acc'])
# hist = le_net.fit(train_data, train_labels, \
#     epochs=60, batch_size=32)
# evals = le_net.evaluate(x=test_data, y=test_labels)
print(evals)
# print(hist.history['acc'])
plt.plot(hist.history['acc'])
plt.title('AlexNet Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# # summarize history for loss
# plt.plot(hist.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()




