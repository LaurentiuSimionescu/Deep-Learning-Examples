# CNN - Convolutional Neural Network = It is an ANN with convolutional layers, great for computer vision

# pip install theano
# pip install tensorflow
# pip install keras
# pip install pillow

# Part 1 - Building the CNN

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialising the CNN
# Steps: Convolution -> Max Pooling -> Flattening -> Full Connection

classifier = Sequential()

# Convolution
# apply a 3 by 3 convolution with 32 output filters on a 64 by 64 image with 3 channels (RGB)
classifier.add(
    Conv2D(activation="relu", input_shape=(64, 64, 3), padding="same", filters=32, kernel_size=(3, 3)))

# Max Pooling
# Using 2 by 2 - information is preserved, reducing the size without reducing the performance
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
#
classifier.add(Flatten())

# Build ANN
# Start by adding the hidden layer (fully connected layer)
classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting CNN to the images
# image augmentation - prevents over fitting (rotates, shrinks, scales, horizontal flip, shear, etc. the image)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),  # size expected by the CNN
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),  # size expected by the CNN
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,  # number of images in the training set
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)  # number of images in the test set
