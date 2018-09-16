import numpy as np
import os
from random import shuffle
from matplotlib import pyplot as plt
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import cv2
import math

def translate(value, leftMin, leftMax, rightMin, rightMax):
	#https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def reversenumber(x,max):
	return abs(x-max)

im = cv2.imread("TestImg.tif")
newIm=np.zeros(784).reshape(-1,28,28,1)


for x in range(0,28):
	for y in range(0,28):
		newIm[0][x][y][0]=translate(reversenumber(im[x][y][0],255),0,255,0,1)
		

MODEL_NAME='Handwritten-Didgits'
train_size=1000
test_size=200
IMG_Size=28
LR = 0.0001

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X_train=X
y_train=Y
X_test=testX
y_test=testY


X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])
	
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh',name='dense1')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh',name='dense2')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)

# Load a model
model.load("model.tfl")

os.system('cls')

lepredict=newIm.reshape([1,28,28,1])
predictions = model.predict(lepredict)
print(np.argmax(predictions))


	
