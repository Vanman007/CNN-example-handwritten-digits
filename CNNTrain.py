import dataset as MNIST
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

model.fit({'input': X_train}, {'target': y_train}, n_epoch=20,
           validation_set=({'input': X_test}, {'target': y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

# Manually save model
model.save("model.tfl")


