import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization, UpSampling2D
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers
from keras.optimizers import SGD
from keras.models import load_model

def loadmodel(dataset):
    if dataset == "mnist":
        g,d=mnistmodel()
    elif dataset == "cifar":
        g,d=cifarmodel()
    elif dataset == "flowers":
        g,d= flowermodel()
    else:
        g=load_model(dataset)
        d=0
    return g,d


def mnistmodel():
    g = Sequential()
    g.add(Dense(128*8*8,input_shape=[10],kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    g.add(LeakyReLU(0.2))
    g.add(Reshape([8,8,128]))
    g.add(UpSampling2D((2,2)))
    g.add(Conv2D(64,(5,5),strides=(1,1),padding='same'))
    g.add(LeakyReLU(0.2))
    g.add(UpSampling2D((2,2)))
    g.add(Conv2D(1,(5,5),strides=(1,1),activation='tanh',padding='valid'))
    g.summary()



    d = Sequential()
    d.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28,1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Flatten())
    d.add(Dense(1, activation='sigmoid'))
    d.summary()
    return g,d

def cifarmodel():
    g=Sequential()
    g.add(Dense(128*8*8,input_shape=[10],kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    g.add(LeakyReLU(0.2))
    g.add(Reshape([8,8,128]))
    g.add(UpSampling2D((2,2)))
    g.add(Conv2D(64,(5,5),strides=(1,1),padding='same'))
    g.add(LeakyReLU(0.2))
    g.add(UpSampling2D((2,2)))
    g.add(Conv2D(3,(5,5),strides=(1,1),activation='tanh',padding='same'))
    g.summary()



    d = Sequential()
    d.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(32, 32,3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Flatten())
    d.add(Dense(1, activation='sigmoid'))
    return g,d
