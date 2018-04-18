import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train=mnist.train.images[0:100,:]
y_train=mnist.train.labels[0:100,:]

X_test=mnist.test.images
y_test=mnist.test.labels

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_test/=255
X_train/=255

X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
print(X_train.shape)

model=Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          batch_size=10, nb_epoch=10, verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)


Generator=Sequential()
Discriminator=Sequential()

Discriminator.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
Discriminator.add(Convolution2D(32, 3, 3, activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2,2)))
Discriminator.add(Dropout(0.25))
 
Discriminator.add(Flatten())
Discriminator.add(Dense(128, activation='relu'))
#Discriminator.add(Dropout(0.5))
Discriminator.add(Dense(1, activation='softmax'))

model





