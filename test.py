import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.utils import np_utils

def loadmnist():
    """
    Loads mnist images and one hot representation labels

    Args:
        None

    Returns:
        mnist: Object containing images and labels
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train = mnist.train.images.astype('float32')
    X_test  = mnist.test.images.astype('float32')
    y_train = mnist.train.labels
    y_test  = mnist.test.labels
    X_train=X_train.reshape(X_train.shape[0],28,28,1)
    X_test =X_test.reshape(X_test.shape[0],28,28,1)

    return X_train, X_test, y_train, y_test

def generatenoise(batch_size):
    """
    Generates white noise vector

    Args:
        None

    Returns:
        z: Noise vector with variance 0.01 of shape 100x1
    """
    z=np.random.normal(0,1,(batch_size,100))
    return z
d=generatenoise(3)
def creategenerator():
    """
    Creates model for the generator network

    Args:
        z: Input vector (only its shape is of interest)

    Returns:
        model: Model object containing the information of the network.
        The output has the same shape as an MNIST image

    """
    model=Sequential()
    model.add(Dense(128,activation='relu',input_shape=[100]))
    model.add(Dense(500,activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(784))

    model.add(Reshape([28,28,1]))
   # model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def creatediscriminator():
    """
    Creates model for the discriminator network

    Args:
        Nonw

    Returns:
        model: Model object containing information of the network.
        The output of the network is scalar probability.

    """
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.summary()
    return model

def settrainable(boolean,model):
    t=model
    if boolean==True:
        for i in t.layers:
            i.trainable=True
    else:
        for i in t.layers:
            i.trainable=False
    t.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return t


def train(epochs=10,batch_size=128):
    a,b,c,d=loadmnist()
    half_batch=batch_size//2
    dis=creatediscriminator()
    gen=creategenerator()
    noise=generatenoise(10000)
    noise_samp=gen.predict(noise)



    for i in range(1):

        indx=np.random.randint(0, a.shape[0], 10000)
        real_samp = a[indx]
        #print(real_samp.shape)
        #a1=dis.evaluate(b,d,verbose=2)
        #print(a1)
        dis.fit(a[0:10000],np.ones(10000),batch_size=100,verbose=1)
        dis.fit(noise_samp,np.zeros(10000),batch_size=100,verbose=1)








    #mod.fit(a,b,batch_size=100,epochs=1,verbose=2)
    return dis




a,b,c,d=loadmnist()
dis=creatediscriminator()
gen=creategenerator()
adv=Sequential()
adv.add(gen)
adv.add(dis)
adv.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
noise=generatenoise(10000)
noise_samp=gen.predict(noise)



#print(dis.predict(noise_samp))
c=np.hstack((np.ones([50,1]),np.zeros([50,1])))
c1=np.hstack((np.ones([100,1]),np.zeros([100,1])))
d=np.hstack((np.zeros([50,1]),np.ones([50,1])))
images=gen.predict(generatenoise(1))
loss=10
loss1=10
start=True
for i in range(200):
    print(i)
    indx=np.random.randint(0,50000,50)
    Xtrue=a[indx]
    Xfalse=gen.predict(generatenoise(50))


    #weight=dis.layers[0].get_weights()[0][0][0][0][0]
    dis=settrainable(True,dis)
    #weight=dis.layers[0].get_weights()[0][0][0][0][0]

    loss,acc=dis.evaluate(Xfalse,d,batch_size=50,verbose=0)
    while(float(loss)>0.75 or start==True):
        print('DISKRIMINATOR')
        dis.fit(Xtrue,c,batch_size=50,verbose=0,epochs=1)
        dis.fit(Xfalse,d,batch_size=50,verbose=0,epochs=2)
        loss,acc=dis.evaluate(Xfalse,d,batch_size=50,verbose=0)
        print(loss)
        start=False

    dis=settrainable(False,dis)
    adv=Sequential()
    adv.add(gen)
    adv.add(dis)
    adv.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    loss1,acc1=adv.evaluate(generatenoise(100),c1,batch_size=100,verbose=0)
    while(float(loss1)>0.75):
        print("GENERATOR")
        adv.fit(generatenoise(100),c1,batch_size=100,verbose=0,epochs=1)
        loss1,acc1=adv.evaluate(generatenoise(100),c1,batch_size=100,verbose=0)
        print(loss1)
    #adv.fit(generatenoise(50),c,batch_size=50,verbose=2,epochs=1)
    images=np.vstack((images,gen.predict(generatenoise(1))))

e=gen.predict(noise)




def plotdigit(digitnr):
    """ Plots
    """

#X,_,_,_=loadmnist()
#X=X.reshape(X.shape[0],28,28,1)
#d=np.reshape(a[1,:],(28,28,1))
