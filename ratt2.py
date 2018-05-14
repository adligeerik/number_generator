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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


np.random.seed(42)
optadam=optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
init=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)



def getdata(digit=None):
    """Hämta bilderna"""
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if digit==None:
        images=mnist.train.images.astype('float32')
    else:
        images=mnist.train.images.astype('float32')
        labels=mnist.train.labels
        col=labels[:,digit]
        col=col!=0
        images=images[col]
    images=images.reshape(images.shape[0],28,28,1)
    #images=np.pad(images,((0,0),(2,2),(2,2),(0,0)),'constant')
    images=2*(images-0.5)

    return images





def getnoise(size):
    """Generera brus till generatorn"""
    noisesize=10
    #noise=np.random.normal(0,1,(size,noisesize))
    noise = np.random.uniform(-1, 1, size=(size, noisesize))
    return noise
"""NY GENERATOR"""
def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def generator1():
    model=Sequential()
    model.add(Dense(128*8*8,input_shape=[10],kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Reshape([8,8,128]))
    model.add(UpSampling2D((2,2)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D((2,2)))
    #model.add(BatchNormalization())
    model.add(Conv2D(1,(5,5),strides=(1,1),activation='tanh',padding='valid'))
    #model.add(BatchNormalization())

    #model.add(LeakyReLU(0.2))
    #model.add(Conv2DTranspose(1,(5,5),strides=(2,2),activation='tanh',padding='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Conv2DTranspose(1,(5,5),strides=(2,2),activation='tanh',padding='same'))
    #model.add(Flatten())
    model.summary()


    return model


def generator2():
    model=Sequential()
    model.add(Dense(256,input_shape=[100]))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(784))
    model.add(Reshape([28,28,1]))
    model.add(Conv2D(5,(9,9),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Reshape([5*784]))
    model.add(Dense(784,activation='tanh'))
    return model

def discriminator3():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28,1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optadam)
    return model

def discriminator2():
    model=Sequential()
    model.add(Dense(1024,input_shape=[784]))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation='sigmoid'))
    model.add(LeakyReLU(0.2))
    #model.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
    return model
def discriminator1():
    model=Sequential()
    model.add(Conv2D(128,(5,5),strides=(2,2),input_shape=(32,32,1),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(1,(4,4),strides=(2,2),padding='valid',activation='sigmoid'))
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Conv2D(1,(5,5),strides=(2,2),activation='sigmoid',padding='same'))
    model.add(Reshape([1]))
    #model.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])

    return model




""" NY DISKRIMINATOR"""
def discriminator():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model



def settrainable(discmodel,Boolean):
    if Boolean == True:
        for i in discmodel.layers:
            i.trainable=True
    else:
        for i in discmodel.layers:
            i.trainable=False
    #discmodel.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])



def creategans(discmodel,genmodel):
    gansmodel=Sequential()
    gansmodel.add(genmodel)
    discmodel.trainable=False
    gansmodel.add(discmodel)
    #gansmodel.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])

    return gansmodel




noise_test=getnoise(5**2)
def showim(genmodel,index,noise):
    n_ims=5
    lk=28
    #noise = getnoise(n_ims**2)

    generated = genmodel.predict(noise).reshape([n_ims,n_ims,lk,lk])

    filename="im"+str(index)+".png"
    imlist=[]
    j=0
    imtot=np.zeros([lk*n_ims,lk*n_ims])
    n=0
    m=0

    #print(imtot.shape)
    for i in range(n_ims):
        m=0
        for j in range(n_ims):
            #print(generated[i,j].shape)
            imtot[n:n+lk,m:m+lk]=generated[i,j]

            m+=lk
        n+=lk
    plt.axis('off')
    plt.imshow(imtot,cmap='gray')
    plt.savefig("ims/"+filename)




def train():
    g=generator1()
    d=discriminator3()
    g.summary()
    d.summary()
    d_on_g=creategans(d,g)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    images=getdata()
    #a,b,c,d1=images.shape
    #images=images.reshape([a,784])
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #images=mnist.train.images.astype('float32')
    #i1,i2,i3,i4=images.shape
    i1,i2,i3,i4=images.shape
    print(i1)
    epochs=50
    batch_size=128//2
    k=0
    for i in range(epochs):
        for j in range(i1//batch_size):
            noise=getnoise(batch_size*2)
            noise_images=g.predict(noise)

            d.train_on_batch(images[j*batch_size:(j+1)*batch_size],np.ones([batch_size,1]))
            ld=d.train_on_batch(noise_images[0:batch_size],np.zeros([batch_size,1]))
            print("Epoch: ",i," D Loss: ",ld)

            d.trainable=False
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            lg=d_on_g.train_on_batch(noise,np.ones([batch_size*2,1]))
            d.trainable=True
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            print("Epoch: ",i," G Loss: ", lg)
            if (j%50==0):
                showim(g,k,noise_test)
                k=k+1
            if (j%100==0):
                print(j)

train()
