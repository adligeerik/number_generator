import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization, UpSampling2D
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers
from keras.optimizers import SGD,Adam
from keras.datasets import cifar10

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

def getcifar(digit=None):
    """Hämta bilderna"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    images=np.float32(2*(np.vstack((x_train,x_test))/255-0.5))
    labels=np.vstack((y_train,y_test))
    o=np.zeros([60000,10])
    for i in range(60000):
        o[i,labels[i]]=1
    labels=o
    print(images.shape)
    print(labels.shape)
    if digit==None:
        images=images
    else:
        images=images
        labels=labels
        col=labels[:,digit]
        col=col!=0
        images=images[col]
    #images=images.reshape(images.shape[0],28,28,1)
    #images=np.pad(images,((0,0),(2,2),(2,2),(0,0)),'constant')
    #images=2*(images-0.5)
    return images



def getnoise(size):
    """Generera brus till generatorn"""
    noisesize=10
    #noise=np.random.normal(0,1,(size,noisesize))
    noise = np.random.uniform(-1, 1, size=(size, noisesize))
    return noise
"""NY GENERATOR"""



def generator():
    model=Sequential()
    model.add(Dense(128*8*8,input_shape=[10],kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Reshape([8,8,128]))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1,(5,5),strides=(1,1),activation='tanh',padding='valid'))
    model.summary()
    return model


def generator_cifar():
    model=Sequential()
    model.add(Dense(128*8*8,input_shape=[10],kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Reshape([8,8,128]))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(3,(5,5),strides=(1,1),activation='tanh',padding='same'))
    model.summary()
    return model


def discriminator_cifar():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(32, 32,3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model



def discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28,1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
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
    return gansmodel




noise_test=getnoise(5**2)
def showim(genmodel,index,noise):
    n_ims=5
    lk=32
    #noise = getnoise(n_ims**2)

    generated = genmodel.predict(noise).reshape([n_ims,n_ims,lk,lk,3])

    filename="im"+str(index)+".png"
    imlist=[]
    j=0
    imtot=np.zeros([lk*n_ims,lk*n_ims,3])
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
    plt.imshow(imtot)
    plt.savefig("ims7/"+filename)




def train():
    getcifar()
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #images=np.float32(*(np.vstack((x_train,x_test))/255-0.5))
    images=getcifar(9)
    print(images.shape)
    plt.imshow(images[234])
    plt.show()
    plt.imshow(images[9])
    plt.show()
    LR = 0.0002  # initial learning rate
    B1 = 0.5 # momentum term
    opt = Adam(lr=LR,beta_1=B1)
    g=generator_cifar()
    d=discriminator_cifar()

    g.summary()
    d.summary()



    d.trainable=True
    d.compile(loss='binary_crossentropy', optimizer=opt)
    g.compile(loss='binary_crossentropy', optimizer=opt)
    d.trainable=False
    d_on_g=creategans(d,g)
    d_on_g.compile(loss='binary_crossentropy', optimizer=opt)
    #images=getdata()
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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

            #d.trainable=False
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            lg=d_on_g.train_on_batch(noise,np.ones([batch_size*2,1]))
            #d.trainable=True
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            print("Epoch: ",i," G Loss: ", lg)
            if (j%50==0):
                showim(g,k,noise_test)
                k=k+1
            if (j%100==0):
                print(j)

train()
