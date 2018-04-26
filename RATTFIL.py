import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU

np.random.seed(42)

class Traindata:

    def __init__(self,real_images, fake_images):
        self.real_images = real_images
        self.fake_images = fake_images

        r_ones = np.ones([real_images.shape[0],1])
        r_zeros = np.zeros([real_images.shape[0],1])

        f_ones = np.ones([fake_images.shape[0],1])
        f_zeros = np.zeros([fake_images.shape[0],1])

        # Create lables for real and fake images
        self.real_lables = np.hstack((r_ones,r_zeros))
        self.fake_lables = np.hstack((f_zeros,f_ones))
        self.lables = np.vstack((self.real_lables,self.fake_lables))
        self.combineimages()
        self.shuffledata()
    
    def combineimages(self):
        # Make the combined image vector equal part real and fake
        trainsize = np.abs(self.fake_lables.shape[0]-self.real_lables.shape[0])
        # Combine real and fake images
        fromindex = np.random.randint(self.real_lables.shape[0]-trainsize,size=1)[0]
        self.images = np.vstack((self.real_images[fromindex:trainsize],self.fake_images[:trainsize]))

    def shuffledata(self):
        # Creat random vector for shuffeling
        rand_vec = np.arange(self.images.shape[0])
        np.random.shuffle(rand_vec)
        self.lables_shuf = self.lables[rand_vec]
        self.images_shuf = self.images[rand_vec]
    
    def shufflenewdata(self,fake_images):
        # Update the shuffle images from new faked images
        self.fake_images = fake_images
        self.combineimages()
        rand_vec = np.arange(self.images.shape[0])
        np.random.shuffle(rand_vec)
        self.lables_shuf = self.lables[rand_vec]
        self.images_shuf = self.images[rand_vec]


def getdata():
    """Hämta bilderna"""
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    images=mnist.train.images.astype('float32')
    images=images.reshape(images.shape[0],28,28,1)
    images=np.pad(images,((0,0),(2,2),(2,2),(0,0)),'constant')
    images=2*(images-0.5)
    print(images.shape)
    return images

def getnoise(size):
    """Generera brus till generatorn"""
    noisesize=100
    noise=np.random.normal(0,1,(size,noisesize))
    return noise

def test ():
    model=Sequential()
    model.add(Dense(16384,input_shape=[100],trainable=False))
    model.add(Reshape([4,4,1024]))
    model.add(Conv2DTranspose(512,(2,2),strides=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(256,(2,2),strides=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(1,(2,2),strides=(2,2),activation='tanh'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def creategenerator():
    model=Sequential()
    model.add(Dense(128,activation='relu',input_shape=[100]))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(784,activation='relu'))
    model.add(Reshape([28,28,1]))
    return model


def creatediscriminator():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,1)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model
    

def settrainable(discmodel,Boolean):
    if Boolean == True:
        for i in discmodel.layers:
            i.trainable=True
    else:
        for i in discmodel.layers:
            i.trainable=False
    discmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


def creategans(discmodel,genmodel):
    gansmodel=Sequential()
    gansmodel.add(genmodel)
    gansmodel.add(discmodel)
    gansmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return gansmodel



def shuffle(batch_size,nr_images=55000):
    random_vec=np.arange(nr_images)
    return random_vec[0:batch_size]


def train():
    images=getdata()
    

    
    discmodel=creatediscriminator()
    discmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    genmodel=creategenerator()
    genmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    noise=getnoise(55000)
    fake_images=genmodel.predict(noise)
    

    
    batch_size=55000
    ones=np.ones([batch_size,1])
    zeros=np.zeros([batch_size,1])

    real_labels=np.hstack((ones,zeros))
    fake_labels=np.hstack((zeros,ones))


    for i in range(10):
        settrainable(discmodel,True)
        discmodel.fit(shuffled_images[0:100],shuffled_labels[0:100],batch_size=100,verbose=1)
        
        settrainable(discmodel,False)
        gansmodel=creategans(discmodel,genmodel)
        gansmodel.fit(noise[0:100], real_labels[0:100],batch_size=100,verbose=1)
images=getdata()
noise=getnoise(1)
gen=test()
disc=creatediscriminator()
fake=gen.predict(noise)
fake2=disc.predict(fake)
print(fake2.shape)
