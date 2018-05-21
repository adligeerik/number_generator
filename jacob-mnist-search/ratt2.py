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
from data import loaddata,loadlables
from models import loadmodel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


np.random.seed(42)
optadam=optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
init=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)




def getnoise(size):
    """Generera brus till generatorn"""
    noisesize=10
    #noise=np.random.normal(0,1,(size,noisesize))
    noise = np.random.uniform(-1, 1, size=(size, noisesize))
    return noise
"""NY GENERATOR"""

def getfakelabels(size):
    fakelabels = np.eye(10,10)[np.random.choice(10, size)]
    return fakelabels



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
def showim(genmodel,index,noise,datashape):
    n_ims=5
    lk=datashape[1]
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

    imtot=(imtot+1)/2
    print('Printing to: '+filename)
    plt.axis('off')
    plt.imshow(imtot)
    #plt.show()
    plt.savefig("imscats/"+filename)




def train():
    dataset="mnist"
    images=loaddata(dataset)
    labels=loadlables(dataset)
    g,d=loadmodel(dataset)
    LR = 0.0002  # initial learning rate
    B1 = 0.5 # momentum term
    opt = Adam(lr=LR,beta_1=B1)
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
    epochs=40
    batch_size=128//2
    k=0
    filename= "imscats/"+str(dataset)+"_n_epochs_"+str(epochs)+".h5"
    for i in range(epochs):
        g.save(filename)
        for j in range(i1//batch_size):
            noise=getfakelabels(batch_size*2)
            noise2 = noise + np.random.uniform(0, 0.1, size=(batch_size*2, 10))
            noise_images=g.predict(noise2)

            ldr=d.train_on_batch(images[j*batch_size:(j+1)*batch_size],labels[j*batch_size:(j+1)*batch_size])
            ld=d.train_on_batch(noise_images[0:batch_size], np.zeros([batch_size,10]))
            #ld = 1;
            print("Epoch: ",i," D Loss: ",ld, ldr)
            #d.trainable=False
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            lg=d_on_g.train_on_batch(noise2,noise)
            #d.trainable=True
            #d.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])
            print("Epoch: ",i," G Loss: ", lg)
            if (j%500==0):
                k=k+1
                showim(g,k,np.eye(25,11),[i1,i2,i3,i4])


train()
