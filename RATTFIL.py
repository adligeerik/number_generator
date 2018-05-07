import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


np.random.seed(42)
optadam=optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
init=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
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
        trainsize = min(self.fake_lables.shape[0],self.real_lables.shape[0])
        self.lables = np.vstack((self.real_lables[:trainsize],self.fake_lables[:trainsize]))
        self.combineimages()
        self.shuffledata()

    def combineimages(self):
        # Make the combined image vector equal part real and fake
        trainsize = min(self.fake_lables.shape[0],self.real_lables.shape[0])
        # Combine real and fake images
        fromindex = np.random.randint(self.real_lables.shape[0]-trainsize,size=1)[0]
        self.images = np.vstack((self.real_images[fromindex:fromindex+trainsize],self.fake_images[:trainsize]))

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


def getdata(digit=1):
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

    #images=ims
    print(images.shape)


    images=images.reshape(images.shape[0],28,28,1)
    images=np.pad(images,((0,0),(2,2),(2,2),(0,0)),'constant')
    images=2*(images-0.5)
    plt.imshow(images[3,:,:,0])
    plt.show()

    plt.imshow(images[444,:,:,0])
    plt.show()
    #print(images.shape)
    return images





def getnoise(size):
    """Generera brus till generatorn"""
    noisesize=100
    noise=np.random.normal(0,1,(size,noisesize))
    return noise
"""NY GENERATOR"""
def generator():
    model=Sequential()
    model.add(Dense(16384,input_shape=[100]))
    model.add(Reshape([4,4,1024]))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(512,(5,5),strides=(2,2),activation ='relu',padding='same'))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(256,(5,5),strides=(2,2),activation ='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1,(5,5),strides=(2,2),activation='tanh',padding='same'))
    model.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])

    return model


""" NY DISKRIMINATOR"""
def discriminator():
    model=Sequential()
    model.add(Conv2D(128//2,(5,5),strides=(2,2),input_shape=(32,32,1),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(256//2,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(512//2,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(1,(4,4),strides=(2,2),padding='valid',activation='sigmoid'))
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Conv2D(1,(5,5),strides=(2,2),activation='sigmoid',padding='same'))
    model.add(Reshape([1]))
    model.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])

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
    discmodel.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])



def creategans(discmodel,genmodel):
    gansmodel=Sequential()
    gansmodel.add(genmodel)
    gansmodel.add(discmodel)

    gansmodel.compile(loss='binary_crossentropy',optimizer=optadam,metrics=['accuracy'])

    return gansmodel



def shuffle(batch_size,nr_images=55000):
    random_vec=np.arange(nr_images)
    return random_vec[0:batch_size]

def showim(genmodel,index):
    noise = getnoise(1)
    generated = genmodel.predict(noise)
    filename="im"+str(index)+".png"
    plt.imshow(generated.reshape([32,32]),cmap='gray')
    plt.savefig(filename)
    #plt.show()


#def train():
images = getdata(1)
discmodel = discriminator()
genmodel = generator()



def dist(truemat,falsemat):
    return np.linalg.norm(truemat-falsemat)





l=100
for i in range(200):
    noise = getnoise(l//2)
    noise_images =  genmodel.predict(noise)

    #traindata = Traindata(images,noise_images)
    settrainable(discmodel,True)

    #discmodel.fit(traindata.images_shuf,traindata.lables_shuf[:,0],batch_size=100,epochs=1,verbose=2)
    print("Discriminator")
    discmodel.fit(images[i*l//2:(i+1)*l//2,:,:,:],np.ones([l//2,1]),batch_size=l//2,epochs=1,verbose=2)
    discmodel.fit(noise_images,np.zeros([l//2,1]),batch_size=l//2,epochs=1,verbose=2)
    settrainable(discmodel,False)

    noise = getnoise(l)
    gansmodel = creategans(discmodel,genmodel)
    print("Generator")
    gansmodel.fit(noise, np.ones([l,1]),batch_size=l,epochs=1,verbose=2)
    distance=dist(images[0],noise_images[0])
    print("Distance:")
    print(distance)

    if (i%5==0):
        print(i)
        showim(genmodel,i//5)

for i in range(40):
    noise = getnoise(50)
    noise_images =  genmodel.predict(noise)

    traindata = Traindata(images,noise_images)
    settrainable(discmodel,True)

    discmodel.fit(traindata.images_shuf,traindata.lables_shuf[:,0],batch_size=100,epochs=1,verbose=2)
    settrainable(discmodel,False)

    noise = getnoise(100)
    gansmodel = creategans(discmodel,genmodel)
    gansmodel.fit(noise, np.ones([100,1]),batch_size=100,epochs=1,verbose=2)





#train()



#images=getdata()
##noise=getnoise(1)
#gen=generator()
#disc=discriminator()
#disc.summary()
#gen.summary()
#noise=getnoise(1)
#gen=test()
#disc=discriminator()
#fake=gen.predict(noise)
#fake2=disc.predict(fake)
##fake2=disc.predict(fake)
#print(fake2)
