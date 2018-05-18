from models import *
from ratt2 import showim,getnoise,showim1
from data import loaddata
# import os
# import tensorflow as tf
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
# from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization, UpSampling2D
# from keras.utils import np_utils
# from keras.layers.advanced_activations import LeakyReLU
# from keras import optimizers, initializers
# from keras.optimizers import SGD

g,d=loadmodel('flowers128ja.h5')
# noise1=getnoise(200)
#
# ims=0.5*(g.predict(noise1)+1)
# for i in range(200):
#     plt.imshow(ims[i])
#     plt.axis('off')
#     plt.show()
#im1=showim(g,0,noise1,[0,28,28,1])

#noise2=getnoise(25)
#im2=showim(g,0,noise2,[0,28,28,1])


#noise4=getnoise(25)
#im4=showim1(g,0,noise4,[0,28,28,1])
vec=np.zeros([1,10])
k=-1
ok=2/10
imtot=np.zeros([128,10*128,3])
imall=np.zeros([128*10,10*128,3])
apa=0
for j in range(10):
    vec=np.zeros([1,10])
    vec=vec+np.random.uniform(-0.5, 0.5, size=(1, 10))
    k=-1
    if j in [0,1,2,3,4,5,6,7,8,9]:
        for i in range(10):
            print(i)
            vec[0,j]=k
            k=k+ok
            im=g.predict(vec)
            imtot[:,i*128:(i+1)*128]=im.reshape([128,128,3])
        imall[apa*128:(apa+1)*128,:,:]=imtot
        apa+=1

#imall=imall[[1*28:2*28,3*28:4*28,4*28:5*28,6*28:7*28]]
imall=0.5*(imall+1)
plt.imshow(imall)
plt.axis('off')
plt.show()




# imtot=np.hstack((im1,im2,im4))
# a=28*2*5
# imtot[:,a-1:a+1]=1
# plt.axis('off')
# plt.imshow(imtot[:,:,0],cmap='gray')
# plt.show()
# ims=0.5*(g.predict(noise)+1)
# for i in range(500):
#     plt.imshow(ims[i])
#     plt.show()
# for k in range(10):
#     noise=np.zeros([64,10])
#     noise=getnoise(64)
#     max=2
#     ökning=max/64
#     j=-1
#     #for i in range(64):
#         #noise[i,k]=j
#         #j+=ökning
#     #print(noise)
#     showim(g,0,noise,[0,128,128,3])
