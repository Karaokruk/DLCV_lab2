from __future__ import print_function
import numpy as np

#In this first part, we just prepare our data (mnist) 
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist

def sigmoid(z):
    return 1/(1+np.exp(-z))

def outputlayer(x,w,b):
    z=0
    for i in range(len(x)):
        z+=x[i]*w[i]
    return sigmoid(z+b)

def lossfunction(x,w,b, trueoutput):
    return -(outputlayer(x,w,b)*numpy.log(trueoutput))

def derivlossforw(x,w,b,trueoutput):
    s = 0
    for i in range(len(w)):
        s+=w[i]*x[i]
    return numpy.exp(b+s)*numpy.log(trueoutput) / (1+numpy.exp(b+s))

(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


# #Display one image and corresponding label 
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron. 


#####TO COMPLETE



