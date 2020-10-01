from __future__ import print_function
import numpy as np

#In this first part, we just prepare our data (mnist)
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
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
np.random.seed()
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


## Display one image and corresponding label
import matplotlib
import matplotlib.pyplot as plt
'''
i = 0
print('y[{}]={}'.format(i, y_train[:,i]))
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
'''

#Let start our work: creating a neural network
#First, we just use a single neuron.
def sigmoid(z):
    return 1./(1. + np.exp(-z))

def crossEntropy(y, Y_hat):
    return -(1./m) * (np.sum(np.multiply(y, np.log(Y_hat))) + np.sum(np.multiply(1-y, np.log(1-Y_hat))))

n = X_train.shape[0]

def simpleNN(kit, nb_epochs, lr):
    (X, Y) = kit
    accuracies = []
    losses = []

    W = np.random.randn(1, n) * 0.01
    b = np.zeros((1, 1))
    for i in range(nb_epochs):
        Z = np.matmul(W, X) + b
        Y_hat = sigmoid(Z)

        print("Epoch : ", i + 1)

        loss = crossEntropy(Y, Y_hat)
        losses.append(loss)
        print("Loss value : ", loss)

        accuracy = checkAccuracy((X_test, y_test), (W, b))
        accuracies.append(accuracy)
        print("Accuracy : ", accuracy)

        dW = (1./m) * np.matmul(Y_hat - Y, X.T)
        db = (1./m) * np.sum(Y_hat - Y, axis=1, keepdims=True)

        W = W - lr * dW
        b = b - lr * db

    return (losses, accuracies)

def hidden64NN(kit, nb_epochs, lr):
    (X, Y) = kit
    accuracies = []
    losses = []

    nh = 64
    W1 = np.random.randn(nh, n) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(1, nh) * 0.01
    b2 = np.zeros((1, 1))

    for i in range(nb_epochs):
        Z1 = np.matmul(W1, X) + b1
        Y1 = sigmoid(Z1)
        Z2 = np.matmul(W2, Y1) + b2
        Y_hat = sigmoid(Z2)

        loss = crossEntropy(Y, Y_hat)
        dZ2 = Y_hat - Y
        dW2 = (1./m) * np.matmul(dZ2, Y1.T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

        dY1 = np.matmul(W2.T, dZ2)
        dZ1 = dY1 * Y1 * (1 - Y1)
        dW1 = (1./m) * np.matmul(dZ1, X.T)
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1

        print("Epoch : ", i + 1)

        loss = crossEntropy(Y, Y_hat)
        losses.append(loss)
        print("Loss value : ", loss)

        accuracy = checkAccuracy((X_test, y_test), ((W1, b1), (W2, b2)))
        accuracies.append(accuracy)
        print("Accuracy : ", accuracy)

    return (losses, accuracies)

def checkAccuracy(kit, trained):
    # Test accuracy (on X_test, y_test)
    if isinstance(trained[0], tuple): # if the trained data has more than 1 layer
        (X1, Y1) = kit
        (W1, b1) = trained[0]
        Y2 = sigmoid(np.matmul(W1, X1) + b1)
        kit = (Y2, Y1)
        trained = trained[1]
    (X, Y) = kit
    (W, b) = trained
    Y_hat = sigmoid(np.matmul(W, X) + b)

    nb_digits = len(Y[0])
    cpt = 0
    for i in range(nb_digits):
        tmp = 0.0 if Y_hat[0][i] < 0.5 else 1.0
        if tmp == y_test[0][i]:
            cpt += 1

    return cpt * 1. / nb_digits


nb_epochs = 20
lr = 0.10
np.random.seed()

## Simple Neuronal Network training
(simpleNN_losses, simpleNN_accuracies) = simpleNN((X_train, y_train), nb_epochs, lr)
## Hidden 64 Neuronal Network training
(hidden64NN_losses, hidden64NN_accuracies) = hidden64NN((X_train, y_train), nb_epochs, lr)

epoch_list = list(range(nb_epochs))

simpleNN_plot = False
hidden64NN_plot = True
simpleNN_vs_hidden64NN_plot = False

## Simple Neuronal Network plot
if simpleNN_plot:
    plt.plot(epoch_list, simpleNN_losses, label='loss')
    plt.plot(epoch_list, simpleNN_accuracies, label='accuracy')
    #plt.plot(okyo, hidd, label='MSE')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.legend()
    plt.title('SimpleNN accuracy & loss')
    plt.savefig("simpleNNaccuracyloss.png")
    plt.show()

## Hidden 64 Neuronal Network plot
if hidden64NN_plot:
    plt.plot(epoch_list, hidden64NN_losses, label='loss')
    plt.plot(epoch_list, hidden64NN_accuracies, label='accuracy')
    #plt.plot(okyo, hidd, label='MSE')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.legend()
    plt.title('Hidden64NN accuracy & loss')
    plt.savefig("hidden64NNaccuracyloss.png")
    plt.show()

## Simple Neuronal Network vs Hidden 64 Neuronal Network plot
if simpleNN_vs_hidden64NN_plot:
    plt.plot(epoch_list, simpleNN_accuracies, label='simpleNN')
    plt.plot(epoch_list, hidden64NN_accuracies, label='hidden64NN')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('simpleNN vs hidden64NN accuracy')
    plt.savefig("simpleNNvshiddenNNaccuracy.png")
    plt.show()
