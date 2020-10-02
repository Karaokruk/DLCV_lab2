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

# one-hot encode labels
digits = 10

def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new

y_train = one_hot_encode(y_train, digits)
y_test = one_hot_encode(y_test, digits)

m = X_train.shape[1] #number of examples
n = X_train.shape[0]

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

def sigmoid(z):
    return 1./(1. + np.exp(-z))

# softmax(zi) = e^zi / sum_j_in[1;10](e^zj)
def softmax(z, i):
    sum = 0
    for j in range(digits):
        sum += np.sum(np.exp(z[j]))
    return np.exp(z[i]) / sum

def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum
    return L

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    return -(1./m) * (np.sum(np.multiply(Y, np.log(Y_hat))) + np.sum(np.multiply(1 - Y, np.log(1 - Y_hat))))

# -sum(yi*log(yhati))
# sum(yhati)=1

def printStats(epoch, loss, accuracy):
    print("Simple NN -- Epoch : ", epoch + 1)
    print("Loss : {:.10f}".format(loss))
    print("Accuracy : {:.2f}%".format(accuracy * 100))

def simpleNN(training_kit, testing_kit, nb_epochs, lr, verbose = True):
    (X, Y) = training_kit
    accuracies = []
    losses = []

    W = np.random.randn(digits, n) * 0.01
    b = np.zeros((digits, 1))
    for epoch in range(nb_epochs):
        Z = np.matmul(W, X) + b
        Y_hat = sigmoid(Z)

        loss = compute_multiclass_loss(Y, Y_hat)
        losses.append(loss)

        accuracy = checkAccuracy(testing_kit, (W, b))
        accuracies.append(accuracy)

        if verbose:
            printStats(epoch, loss, accuracy)

        dW = (1./m) * np.matmul(Y_hat - Y, X.T)
        db = (1./m) * np.sum(Y_hat - Y, axis=1, keepdims=True)

        W = W - lr * dW
        b = b - lr * db

    return (losses, accuracies)

def hidden64NN(training_kit, testing_kit, nb_epochs, lr, verbose = True):
    (X, Y) = training_kit
    accuracies = []
    losses = []

    nh = 64
    W1 = np.random.randn(nh, n) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(digits, nh) * 0.01
    b2 = np.zeros((digits, 1))

    for epoch in range(nb_epochs):
        Z1 = np.matmul(W1, X) + b1
        Y1 = sigmoid(Z1)
        Z2 = np.matmul(W2, Y1) + b2
        Y_hat = sigmoid(Z2)

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

        loss = compute_multiclass_loss(Y, Y_hat)
        losses.append(loss)

        accuracy = checkAccuracy(testing_kit, ((W1, b1), (W2, b2)))
        accuracies.append(accuracy)

        if verbose:
            printStats(epoch, loss, accuracy)

    return (losses, accuracies)

def checkAccuracy(testing_kit, trained, verbose = False):
    # Test accuracy (on X_test, y_test)
    if isinstance(trained[0], tuple): # if the trained data has more than 1 layer
        (X1, Y1) = testing_kit
        (W1, b1) = trained[0]
        Y2 = sigmoid(np.matmul(W1, X1) + b1)
        testing_kit = (Y2, Y1)
        trained = trained[1]
    (X, Y) = testing_kit
    (W, b) = trained
    Y_hat = sigmoid(np.matmul(W, X) + b)

    if verbose:
        print("testing kit :")
        for row in Y:
            print(row)
        print("trained kit :")
        for row in Y_hat:
            print(row)
    batch_size = len(Y[0])
    cpts = [0] * digits
    for i in range(batch_size):
        for d in range(digits):
            tmp = 0.0 if Y_hat[0][i] < 0.5 else 1.0
            if tmp == y_test[0][i]:
                cpts[d] += 1

    accuracies = []
    for d in range(digits):
        accuracies.append(cpts[d] * 1. / batch_size)

    global_accuracy = 0
    for accuracy in accuracies:
        global_accuracy += accuracy
    return global_accuracy / digits


training_kit = (X_train, y_train)
testing_kit = (X_test, y_test)
nb_epochs = 5
lr = 0.10
verbose = True
np.random.seed()

runSimpleNN = True
runHidden64NN = True

## Simple Neuronal Network training & testing
(simpleNN_losses, simpleNN_accuracies) = ([], []) if not runSimpleNN else simpleNN(training_kit, testing_kit, nb_epochs, lr, verbose)
## Hidden 64-sized layer Neuronal Network training & testing
(hidden64NN_losses, hidden64NN_accuracies) = ([], []) if not runHidden64NN else hidden64NN(training_kit, testing_kit, nb_epochs, lr, verbose)

epoch_list = list(range(nb_epochs))

## Simple Neuronal Network plot
if runSimpleNN:
    plt.plot(epoch_list, simpleNN_losses, label='loss')
    plt.plot(epoch_list, simpleNN_accuracies, label='accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.legend()
    plt.title('SimpleNN accuracy & loss')
    plt.savefig("simpleNNaccuracyloss.png")
    plt.show()

## Hidden 64 Neuronal Network plot
if runHidden64NN:
    plt.plot(epoch_list, hidden64NN_losses, label='loss')
    plt.plot(epoch_list, hidden64NN_accuracies, label='accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.legend()
    plt.title('Hidden64NN accuracy & loss')
    plt.savefig("hidden64NNaccuracyloss.png")
    plt.show()

## Simple Neuronal Network vs Hidden 64 Neuronal Network plot
if runSimpleNN and runHidden64NN:
    plt.plot(epoch_list, simpleNN_accuracies, label='simpleNN')
    plt.plot(epoch_list, hidden64NN_accuracies, label='hidden64NN')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('simpleNN vs hidden64NN accuracy')
    plt.savefig("simpleNNvshiddenNNaccuracy.png")
    plt.show()
