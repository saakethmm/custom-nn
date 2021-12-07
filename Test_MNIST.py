from Network import Network
from FCLayer import FCLayer
from ActivationFunction import ActivationLayer
from LossFunction import mse, mse_prime
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

'''
network = Network(mse, mse_prime)
network.add_layer(FCLayer(28*28, 100))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(100, 50))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(50, 10))
#network.add_layer(ActivationLayer('tan'))
'''

network = Network(mse, mse_prime)
network.add_layer(FCLayer(28*28, 28*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(28*7, 14*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(14*7, 7*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(7*7, 10))

# Change everything to use numpy arrays

train_accuracy = []
test_accuracy = []
losses = []
train_vals = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

for num_train in train_vals:
    num_epochs = 50
    learning_rate = 0.1
    train_loss, val_loss = network.train(x_train[0:num_train], y_train[0:num_train], niter=num_epochs, lr=learning_rate)

    num_test = 10000
    results = network.predict(x_test[0:num_test])
    results_test, acc_test = network.accuracy(results, y_test[0:num_test])

    results2 = network.predict(x_train[0:num_train])
    results_train, acc_train = network.accuracy(results2, y_train[0:num_train])

    plt.plot(range(num_epochs), train_loss, '-b', label="Train")
    plt.plot(range(num_epochs), val_loss, '-k', label="Validation")
    plt.xlabel('epoch #')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('MSE Loss with ' + str(num_train) + ' Training Samples\n(85% Train, 15% Validation)')
    plt.savefig('MNIST_Loss_' + str(num_train) + '.png')
    plt.clf()

    acc_train *= 100
    train_accuracy.append(acc_train)
    acc_test *= 100
    test_accuracy.append(acc_test)


plt.plot(train_vals, train_accuracy, '-b', label="Train")
plt.plot(train_vals, test_accuracy, '-r', label="Test")
plt.xlabel('# of training samples')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Change in Train/Test Accuracy with Increasing\n# Training Samples')
plt.savefig('MNIST_accuracy.png')



# print("The Train Accuracy is " + ("{:.2f}".format(acc_train*100)) + "%")
# print("The Test Accuracy is " + ("{:.2f}".format(acc_test*100)) + "%")

