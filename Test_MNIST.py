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
network.add_layer(FCLayer(100, 10))
'''


network = Network(mse, mse_prime)
network.add_layer(FCLayer(28*28, 28*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(28*7, 14*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(14*7, 7*7))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(7*7, 10))

# TODO: Nearly 32 minutes to train 60000 training data size network (18->19->23->28)

num_train = 60000
num_epochs = 100
learning_rate = 0.1
loss = network.train(x_train[0:num_train], y_train[0:num_train], niter=num_epochs, lr=learning_rate)
plt.plot(range(num_epochs), loss)
plt.xlabel('epoch #')
plt.ylabel('Training MSE Loss')
plt.title('MNIST Training Loss with 5000 training samples')
plt.savefig('MNIST_Test2.png')


num_test = 10000
results = network.predict(x_test[0:num_test])
results_test, acc_test = network.accuracy(results, y_test[0:num_test])

results2 = network.predict(x_train[0:num_train])
results_train, acc_train = network.accuracy(results2, y_train[0:num_train])

print("The Train Accuracy is " + ("{:.2f}".format(acc_train*100)) + "%")
print("The Test Accuracy is " + ("{:.2f}".format(acc_test*100)) + "%")

