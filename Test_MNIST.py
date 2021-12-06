from Network import Network
from FCLayer import FCLayer
from ActivationFunction import ActivationLayer
from LossFunction import mse, mse_prime

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
network.add_layer(FCLayer(28*28, 100))
network.add_layer(ActivationLayer('tan'))
network.add_layer(FCLayer(100, 10))


network.train(x_train[0:1000], y_train[0:1000], niter=35, lr=0.1)
num_test = 100
results = (network.predict(x_test[0:num_test]))
#print(results)
results2, acc = network.accuracy(results, y_test[0:num_test])
print(acc)
