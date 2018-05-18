#lab02-4 model
import tensorflow as tf
import numpy as np

pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')

pendigits_data = np.append(pendigits_train, pendigits_test, axis = 0)
nsamples = np.size(pendigits_data, 0)

np.random.seed(0)
mask = np.random.permutation(nsamples)
pendigits_data = pendigits_data[mask]

x_data = pendigits_data[:,:-1]
y_data = pendigits_data[:,[-1]].astype(int)

ndim = np.size(x_data, 1)

ntrain = int(nsamples*0.7)
nvalidation = int(nsamples*0.1)
ntest = nsamples-ntrain-nvalidation

x_train = x_data[:ntrain]
x_validation = x_data[ntrain:(ntrain+nvalidation)]
x_test = x_data[-ntest:]

print("The number of total samples : ", nsamples)
print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)
