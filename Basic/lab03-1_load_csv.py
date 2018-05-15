import numpy as np

pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')

x_train = pendigits_train[:,:-1]
y_train = pendigits_train[:,[-1]].astype(np.int32)
x_test = pendigits_test[:,:-1]
y_test = pendigits_test[:,[-1]].astype(np.int32)

ntrain = np.size(x_train,0)
ntest = np.size(x_test, 0)
ndim = np.size(x_train,1)
nclass = len(np.unique(y_train))
print("The total number of train samples : {}".format(ntrain))
print("The total number of test samples : {}".format(ntest))
print("The dimension of samples : {}".format(ndim))
print("The number of classes : {}".format(nclass))

'''
The total number of train samples : 7494
The total number of test samples : 3498
The dimension of samples : 16
The number of classes : 10
'''
