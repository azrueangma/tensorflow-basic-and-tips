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

print("=== numpy ===")
print("The total number of train samples : {}".format(ntrain))
print("The total number of test samples : {}".format(ntest))
print("The dimension of samples : {}".format(ndim))
print("The number of classes : {}".format(nclass))

np.savetxt(fname = './data/lab04_2/pendigits_train_x.csv', X = x_train, fmt = "%d", delimiter = ',')
np.savetxt(fname = './data/lab04_2/pendigits_train_y.csv', X = y_train, fmt = "%d", delimiter = ',')
np.savetxt(fname = './data/lab04_2/pendigits_test_x.csv', X = x_test, fmt = "%d", delimiter = ',')
np.savetxt(fname = './data/lab04_2/pendigits_test_y.csv', X = y_test, fmt = "%d", delimiter = ',')

'''
=== numpy ===
The total number of train samples : 7494
The total number of test samples : 3498
The dimension of samples : 16
The number of classes : 10
'''

import pandas as pd

pd_pendigits_train = pd.read_csv('./data/pendigits_train.csv', header = None)
pd_pendigits_test = pd.read_csv('./data/pendigits_test.csv', header = None)

pd_x_train = pd_pendigits_train.values[:,:-1]
pd_y_train = pd_pendigits_test.values[:,[-1]].astype(np.int32)
pd_x_test = pd_pendigits_test.values[:,:-1]
pd_y_test = pd_pendigits_test.values[:,[-1]].astype(np.int32)

pd_ntrain = np.size(pd_x_train,0)
pd_ntest = np.size(pd_x_test, 0)
pd_ndim = np.size(pd_x_train,1)
pd_nclass = len(np.unique(pd_y_train))

pd_pendigits_train.to_csv('./data/lab04_2/pd_pendigits_train.csv', header = False, index = False)
pd_pendigits_test.to_csv('./data/lab04_2/pd_pendigits_test.csv', header = False, index = False)

print("\n=== pandas ===")
print("The total number of train samples : {}".format(pd_ntrain))
print("The total number of test samples : {}".format(pd_ntest))
print("The dimension of samples : {}".format(pd_ndim))
print("The number of classes : {}".format(pd_nclass))

'''
=== pandas ===
The total number of train samples : 7494
The total number of test samples : 3498
The dimension of samples : 16
The number of classes : 10
'''
