import numpy as np

'''
1.For Classification Problem
Data frame : (features, class)
Data format : csv
'''

iris_data = np.loadtxt("./data/iris.csv", delimiter=',')
features = iris_data[:, :-1]
labels = iris_data[:, -1]

nsamples = len(iris_data)
ndims = np.size(features, 1) #np.shape(features)[1]
nclass = len(set(labels))

print("#"*15,"Data Summary", "#"*15)
print("\n> The number of data samples   : {:d}".format(nsamples))
print("> Dimensions of data           : {:d}".format(ndims))
print("> The number of classes        : {:d}\n".format(nclass))


'''
2.For Regression Problem
Data frame : (features, target)
Data format : csv
'''

nile_data = np.loadtxt("./data/nile.csv", delimiter=',')
features = nile_data[:, :-1]
targets = nile_data[:, -1]

nsamples = len(nile_data)
ndims = np.size(features, 1)

print("#"*15, "Data Summary", "#"*15)
print("\n> The number of data samples   : {:d}".format(nsamples))
print("> Dimensions of data           : {:d}".format(ndims))

print(targets)