import numpy as np
import tensorflow as tf

#for Big Data
def MinMaxScaler(x):
    col_min = np.min(x, axis = 0)
    col_max = np.max(x, axis = 0)
    denominator = (col_max - col_min) + 1e-7
    numerator = x - col_min
    return numerator/denominator

SEED = 0

#Assume that each pendigits data samples is too large to be loaded
pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')

pendigits_data = np.append(pendigits_train, pendigits_test, axis = 0)
nsamples = np.size(pendigits_data, 0)

np.random.seed(SEED)
mask = np.random.permutation(nsamples)
pendigits_data = pendigits_data[mask]
x_data = MinMaxScaler(pendigits_data[:,:-1])
y_data = pendigits_data[:,-1].astype(int)

ndim = np.size(x_data, 1)

for i in range(len(pendigits_train)):
    writer = tf.python_io.TFRecordWriter('./data/lab04_7/pendigits{}.tfrecords'.format(i))
    example = tf.train.Example(features = tf.train.Features(feature = {
        'features':tf.train.Feature(float_list = tf.train.FloatList(value = x_data[i,:].flatten())),
        'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [y_data[i]]))
        }))
    writer.write(example.SerializeToString())
writer.close()
