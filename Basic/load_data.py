import numpy as np
from sklearn.datasets import load_digits
import gzip
import os
import urllib.request
import sys
import tarfile
import zipfile

def MinMaxScaler(x):
    col_min = np.min(x, axis = 0)
    col_max = np.max(x, axis = 0)
    denominator = (col_max - col_min) + 1e-7
    numerator = x - col_min
    return numerator/denominator, col_min, col_max

def MinMaxScaler_with(cur_col, min_col, max_col):
    numerator = cur_col - min_col
    denominator = max_col - min_col
    return numerator / (denominator + 1e-7)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def print_download_progress(count, block_size, total_size):
    decimals = 1
    format_str = "{0:."+str(decimals)+"f}"
    bar_length = 100
    pct_complete = format_str.format((float(count * block_size) / total_size)*100)
    total = int(total_size/block_size)+1
    filled_length = int(round(bar_length*count/total))
    if float(pct_complete) > 100.:
        pct_complete = "100"
    bar = '#'*filled_length+'-'*(bar_length-filled_length)
    sys.stdout.write('\r |%s| %s%s ' % (bar,  pct_complete, '%')),
    if pct_complete == 1.0:
        sys.stdout.write('\n')
    sys.stdout.flush()

def generate_data_for_linear_regression(npoints):
    for i in range(npoints):
        np.random.seed(i)
        x = np.random.normal(0.0, 0.5)
        noise = np.random.normal(0.0, 0.05)
        y = x * 0.2 + 0.5 + noise
        tmp = np.expand_dims(np.array([x, y]), axis=0)
        if i == 0:
            vectors = tmp
        else:
            vectors = np.append(vectors, tmp, axis=0)

    trainX = np.expand_dims(vectors[:, 0], axis=1)
    trainY = np.expand_dims(vectors[:, 1], axis=1)
    return trainX, trainY

def generate_data_for_two_class_classification(npoints):
    for i in range(npoints):
        np.random.seed(i)
        x1 = np.random.normal(0.0, 0.5)
        x2 = np.random.normal(2.0, 0.5)
        tmp1 = np.expand_dims(np.array([x1, 0]), axis=0)
        tmp2 = np.expand_dims(np.array([x2, 1]), axis=0)
        if i == 0:
            vectors = tmp1
            vectors = np.append(vectors, tmp2, axis=0)
        else:
            vectors = np.append(vectors, tmp1, axis=0)
            vectors = np.append(vectors, tmp2, axis=0)

    trainX = np.expand_dims(vectors[:, 0], axis=1)
    trainY = np.expand_dims(vectors[:, 1], axis=1)
    return trainX, trainY.astype(int)

def generate_data_for_multi_class_classification(seed=0, scaling = False):
    digits = load_digits()
    trainX = digits['data']
    if scaling == True:
        trainX,_ ,_ = MinMaxScaler(trainX)
    trainY = digits['target']
    np.random.seed(seed)
    mask = np.random.permutation(len(trainX))
    return trainX[mask], np.expand_dims(trainY[mask], axis=1)

#from UCI data set
def load_pendigits(seed = 0, scaling = False):

    pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
    pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')

    pendigits_data = np.append(pendigits_train, pendigits_test, axis = 0)
    nsamples = np.size(pendigits_data, 0)

    np.random.seed(seed)
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

    if scaling == True:
        x_train, train_min_col, train_max_col = MinMaxScaler(x_train)
        x_validation = MinMaxScaler_with(x_validation, train_min_col, train_max_col)
        x_test = MinMaxScaler_with(x_test, train_min_col, train_max_col)

    y_train = y_data[:ntrain]
    y_validation = y_data[ntrain:(ntrain+nvalidation)]
    y_test = y_data[-ntest:]

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def load_mnist(save_path, seed = 0, as_image = False, scaling = False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        if not os.path.exists(save_path+file_name):
            print("\n>>> Download "+file_name+" : ")
            file_path, _ = urllib.request.urlretrieve(url=data_url+file_name, filename = save_path+file_name, reporthook=print_download_progress)
        else:
            print(">>> {} data has apparently already been downloaded".format(file_name))


    with gzip.open(save_path+'train-images-idx3-ubyte.gz') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * 60000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data
        if as_image == True:
            x_train = data.reshape(60000, 28, 28, 1)
        else:
            x_train = data.reshape(60000, 784)

    with gzip.open(save_path+'train-labels-idx1-ubyte.gz') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(60000)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data
        y_train = np.expand_dims(data, 1)

    with gzip.open(save_path+'t10k-images-idx3-ubyte.gz') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * 10000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data
        if as_image == True:
            x_test = data.reshape(10000, 28, 28, 1)
        else:
            x_test = data.reshape(10000, 784)

    with gzip.open(save_path+'t10k-labels-idx1-ubyte.gz') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(10000)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data
        y_test = np.expand_dims(data, 1)

    np.random.seed(seed)
    mask = np.random.permutation(len(x_train))
    x_train = x_train[mask]
    y_train = y_train[mask]

    ntrain = int(len(x_train)*0.9)
    nvalidation = len(x_train)-ntrain
    x_validation = x_train[:nvalidation]
    y_validation = y_train[:nvalidation]
    x_train = x_train[nvalidation:]
    y_train = y_train[nvalidation:]

    if scaling == True:
        return x_train/255., x_validation/255., x_test/255., y_train, y_validation, y_test
    else:
        return x_train, x_validation, x_test, y_train, y_validation, y_test

def load_cifar(save_path, seed = 0, as_image = False, scaling = False):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split('/')[-1]
    file_path = os.path.join(save_path, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("\n>>> Download start")
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=print_download_progress)
        print("\n>>> Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(save_path)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(save_path)
        print("\n>>> Done.")
    else:
        print(">>> Data has apparently already been downloaded and unpacked.")

    name = 'data_batch_1'
    batch = unpickle(save_path+"cifar-10-batches-py/{}".format(name))
    x_train = batch[b'data']
    y_train = batch[b'labels']
    for name in ['data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
        batch = unpickle(save_path+"cifar-10-batches-py/{}".format(name))
        x_train = np.append(x_train, batch[b'data'], axis=0)
        y_train = np.append(y_train, batch[b'labels'], axis=0)

    test = unpickle(save_path+"cifar-10-batches-py/test_batch")
    x_test = test[b'data']
    y_test = test[b'labels']

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    if as_image == True:
        x_train = x_train.reshape(len(x_train), 3, 32, 32).transpose(0, 2, 3, 1).astype(float)
        x_test = x_test.reshape(len(x_test), 3, 32, 32).transpose(0, 2, 3, 1).astype(float)

    np.random.seed(seed)
    mask = np.random.permutation(len(x_train))
    x_train = x_train[mask]
    y_train = y_train[mask]

    ntrain = int(len(x_train) * 0.9)
    nvalidation = int(len(x_train) - ntrain)
    x_validation = x_train[:nvalidation]
    y_validation = y_train[:nvalidation]
    x_train = x_train[nvalidation:]
    y_train = y_train[nvalidation:]

    if scaling == True:
        return x_train/255., x_validation/255., x_test/255., y_train, y_validation, y_test
    else:
        return x_train, x_validation, x_test, y_train, y_validation, y_test
