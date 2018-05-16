import numpy as np
from sklearn.datasets import load_digits

def MinMaxScale(x):
    col_min = np.min(x, axis = 0)
    col_max = np.max(x, axis = 0)
    denominator = (col_max - col_min) + 1e-7
    numerator = x - col_min
    return numerator/denominator

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
        trainX = MinMaxScale(trainX)
    trainY = digits['target']
    np.random.seed(seed)
    mask = np.random.permutation(len(trainX))
    return trainX[mask], np.expand_dims(trainY[mask], axis=1)
