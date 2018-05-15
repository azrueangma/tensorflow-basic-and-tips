import numpy as np

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
