import numpy as np
import matplotlib.pyplot as plt

def plot_cifar(images, n_images, seed=0):
    plt.figure()
    plt.gca().set_axis_off()
    h_num = int(np.sqrt(n_images))
    v_num = int(np.sqrt(n_images))
    v_list = []
    np.random.seed(seed)
    mask = np.random.permutation(len(images))
    count = 0
    for j in range(v_num):
        h_list = []
        for i in range(h_num):
            h_list.append(images[mask[count]])
            count += 1
        tmp = np.hstack(h_list)
        v_list.append(tmp)
    im = np.vstack(v_list)
    plt.imshow(im)
    plt.show()

def plot_mnist(images, n_images, seed=0):
    images = np.reshape(images, [len(images), 28, 28])
    plt.figure()
    plt.gca().set_axis_off()
    h_num = int(np.sqrt(n_images))
    v_num = int(np.sqrt(n_images))
    v_list = []
    np.random.seed(seed)
    mask = np.random.permutation(len(images))
    count = 0
    for j in range(v_num):
        h_list = []
        for i in range(h_num):
            h_list.append(images[mask[count]])
            count+=1
        tmp = np.hstack(h_list)
        v_list.append(tmp)
    im = np.vstack(v_list)
    plt.imshow(im, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
