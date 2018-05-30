import numpy as np
import matplotlib.pyplot as plt

def plot_mnist(images, n_images):
    images = np.reshape(images, [len(images), 28, 28])
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n_images, replace=False)] for _ in range(10)]) for _ in range(10)])
    plt.imshow(im, cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.show()

def plot_cifar(images, n_images):
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n_images, replace=False)] for _ in range(10)]) for _ in range(10)])
    plt.imshow(im)
    plt.show()