import load_data
import plot_help

x_train, x_validation, x_test, y_train, y_validation, y_test=load_data.load_cifar('./data/cifar/', seed=0, as_image=True, scaling=True)
plot_help.plot_cifar(x_train, 100)

x_train, x_validation, x_test, y_train, y_validation, y_test=load_data.load_mnist('./data/cmnist/', seed=0, as_image=True, scaling=True)
plot_help.plot_mnist(x_train, 100)