import numpy as np
import time

pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')

np.save('./data/pendigits_train', pendigits_train)
np.save('./data/pendigits_test', pendigits_test)

csv_load_start_time = time.perf_counter()
pendigits_train = np.loadtxt('./data/pendigits_train.csv', delimiter = ',')
pendigits_test = np.loadtxt('./data/pendigits_test.csv', delimiter = ',')
csv_load_end_time = time.perf_counter()
csv_load_duration = csv_load_end_time - csv_load_start_time
print('csv data loading time : {:.6f}(s)'.format(csv_load_duration))

npy_load_start_time = time.perf_counter()
pendigits_train = np.load('./data/pendigits_train.npy')
pendigits_test = np.load('./data/pendigits_test.npy')
npy_load_end_time = time.perf_counter()
npy_load_duration = npy_load_end_time - npy_load_start_time
print('npy data loading time : {:.6f}(s)'.format(npy_load_duration))

'''
csv data loading time : 0.132077(s)
npy data loading time : 0.001667(s)
'''