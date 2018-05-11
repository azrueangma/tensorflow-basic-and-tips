import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import basic_utils
import shutil
import os

def create_weight_variable(shape):
    W = tf.get_variable(name = 'W', shape = shape, dtype = tf.float32, initializer= tf.truncated_normal_initializer())
    b = tf.get_variable(name = 'b', shape = shape, dtype = tf.float32, initializer= tf.constant_initializer(0.0))
    return W, b

NPOINTS = 1000
TOTAL_EPOCH = 1000

dataX, dataY = basic_utils.generate_data_for_linear_regression(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

W, b = create_weight_variable([1])
hypothesis = tf.nn.bias_add(tf.multiply(X, W), b, name = 'hypothesis')

loss = tf.reduce_mean(tf.square(Y-hypothesis), name = 'loss')
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
grads = optim.compute_gradients(loss)
apply_gradients = optim.apply_gradients(grads)
#it is equal to tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
grads_and_vars_list = [[_grad, _var] for _grad, _var in grads]

'''
for grads_and_vars in grads:
    print(grads_and_vars)
    
(<tf.Tensor 'gradients/Mul_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>, <tf.Variable 'W:0' shape=(1,) dtype=float32_ref>)
(<tf.Tensor 'gradients/hypothesis_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>, <tf.Variable 'b:0' shape=(1,) dtype=float32_ref>)
'''

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(TOTAL_EPOCH):
        l, _ = sess.run([loss, apply_gradients], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %100 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, TOTAL_EPOCH, l))

'''
Epoch [100/1000], loss = 0.174033
Epoch [200/1000], loss = 0.118036
Epoch [300/1000], loss = 0.080456
Epoch [400/1000], loss = 0.055220
Epoch [500/1000], loss = 0.038261
Epoch [600/1000], loss = 0.026852
Epoch [700/1000], loss = 0.019166
Epoch [800/1000], loss = 0.013979
Epoch [900/1000], loss = 0.010470
Epoch [1000/1000], loss = 0.008087
'''