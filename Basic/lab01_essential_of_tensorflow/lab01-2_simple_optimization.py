#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#constant
TOTAL_EPOCH = 100

#load data
x_train = np.array([[3.5], [4.2], [3.8], [5.0], [5.1], [0.2], [1.3], [2.4], [1.5], [0.5]])
y_train = np.array([[3.3], [4.4], [4.0], [4.8], [5.6], [0.1], [1.2], [2.4], [1.7], [0.6]])

g = tf.Graph()


with g.as_default():
    tf.set_random_seed(0)

    #create placeholder for input
    X = tf.placeholder(shape=[None,1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name='Y')

    #weigth and bias
    W = tf.Variable(tf.truncated_normal(shape=[1]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')
    output = tf.nn.bias_add(tf.multiply(X, W), b, name='output')

    loss = tf.reduce_mean(tf.square(Y-output), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)


    init_op = tf.global_variables_initializer()
    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        for epoch in range(TOTAL_EPOCH):
            l, W_val, b_val, _ = sess.run([loss, W, b, train], feed_dict={X:x_train, Y:y_train})
            if (epoch+1)%10 == 0:
                print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, TOTAL_EPOCH, l))


'''
Epoch [ 10/100], loss = 3.577132
Epoch [ 20/100], loss = 2.286805
Epoch [ 30/100], loss = 1.468878
Epoch [ 40/100], loss = 0.950385
Epoch [ 50/100], loss = 0.621690
Epoch [ 60/100], loss = 0.413300
Epoch [ 70/100], loss = 0.281166
Epoch [ 80/100], loss = 0.197369
Epoch [ 90/100], loss = 0.144210
Epoch [100/100], loss = 0.110473
'''
