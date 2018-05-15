import tensorflow as tf
import numpy as np
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 1000

dataX, dataY = load_data.generate_data_for_two_class_classification(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

W = tf.Variable(tf.truncated_normal(shape = [1]), name = 'W')
b = tf.Variable(tf.zeros([1]), name = 'b')
logits = tf.nn.bias_add(tf.multiply(X, W), b, name = 'logits')

hypothesis = tf.nn.sigmoid(logits, name = 'hypothesis')
loss = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.cast(hypothesis>0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %100 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [100/1000], loss = 0.551494, accuracy = 75.80%
Epoch [200/1000], loss = 0.541093, accuracy = 77.40%
Epoch [300/1000], loss = 0.531392, accuracy = 78.05%
Epoch [400/1000], loss = 0.522310, accuracy = 78.95%
Epoch [500/1000], loss = 0.513776, accuracy = 79.60%
Epoch [600/1000], loss = 0.505731, accuracy = 80.25%
Epoch [700/1000], loss = 0.498122, accuracy = 80.90%
Epoch [800/1000], loss = 0.490904, accuracy = 81.45%
Epoch [900/1000], loss = 0.484038, accuracy = 81.85%
Epoch [1000/1000], loss = 0.477491, accuracy = 82.45%
'''
