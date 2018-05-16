import tensorflow as tf
import numpy as np
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 10000
NCLASS = 2
C = 10.0

dataX, dataY = load_data.generate_data_for_two_class_classification(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')
Y_svm_target = tf.subtract(tf.multiply(Y_one_hot, 2.), 1., 'Y_svm_target')

W = tf.Variable(tf.truncated_normal(shape = [1, 2]), name = 'W')
b = tf.Variable(tf.zeros([2]), name = 'b')
logits = tf.nn.bias_add(tf.multiply(X, W), b, name = 'logits')

l2_norm = tf.reduce_sum(tf.square(W),axis=0, name = 'l2_norm')
const = tf.constant([C],name="c")
tmp =  tf.subtract(1., tf.multiply(logits, Y_svm_target))
hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros_like(tmp),tmp)),axis=0, name = 'l2_hinge_loss')
loss = tf.reduce_mean(tf.add(tf.multiply(const,hinge_loss),l2_norm))/tf.cast(tf.shape(logits)[0], tf.float32, 'loss')
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_svm_target, axis = 1)), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    arg_ = 0
    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %1000 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [1000/10000], loss = 1.064117, accuracy = 98.05%
Epoch [2000/10000], loss = 0.890877, accuracy = 97.95%
Epoch [3000/10000], loss = 0.822733, accuracy = 98.10%
Epoch [4000/10000], loss = 0.784905, accuracy = 98.15%
Epoch [5000/10000], loss = 0.760539, accuracy = 98.15%
Epoch [6000/10000], loss = 0.743657, accuracy = 98.15%
Epoch [7000/10000], loss = 0.731406, accuracy = 98.15%
Epoch [8000/10000], loss = 0.722229, accuracy = 98.15%
Epoch [9000/10000], loss = 0.715138, accuracy = 98.05%
Epoch [10000/10000], loss = 0.709498, accuracy = 98.05%
'''