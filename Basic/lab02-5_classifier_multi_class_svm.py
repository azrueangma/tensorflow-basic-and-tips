import tensorflow as tf
import numpy as np
import load_data

SEED = 0
dataX, dataY = load_data.generate_data_for_multi_class_classification(seed = SEED, scaling = True)

NSAMPLES = np.size(dataX, 0)
INPUT_DIM = np.size(dataX,1)
NCLASS = len(np.unique(dataY))
TOTAL_EPOCH = 10000

C = 10.0

print("The number of data samples : ", NSAMPLES)
print("The dimension of data samples : ", INPUT_DIM)

def linear(x, output_dim, with_W, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        if with_W == True:
            return h, W
        else:
            return h

def sigmoid_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')
Y_svm_target = tf.subtract(tf.multiply(Y_one_hot, 2.), 1., 'Y_svm_target')

h1 = sigmoid_linear(X, 128, 'FC_Layer1')
h2 = sigmoid_linear(h1, 256, 'FC_Layer2')
logits, W = linear(h2, NCLASS, with_W = True, name = 'FC_Layer3')

l2_norm = tf.reduce_sum(tf.square(W),axis=0, name = 'l2_norm')
const = tf.constant([C],name="c")
tmp =  tf.subtract(1., tf.multiply(logits, Y_svm_target))
hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros_like(tmp),tmp)),axis=0, name = 'l2_hinge_loss')
loss = tf.reduce_mean(tf.add(tf.multiply(const,hinge_loss),0.5*l2_norm))/tf.cast(tf.shape(logits)[0], tf.float32, 'loss')
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_svm_target, axis = 1)), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %1000 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [1000/10000], loss = 2.958015, accuracy = 74.79%
Epoch [2000/10000], loss = 1.765386, accuracy = 84.42%
Epoch [3000/10000], loss = 1.305481, accuracy = 88.54%
Epoch [4000/10000], loss = 1.047895, accuracy = 90.65%
Epoch [5000/10000], loss = 0.879552, accuracy = 92.54%
Epoch [6000/10000], loss = 0.759388, accuracy = 93.71%
Epoch [7000/10000], loss = 0.668932, accuracy = 94.49%
Epoch [8000/10000], loss = 0.598465, accuracy = 95.44%
Epoch [9000/10000], loss = 0.541956, accuracy = 95.99%
Epoch [10000/10000], loss = 0.495593, accuracy = 96.38%
'''