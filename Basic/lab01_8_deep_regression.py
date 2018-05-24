import tensorflow as tf
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 100

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def sigmoid_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

dataX, dataY = load_data.generate_data_for_linear_regression(NPOINTS)

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

h1 = sigmoid_linear(X, 5, 'FC_Layer1')
h2 = sigmoid_linear(h1, 10, 'FC_Layer2')
h3 = linear(h2, 1, 'FC_Layer3')
hypothesis = tf.identity(h3, name = 'hypothesis')

with tf.variable_scope('Optimization'):
    loss = tf.reduce_mean(tf.square(Y-hypothesis), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(TOTAL_EPOCH):
        l, _ = sess.run([loss,optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %10 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, TOTAL_EPOCH, l))

'''
Epoch [ 10/100], loss = 0.011773
Epoch [ 20/100], loss = 0.010828
Epoch [ 30/100], loss = 0.010030
Epoch [ 40/100], loss = 0.009357
Epoch [ 50/100], loss = 0.008789
Epoch [ 60/100], loss = 0.008309
Epoch [ 70/100], loss = 0.007905
Epoch [ 80/100], loss = 0.007563
Epoch [ 90/100], loss = 0.007275
Epoch [100/100], loss = 0.007031 
'''
