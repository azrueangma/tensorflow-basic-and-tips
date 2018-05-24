import tensorflow as tf
import load_data
import shutil
import time
import os

NPOINTS = 100000
TOTAL_EPOCH = 100
CPU_DEVICE = 0
GPU_DEVICE = 0
BOARD_PATH = "./board/lab03-4_board"

def linear(x, output_dim, with_W, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, 
                            initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        if with_W:
            return h, W
        else:
            return h

def sigmoid_linear(x, output_dim, with_W, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, 
                            initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        if with_W:
            return h, W
        else:
            return h

tf.set_random_seed(0)
with tf.device('/gpu:{}'.format(GPU_DEVICE)):
    with tf.variable_scope("Inputs"):
        X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
        Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

    h1, W1 = sigmoid_linear(X, 5, True, 'FC_Layer1')
    h2, W2 = sigmoid_linear(h1, 10, True, 'FC_Layer2')
    h3, W3 = linear(h2, 1, True, 'FC_Layer3')
    hypothesis = tf.identity(h3, name = 'hypothesis')

    with tf.variable_scope('Optimization'):
        loss = tf.reduce_mean(tf.square(Y-hypothesis), name = 'loss')
        optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

    W1_hist = tf.summary.histogram("Weight1", W1)
    W2_hist = tf.summary.histogram("Weight2", W2)
    W3_hist = tf.summary.histogram("Weight3", W3)
    loss_scalar = tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

with tf.device('/cpu:{}'.format(CPU_DEVICE)):
    dataX, dataY = load_data.generate_data_for_linear_regression(NPOINTS)

    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        init_op = tf.global_variables_initializer()
        if os.path.exists(BOARD_PATH):
            shutil.rmtree(BOARD_PATH)
        writer = tf.summary.FileWriter(BOARD_PATH)
        writer.add_graph(sess.graph)
        sess.run(init_op)
        for epoch in range(TOTAL_EPOCH):
            train_start_time = time.perf_counter()
            m, l, _ = sess.run([merged, loss,optim], feed_dict={X: dataX, Y: dataY})
            train_end_time = time.perf_counter()
            train_duration = train_end_time - train_start_time
            writer.add_summary(m, global_step = epoch)
            if (epoch+1) %10 == 0:
                print("Epoch [{:3d}/{:3d}], loss = {:.6f}, train duration = {:.6f}(s)".format(epoch + 1, TOTAL_EPOCH, l, train_duration))
'''
#only  CPU
Epoch [ 10/100], loss = 0.012133, train duration = 0.010649(s)
Epoch [ 20/100], loss = 0.011171, train duration = 0.013035(s)
Epoch [ 30/100], loss = 0.010359, train duration = 0.010365(s)
Epoch [ 40/100], loss = 0.009674, train duration = 0.010358(s)
Epoch [ 50/100], loss = 0.009096, train duration = 0.010643(s)
Epoch [ 60/100], loss = 0.008608, train duration = 0.010377(s)
Epoch [ 70/100], loss = 0.008196, train duration = 0.010487(s)
Epoch [ 80/100], loss = 0.007848, train duration = 0.014178(s)
Epoch [ 90/100], loss = 0.007555, train duration = 0.010508(s)
Epoch [100/100], loss = 0.007307, train duration = 0.010370(s)

#CPU+GPU
Epoch [ 10/100], loss = 0.012133, train duration = 0.006319(s)
Epoch [ 20/100], loss = 0.011171, train duration = 0.006301(s)
Epoch [ 30/100], loss = 0.010359, train duration = 0.005700(s)
Epoch [ 40/100], loss = 0.009674, train duration = 0.005702(s)
Epoch [ 50/100], loss = 0.009096, train duration = 0.005643(s)
Epoch [ 60/100], loss = 0.008608, train duration = 0.006422(s)
Epoch [ 70/100], loss = 0.008196, train duration = 0.005611(s)
Epoch [ 80/100], loss = 0.007848, train duration = 0.005709(s)
Epoch [ 90/100], loss = 0.007555, train duration = 0.005681(s)
Epoch [100/100], loss = 0.007307, train duration = 0.005802(s)
'''


