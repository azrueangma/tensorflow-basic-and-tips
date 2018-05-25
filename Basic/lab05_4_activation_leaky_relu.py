import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-4_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def leaky_relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = leaky_relu_layer(X, 256, 'Relu_Layer1')
h2 = leaky_relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_train_loss = tf.placeholder(tf.float32)
    loss_train_avg  = tf.summary.scalar('avg_train_loss', avg_train_loss)
    avg_train_acc = tf.placeholder(tf.float32)
    acc_train_avg = tf.summary.scalar('avg_train_acc', avg_train_acc)
    avg_validation_loss = tf.placeholder(tf.float32)
    loss_validation_avg = tf.summary.scalar('avg_validation_loss', avg_validation_loss)
    avg_validation_acc = tf.placeholder(tf.float32)
    acc_validation_avg = tf.summary.scalar('avg_validation_acc', avg_validation_acc)
    merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
total_step = int(ntrain/BATCH_SIZE)
print("Total step : ", total_step)
with tf.Session() as sess:
    if os.path.exists(BOARD_PATH):
        shutil.rmtree(BOARD_PATH)
    writer = tf.summary.FileWriter(BOARD_PATH)
    writer.add_graph(sess.graph)

    sess.run(init_op)

    train_start_time = time.perf_counter()
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))

        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:]})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*BATCH_SIZE
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict = {avg_train_loss:loss_per_epoch, avg_train_acc:acc_per_epoch,
                                          avg_validation_loss:epoch_valid_loss, avg_validation_acc:epoch_valid_acc})
        writer.add_summary(s, global_step = epoch)

        if (epoch+1) % 1 == 0:

            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 16.273132, train accuracy = 87.25%, valid loss = 3.871758, valid accuracy = 90.57%, duration = 2.363588(s)
Epoch [ 2/30], train loss = 2.863374, train accuracy = 91.98%, valid loss = 2.720269, valid accuracy = 92.10%, duration = 2.034916(s)
Epoch [ 3/30], train loss = 1.789759, train accuracy = 93.37%, valid loss = 1.950694, valid accuracy = 93.28%, duration = 2.033834(s)
Epoch [ 4/30], train loss = 1.320178, train accuracy = 94.33%, valid loss = 1.824393, valid accuracy = 93.33%, duration = 2.036113(s)
Epoch [ 5/30], train loss = 1.028459, train accuracy = 95.02%, valid loss = 1.611705, valid accuracy = 93.75%, duration = 2.058440(s)
Epoch [ 6/30], train loss = 0.831608, train accuracy = 95.56%, valid loss = 1.415752, valid accuracy = 93.92%, duration = 2.006404(s)
Epoch [ 7/30], train loss = 0.712924, train accuracy = 95.85%, valid loss = 1.444793, valid accuracy = 94.10%, duration = 2.046293(s)
Epoch [ 8/30], train loss = 0.598149, train accuracy = 96.34%, valid loss = 1.430695, valid accuracy = 94.38%, duration = 2.027685(s)
Epoch [ 9/30], train loss = 0.497383, train accuracy = 96.76%, valid loss = 1.255169, valid accuracy = 94.97%, duration = 2.011787(s)
Epoch [10/30], train loss = 0.425280, train accuracy = 97.02%, valid loss = 1.333563, valid accuracy = 94.80%, duration = 2.050958(s)
Epoch [11/30], train loss = 0.386088, train accuracy = 97.28%, valid loss = 1.463395, valid accuracy = 94.40%, duration = 2.039984(s)
Epoch [12/30], train loss = 0.329818, train accuracy = 97.54%, valid loss = 1.391823, valid accuracy = 94.93%, duration = 2.022179(s)
Epoch [13/30], train loss = 0.285365, train accuracy = 97.80%, valid loss = 1.270883, valid accuracy = 95.18%, duration = 2.033683(s)
Epoch [14/30], train loss = 0.256443, train accuracy = 98.02%, valid loss = 1.342359, valid accuracy = 94.43%, duration = 2.027458(s)
Epoch [15/30], train loss = 0.226924, train accuracy = 98.12%, valid loss = 1.346266, valid accuracy = 94.95%, duration = 2.073690(s)
Epoch [16/30], train loss = 0.192833, train accuracy = 98.36%, valid loss = 1.422907, valid accuracy = 94.87%, duration = 2.027308(s)
Epoch [17/30], train loss = 0.166421, train accuracy = 98.50%, valid loss = 1.435361, valid accuracy = 95.12%, duration = 2.061402(s)
Epoch [18/30], train loss = 0.153870, train accuracy = 98.59%, valid loss = 1.442440, valid accuracy = 94.97%, duration = 2.019589(s)
Epoch [19/30], train loss = 0.128646, train accuracy = 98.77%, valid loss = 1.314976, valid accuracy = 95.20%, duration = 2.166265(s)
Epoch [20/30], train loss = 0.117530, train accuracy = 98.81%, valid loss = 1.288233, valid accuracy = 95.63%, duration = 2.040300(s)
Epoch [21/30], train loss = 0.101439, train accuracy = 99.03%, valid loss = 1.259576, valid accuracy = 95.87%, duration = 2.044055(s)
Epoch [22/30], train loss = 0.093889, train accuracy = 99.08%, valid loss = 1.473838, valid accuracy = 95.25%, duration = 2.086285(s)
Epoch [23/30], train loss = 0.077476, train accuracy = 99.14%, valid loss = 1.323489, valid accuracy = 95.78%, duration = 2.120846(s)
Epoch [24/30], train loss = 0.061683, train accuracy = 99.29%, valid loss = 1.334951, valid accuracy = 95.62%, duration = 2.077543(s)
Epoch [25/30], train loss = 0.060455, train accuracy = 99.31%, valid loss = 1.332632, valid accuracy = 95.62%, duration = 2.079583(s)
Epoch [26/30], train loss = 0.049065, train accuracy = 99.41%, valid loss = 1.382443, valid accuracy = 95.55%, duration = 2.282994(s)
Epoch [27/30], train loss = 0.047726, train accuracy = 99.41%, valid loss = 1.380660, valid accuracy = 95.55%, duration = 2.083464(s)
Epoch [28/30], train loss = 0.049590, train accuracy = 99.42%, valid loss = 1.376996, valid accuracy = 95.77%, duration = 2.037819(s)
Epoch [29/30], train loss = 0.045624, train accuracy = 99.41%, valid loss = 1.514445, valid accuracy = 95.58%, duration = 2.069417(s)
Epoch [30/30], train loss = 0.037925, train accuracy = 99.51%, valid loss = 1.436260, valid accuracy = 95.43%, duration = 2.012338(s)
Duration for train : 63.165946(s)
<<< Train Finished >>>
Test Accraucy : 95.28%
'''
