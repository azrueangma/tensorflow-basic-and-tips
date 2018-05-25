import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-2_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 20

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

def tanh_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = tanh_layer(X, 256, 'Tanh_Layer1')
h2 = tanh_layer(h1, 128, 'Tanh_Layer2')
h3 = tanh_layer(h2, 64, 'Tanh_Layer3')
logits = linear(h3, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

with tf.variable_scope("Pred_and_Acc"):
    predict = tf.argmax(hypothesis, axis=1)
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
Total step :  1687
Epoch [ 1/20], train loss = 2.346969, train accuracy = 52.25%, valid loss = 1.155028, valid accuracy = 66.27%, duration = 2.348864(s)
Epoch [ 2/20], train loss = 0.955150, train accuracy = 71.18%, valid loss = 0.830513, valid accuracy = 74.83%, duration = 1.987080(s)
Epoch [ 3/20], train loss = 0.743260, train accuracy = 76.74%, valid loss = 0.701448, valid accuracy = 77.98%, duration = 2.095097(s)
Epoch [ 4/20], train loss = 0.632374, train accuracy = 79.86%, valid loss = 0.634420, valid accuracy = 79.78%, duration = 1.902889(s)
Epoch [ 5/20], train loss = 0.557483, train accuracy = 82.42%, valid loss = 0.578836, valid accuracy = 81.58%, duration = 2.035855(s)
Epoch [ 6/20], train loss = 0.499881, train accuracy = 84.28%, valid loss = 0.546664, valid accuracy = 83.58%, duration = 1.907409(s)
Epoch [ 7/20], train loss = 0.450979, train accuracy = 85.75%, valid loss = 0.539891, valid accuracy = 83.62%, duration = 2.061956(s)
Epoch [ 8/20], train loss = 0.411023, train accuracy = 87.04%, valid loss = 0.514006, valid accuracy = 84.40%, duration = 1.868290(s)
Epoch [ 9/20], train loss = 0.373608, train accuracy = 88.23%, valid loss = 0.510061, valid accuracy = 84.68%, duration = 2.074502(s)
Epoch [10/20], train loss = 0.340424, train accuracy = 89.35%, valid loss = 0.502445, valid accuracy = 85.02%, duration = 1.905600(s)
Epoch [11/20], train loss = 0.311776, train accuracy = 90.27%, valid loss = 0.496763, valid accuracy = 85.20%, duration = 2.028892(s)
Epoch [12/20], train loss = 0.286189, train accuracy = 91.18%, valid loss = 0.493713, valid accuracy = 85.32%, duration = 1.842779(s)
Epoch [13/20], train loss = 0.263770, train accuracy = 91.93%, valid loss = 0.488232, valid accuracy = 85.90%, duration = 1.828462(s)
Epoch [14/20], train loss = 0.242865, train accuracy = 92.65%, valid loss = 0.488236, valid accuracy = 86.12%, duration = 1.879122(s)
Epoch [15/20], train loss = 0.226410, train accuracy = 93.16%, valid loss = 0.491854, valid accuracy = 85.87%, duration = 1.889522(s)
Epoch [16/20], train loss = 0.211833, train accuracy = 93.79%, valid loss = 0.492490, valid accuracy = 85.78%, duration = 1.840889(s)
Epoch [17/20], train loss = 0.199241, train accuracy = 94.21%, valid loss = 0.492557, valid accuracy = 86.00%, duration = 1.833272(s)
Epoch [18/20], train loss = 0.188022, train accuracy = 94.55%, valid loss = 0.493061, valid accuracy = 86.45%, duration = 2.041234(s)
Epoch [19/20], train loss = 0.177905, train accuracy = 95.00%, valid loss = 0.491827, valid accuracy = 86.20%, duration = 2.026700(s)
Epoch [20/20], train loss = 0.168310, train accuracy = 95.32%, valid loss = 0.495253, valid accuracy = 86.37%, duration = 2.173393(s)
Duration for train : 40.209751(s)
<<< Train Finished >>>
Test Accraucy : 86.66%
'''
