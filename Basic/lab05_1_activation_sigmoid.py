import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-1_board"
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

def sigmoid_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = sigmoid_layer(X, 256, 'Sigmoid_Layer1')
h2 = sigmoid_layer(h1, 128, 'Sigmoid_Layer2')
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
Epoch [ 1/30], train loss = 1.455614, train accuracy = 60.25%, valid loss = 0.776444, valid accuracy = 75.80%, duration = 2.032732(s)
Epoch [ 2/30], train loss = 0.688446, train accuracy = 78.79%, valid loss = 0.586213, valid accuracy = 81.75%, duration = 2.263734(s)
Epoch [ 3/30], train loss = 0.546542, train accuracy = 83.29%, valid loss = 0.505002, valid accuracy = 84.60%, duration = 1.972940(s)
Epoch [ 4/30], train loss = 0.473918, train accuracy = 85.47%, valid loss = 0.459786, valid accuracy = 85.80%, duration = 2.407496(s)
Epoch [ 5/30], train loss = 0.426833, train accuracy = 86.92%, valid loss = 0.422621, valid accuracy = 86.90%, duration = 1.985831(s)
Epoch [ 6/30], train loss = 0.392656, train accuracy = 87.90%, valid loss = 0.398211, valid accuracy = 87.57%, duration = 1.937322(s)
Epoch [ 7/30], train loss = 0.365426, train accuracy = 88.74%, valid loss = 0.378575, valid accuracy = 88.18%, duration = 1.965230(s)
Epoch [ 8/30], train loss = 0.344011, train accuracy = 89.44%, valid loss = 0.360172, valid accuracy = 88.63%, duration = 1.953900(s)
Epoch [ 9/30], train loss = 0.325258, train accuracy = 90.11%, valid loss = 0.353109, valid accuracy = 89.25%, duration = 2.129417(s)
Epoch [10/30], train loss = 0.309857, train accuracy = 90.62%, valid loss = 0.334715, valid accuracy = 89.72%, duration = 2.100878(s)
Epoch [11/30], train loss = 0.296122, train accuracy = 91.13%, valid loss = 0.324071, valid accuracy = 90.30%, duration = 1.944086(s)
Epoch [12/30], train loss = 0.284089, train accuracy = 91.44%, valid loss = 0.313126, valid accuracy = 90.43%, duration = 2.056585(s)
Epoch [13/30], train loss = 0.273076, train accuracy = 91.74%, valid loss = 0.307189, valid accuracy = 90.55%, duration = 1.930997(s)
Epoch [14/30], train loss = 0.263279, train accuracy = 92.07%, valid loss = 0.301682, valid accuracy = 90.70%, duration = 1.957559(s)
Epoch [15/30], train loss = 0.254198, train accuracy = 92.46%, valid loss = 0.292010, valid accuracy = 91.08%, duration = 2.482160(s)
Epoch [16/30], train loss = 0.245974, train accuracy = 92.68%, valid loss = 0.286373, valid accuracy = 91.37%, duration = 2.029369(s)
Epoch [17/30], train loss = 0.238260, train accuracy = 92.93%, valid loss = 0.280072, valid accuracy = 91.45%, duration = 2.003680(s)
Epoch [18/30], train loss = 0.231257, train accuracy = 93.10%, valid loss = 0.275361, valid accuracy = 91.63%, duration = 2.120293(s)
Epoch [19/30], train loss = 0.224539, train accuracy = 93.35%, valid loss = 0.270357, valid accuracy = 91.83%, duration = 2.014512(s)
Epoch [20/30], train loss = 0.218197, train accuracy = 93.49%, valid loss = 0.266236, valid accuracy = 92.02%, duration = 2.158000(s)
Epoch [21/30], train loss = 0.212187, train accuracy = 93.74%, valid loss = 0.263386, valid accuracy = 92.12%, duration = 2.073445(s)
Epoch [22/30], train loss = 0.206755, train accuracy = 93.87%, valid loss = 0.258818, valid accuracy = 92.18%, duration = 1.937507(s)
Epoch [23/30], train loss = 0.201644, train accuracy = 93.96%, valid loss = 0.254443, valid accuracy = 92.17%, duration = 2.151242(s)
Epoch [24/30], train loss = 0.196420, train accuracy = 94.17%, valid loss = 0.251680, valid accuracy = 92.27%, duration = 1.915977(s)
Epoch [25/30], train loss = 0.191328, train accuracy = 94.37%, valid loss = 0.249987, valid accuracy = 92.50%, duration = 2.019200(s)
Epoch [26/30], train loss = 0.186990, train accuracy = 94.45%, valid loss = 0.245609, valid accuracy = 92.40%, duration = 2.091873(s)
Epoch [27/30], train loss = 0.182652, train accuracy = 94.62%, valid loss = 0.243491, valid accuracy = 92.67%, duration = 1.979157(s)
Epoch [28/30], train loss = 0.178289, train accuracy = 94.70%, valid loss = 0.239601, valid accuracy = 92.75%, duration = 2.061276(s)
Epoch [29/30], train loss = 0.174228, train accuracy = 94.82%, valid loss = 0.237757, valid accuracy = 92.68%, duration = 2.098268(s)
Epoch [30/30], train loss = 0.170505, train accuracy = 94.87%, valid loss = 0.236409, valid accuracy = 92.92%, duration = 1.933548(s)
Duration for train : 62.695116(s)
<<< Train Finished >>>
Test Accraucy : 92.47%
'''
