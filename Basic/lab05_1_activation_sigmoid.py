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
Epoch [ 1/30], train loss = 1.468763, train accuracy = 59.97%, valid loss = 0.786915, valid accuracy = 75.32%, duration = 1.936174(s)
Epoch [ 2/30], train loss = 0.694213, train accuracy = 78.43%, valid loss = 0.585209, valid accuracy = 81.97%, duration = 1.815053(s)
Epoch [ 3/30], train loss = 0.554398, train accuracy = 83.03%, valid loss = 0.499855, valid accuracy = 84.60%, duration = 1.849631(s)
Epoch [ 4/30], train loss = 0.479633, train accuracy = 85.36%, valid loss = 0.447635, valid accuracy = 85.98%, duration = 1.833987(s)
Epoch [ 5/30], train loss = 0.430357, train accuracy = 86.89%, valid loss = 0.410011, valid accuracy = 87.30%, duration = 1.826692(s)
Epoch [ 6/30], train loss = 0.393892, train accuracy = 88.00%, valid loss = 0.385743, valid accuracy = 88.18%, duration = 1.829732(s)
Epoch [ 7/30], train loss = 0.365261, train accuracy = 88.84%, valid loss = 0.368573, valid accuracy = 88.80%, duration = 1.840367(s)
Epoch [ 8/30], train loss = 0.341972, train accuracy = 89.62%, valid loss = 0.347889, valid accuracy = 89.68%, duration = 1.832383(s)
Epoch [ 9/30], train loss = 0.322265, train accuracy = 90.31%, valid loss = 0.339569, valid accuracy = 89.60%, duration = 1.785304(s)
Epoch [10/30], train loss = 0.305438, train accuracy = 90.83%, valid loss = 0.322817, valid accuracy = 90.42%, duration = 1.812493(s)
Epoch [11/30], train loss = 0.290789, train accuracy = 91.29%, valid loss = 0.312144, valid accuracy = 90.63%, duration = 1.798330(s)
Epoch [12/30], train loss = 0.277987, train accuracy = 91.70%, valid loss = 0.303720, valid accuracy = 90.95%, duration = 1.816442(s)
Epoch [13/30], train loss = 0.266254, train accuracy = 92.08%, valid loss = 0.295645, valid accuracy = 91.17%, duration = 1.801357(s)
Epoch [14/30], train loss = 0.255860, train accuracy = 92.38%, valid loss = 0.289301, valid accuracy = 91.20%, duration = 1.827973(s)
Epoch [15/30], train loss = 0.245925, train accuracy = 92.68%, valid loss = 0.287328, valid accuracy = 91.18%, duration = 1.797828(s)
Epoch [16/30], train loss = 0.237345, train accuracy = 92.86%, valid loss = 0.278883, valid accuracy = 91.43%, duration = 1.841519(s)
Epoch [17/30], train loss = 0.229075, train accuracy = 93.09%, valid loss = 0.273355, valid accuracy = 91.68%, duration = 1.839130(s)
Epoch [18/30], train loss = 0.221721, train accuracy = 93.32%, valid loss = 0.267827, valid accuracy = 91.77%, duration = 1.844335(s)
Epoch [19/30], train loss = 0.214790, train accuracy = 93.60%, valid loss = 0.262406, valid accuracy = 91.98%, duration = 1.838717(s)
Epoch [20/30], train loss = 0.208023, train accuracy = 93.78%, valid loss = 0.260535, valid accuracy = 91.90%, duration = 1.832736(s)
Epoch [21/30], train loss = 0.201911, train accuracy = 94.01%, valid loss = 0.255686, valid accuracy = 92.22%, duration = 1.787104(s)
Epoch [22/30], train loss = 0.196105, train accuracy = 94.19%, valid loss = 0.250848, valid accuracy = 92.37%, duration = 1.855690(s)
Epoch [23/30], train loss = 0.190459, train accuracy = 94.32%, valid loss = 0.250170, valid accuracy = 92.37%, duration = 1.801637(s)
Epoch [24/30], train loss = 0.185452, train accuracy = 94.49%, valid loss = 0.245272, valid accuracy = 92.38%, duration = 1.837872(s)
Epoch [25/30], train loss = 0.180255, train accuracy = 94.67%, valid loss = 0.243287, valid accuracy = 92.58%, duration = 1.837355(s)
Epoch [26/30], train loss = 0.175626, train accuracy = 94.82%, valid loss = 0.240479, valid accuracy = 92.63%, duration = 1.834492(s)
Epoch [27/30], train loss = 0.171150, train accuracy = 94.96%, valid loss = 0.239439, valid accuracy = 92.73%, duration = 1.814794(s)
Epoch [28/30], train loss = 0.166930, train accuracy = 95.09%, valid loss = 0.236437, valid accuracy = 92.75%, duration = 1.820010(s)
Epoch [29/30], train loss = 0.162703, train accuracy = 95.19%, valid loss = 0.233887, valid accuracy = 92.93%, duration = 1.770262(s)
Epoch [30/30], train loss = 0.158831, train accuracy = 95.36%, valid loss = 0.233620, valid accuracy = 92.75%, duration = 1.830121(s)
Duration for train : 55.719756(s)
<<< Train Finished >>>
Test Accraucy : 92.67%
'''
