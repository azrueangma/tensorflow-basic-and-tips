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

def tanh_linear(x, output_dim, name):
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

h1 = tanh_linear(X, 256, 'Tanh_Layer1')
h2 = tanh_linear(h1, 128, 'Tanh_Layer2')
h3 = tanh_linear(h2, 64, 'Tanh_Layer3')
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
        x_trian = x_train[mask]
        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[s:t,:], Y: y_train[s:t,:]})
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
Epoch [ 1/20], train loss = 2.344722, train accuracy = 53.54%, valid loss = 1.159778, valid accuracy = 67.48%, duration = 2.067302(s)
Epoch [ 2/20], train loss = 0.979846, train accuracy = 71.37%, valid loss = 0.817574, valid accuracy = 74.72%, duration = 2.154801(s)
Epoch [ 3/20], train loss = 0.737619, train accuracy = 77.15%, valid loss = 0.693694, valid accuracy = 78.60%, duration = 1.957235(s)
Epoch [ 4/20], train loss = 0.623200, train accuracy = 80.32%, valid loss = 0.622445, valid accuracy = 80.62%, duration = 1.884581(s)
Epoch [ 5/20], train loss = 0.542388, train accuracy = 82.84%, valid loss = 0.553055, valid accuracy = 82.87%, duration = 2.071323(s)
Epoch [ 6/20], train loss = 0.483368, train accuracy = 84.77%, valid loss = 0.530915, valid accuracy = 83.50%, duration = 1.911856(s)
Epoch [ 7/20], train loss = 0.434593, train accuracy = 86.22%, valid loss = 0.516984, valid accuracy = 84.33%, duration = 1.909308(s)
Epoch [ 8/20], train loss = 0.392079, train accuracy = 87.58%, valid loss = 0.506902, valid accuracy = 84.67%, duration = 1.982191(s)
Epoch [ 9/20], train loss = 0.351216, train accuracy = 88.92%, valid loss = 0.489587, valid accuracy = 85.18%, duration = 2.060633(s)
Epoch [10/20], train loss = 0.316297, train accuracy = 90.16%, valid loss = 0.486891, valid accuracy = 85.90%, duration = 1.976410(s)
Epoch [11/20], train loss = 0.285910, train accuracy = 91.05%, valid loss = 0.479933, valid accuracy = 85.90%, duration = 1.845755(s)
Epoch [12/20], train loss = 0.262413, train accuracy = 91.95%, valid loss = 0.482992, valid accuracy = 85.72%, duration = 1.895113(s)
Epoch [13/20], train loss = 0.242349, train accuracy = 92.76%, valid loss = 0.486971, valid accuracy = 85.88%, duration = 1.804062(s)
Epoch [14/20], train loss = 0.225493, train accuracy = 93.41%, valid loss = 0.491379, valid accuracy = 86.00%, duration = 1.935106(s)
Epoch [15/20], train loss = 0.212337, train accuracy = 93.86%, valid loss = 0.491805, valid accuracy = 86.33%, duration = 1.967194(s)
Epoch [16/20], train loss = 0.200878, train accuracy = 94.28%, valid loss = 0.493380, valid accuracy = 86.20%, duration = 1.969947(s)
Epoch [17/20], train loss = 0.189282, train accuracy = 94.66%, valid loss = 0.494319, valid accuracy = 86.37%, duration = 1.853639(s)
Epoch [18/20], train loss = 0.179103, train accuracy = 95.02%, valid loss = 0.495960, valid accuracy = 86.53%, duration = 1.928669(s)
Epoch [19/20], train loss = 0.170418, train accuracy = 95.25%, valid loss = 0.498185, valid accuracy = 86.42%, duration = 1.923597(s)
Epoch [20/20], train loss = 0.162639, train accuracy = 95.58%, valid loss = 0.501117, valid accuracy = 86.40%, duration = 2.040135(s)
Duration for train : 41.911079(s)
<<< Train Finished >>>
Test Accraucy : 86.11%
'''
