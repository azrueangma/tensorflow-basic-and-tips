#lab04_4 model
import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-5_board"
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
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def relu_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_linear(X, 256, 'Relu_Layer1')
h2 = relu_linear(h1, 128, 'Relu_Layer2')
h3 = relu_linear(h2, 64, 'Relu_Layer3')
logits = linear(h1, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
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
Epoch [ 1/20], train loss = 0.424913, train accuracy = 88.66%, valid loss = 0.260437, valid accuracy = 92.50%, duration = 1.643205(s)
Epoch [ 2/20], train loss = 0.240101, train accuracy = 93.24%, valid loss = 0.199711, valid accuracy = 94.65%, duration = 1.482888(s)
Epoch [ 3/20], train loss = 0.189335, train accuracy = 94.73%, valid loss = 0.164547, valid accuracy = 95.43%, duration = 1.420391(s)
Epoch [ 4/20], train loss = 0.156605, train accuracy = 95.66%, valid loss = 0.141518, valid accuracy = 96.08%, duration = 1.521686(s)
Epoch [ 5/20], train loss = 0.133592, train accuracy = 96.28%, valid loss = 0.125231, valid accuracy = 96.47%, duration = 1.431490(s)
Epoch [ 6/20], train loss = 0.116397, train accuracy = 96.79%, valid loss = 0.113790, valid accuracy = 96.70%, duration = 1.633122(s)
Epoch [ 7/20], train loss = 0.103023, train accuracy = 97.20%, valid loss = 0.105314, valid accuracy = 96.88%, duration = 1.483426(s)
Epoch [ 8/20], train loss = 0.092357, train accuracy = 97.46%, valid loss = 0.098648, valid accuracy = 96.95%, duration = 1.403398(s)
Epoch [ 9/20], train loss = 0.083542, train accuracy = 97.74%, valid loss = 0.093187, valid accuracy = 97.12%, duration = 1.512071(s)
Epoch [10/20], train loss = 0.076138, train accuracy = 97.99%, valid loss = 0.088772, valid accuracy = 97.22%, duration = 1.404577(s)
Epoch [11/20], train loss = 0.069821, train accuracy = 98.17%, valid loss = 0.085039, valid accuracy = 97.42%, duration = 1.421180(s)
Epoch [12/20], train loss = 0.064362, train accuracy = 98.33%, valid loss = 0.081969, valid accuracy = 97.50%, duration = 1.513851(s)
Epoch [13/20], train loss = 0.059520, train accuracy = 98.48%, valid loss = 0.079262, valid accuracy = 97.55%, duration = 1.342157(s)
Epoch [14/20], train loss = 0.055262, train accuracy = 98.59%, valid loss = 0.076923, valid accuracy = 97.67%, duration = 1.455881(s)
Epoch [15/20], train loss = 0.051421, train accuracy = 98.68%, valid loss = 0.074938, valid accuracy = 97.68%, duration = 1.432698(s)
Epoch [16/20], train loss = 0.047934, train accuracy = 98.80%, valid loss = 0.073371, valid accuracy = 97.65%, duration = 1.332904(s)
Epoch [17/20], train loss = 0.044819, train accuracy = 98.88%, valid loss = 0.071822, valid accuracy = 97.67%, duration = 1.324953(s)
Epoch [18/20], train loss = 0.041982, train accuracy = 98.99%, valid loss = 0.070529, valid accuracy = 97.70%, duration = 1.373789(s)
Epoch [19/20], train loss = 0.039356, train accuracy = 99.07%, valid loss = 0.069475, valid accuracy = 97.72%, duration = 1.439676(s)
Epoch [20/20], train loss = 0.036952, train accuracy = 99.14%, valid loss = 0.068388, valid accuracy = 97.70%, duration = 1.361548(s)
Duration for train : 31.496403(s)
<<< Train Finished >>>
Test Accraucy : 97.73%
'''