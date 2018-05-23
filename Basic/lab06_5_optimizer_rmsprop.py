import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-5_board"
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
    optim = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(loss)

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
Total step :  1687
Epoch [ 1/20], train loss = 0.259838, train accuracy = 92.49%, valid loss = 0.116316, valid accuracy = 96.50%, duration = 2.083493(s)
Epoch [ 2/20], train loss = 0.106106, train accuracy = 96.86%, valid loss = 0.094500, valid accuracy = 97.15%, duration = 1.849574(s)
Epoch [ 3/20], train loss = 0.074696, train accuracy = 97.89%, valid loss = 0.087937, valid accuracy = 97.55%, duration = 1.620212(s)
Epoch [ 4/20], train loss = 0.056900, train accuracy = 98.44%, valid loss = 0.085456, valid accuracy = 97.75%, duration = 1.782180(s)
Epoch [ 5/20], train loss = 0.044551, train accuracy = 98.83%, valid loss = 0.088404, valid accuracy = 97.68%, duration = 1.819118(s)
Epoch [ 6/20], train loss = 0.036103, train accuracy = 99.08%, valid loss = 0.092642, valid accuracy = 97.67%, duration = 2.256086(s)
Epoch [ 7/20], train loss = 0.029810, train accuracy = 99.29%, valid loss = 0.091917, valid accuracy = 97.83%, duration = 1.806443(s)
Epoch [ 8/20], train loss = 0.024866, train accuracy = 99.44%, valid loss = 0.093812, valid accuracy = 97.85%, duration = 2.061844(s)
Epoch [ 9/20], train loss = 0.020691, train accuracy = 99.54%, valid loss = 0.098004, valid accuracy = 97.93%, duration = 1.736157(s)
Epoch [10/20], train loss = 0.017291, train accuracy = 99.63%, valid loss = 0.101468, valid accuracy = 97.97%, duration = 1.893189(s)
Epoch [11/20], train loss = 0.014648, train accuracy = 99.67%, valid loss = 0.095778, valid accuracy = 98.05%, duration = 1.779299(s)
Epoch [12/20], train loss = 0.012463, train accuracy = 99.74%, valid loss = 0.106779, valid accuracy = 98.03%, duration = 1.680670(s)
Epoch [13/20], train loss = 0.010718, train accuracy = 99.77%, valid loss = 0.114635, valid accuracy = 97.98%, duration = 1.835607(s)
Epoch [14/20], train loss = 0.009409, train accuracy = 99.81%, valid loss = 0.110029, valid accuracy = 98.05%, duration = 1.591240(s)
Epoch [15/20], train loss = 0.007422, train accuracy = 99.85%, valid loss = 0.119213, valid accuracy = 98.10%, duration = 1.801876(s)
Epoch [16/20], train loss = 0.006675, train accuracy = 99.85%, valid loss = 0.128560, valid accuracy = 98.03%, duration = 1.731313(s)
Epoch [17/20], train loss = 0.006335, train accuracy = 99.89%, valid loss = 0.137669, valid accuracy = 97.87%, duration = 1.837098(s)
Epoch [18/20], train loss = 0.005247, train accuracy = 99.90%, valid loss = 0.137302, valid accuracy = 97.98%, duration = 1.734008(s)
Epoch [19/20], train loss = 0.004460, train accuracy = 99.90%, valid loss = 0.153169, valid accuracy = 97.83%, duration = 1.624976(s)
Epoch [20/20], train loss = 0.003593, train accuracy = 99.93%, valid loss = 0.147666, valid accuracy = 98.03%, duration = 1.742436(s)
Duration for train : 38.959856(s)
<<< Train Finished >>>
Test Accraucy : 97.68%
'''