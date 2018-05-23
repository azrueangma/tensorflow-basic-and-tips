import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-6_board"
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
    optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

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
Epoch [ 1/20], train loss = 0.236856, train accuracy = 93.15%, valid loss = 0.120456, valid accuracy = 96.38%, duration = 1.898321(s)
Epoch [ 2/20], train loss = 0.097741, train accuracy = 97.08%, valid loss = 0.102480, valid accuracy = 96.93%, duration = 1.557079(s)
Epoch [ 3/20], train loss = 0.062926, train accuracy = 98.08%, valid loss = 0.091434, valid accuracy = 97.38%, duration = 1.560519(s)
Epoch [ 4/20], train loss = 0.043049, train accuracy = 98.73%, valid loss = 0.085437, valid accuracy = 97.50%, duration = 1.632492(s)
Epoch [ 5/20], train loss = 0.030023, train accuracy = 99.14%, valid loss = 0.092699, valid accuracy = 97.25%, duration = 1.801769(s)
Epoch [ 6/20], train loss = 0.021357, train accuracy = 99.43%, valid loss = 0.091981, valid accuracy = 97.53%, duration = 1.597954(s)
Epoch [ 7/20], train loss = 0.016986, train accuracy = 99.54%, valid loss = 0.073825, valid accuracy = 97.98%, duration = 1.798308(s)
Epoch [ 8/20], train loss = 0.014111, train accuracy = 99.55%, valid loss = 0.108636, valid accuracy = 97.18%, duration = 1.721063(s)
Epoch [ 9/20], train loss = 0.013341, train accuracy = 99.54%, valid loss = 0.100827, valid accuracy = 97.25%, duration = 1.646226(s)
Epoch [10/20], train loss = 0.009314, train accuracy = 99.71%, valid loss = 0.080394, valid accuracy = 97.98%, duration = 1.801355(s)
Epoch [11/20], train loss = 0.008198, train accuracy = 99.76%, valid loss = 0.083006, valid accuracy = 98.07%, duration = 1.621012(s)
Epoch [12/20], train loss = 0.008517, train accuracy = 99.71%, valid loss = 0.083945, valid accuracy = 97.98%, duration = 1.538324(s)
Epoch [13/20], train loss = 0.008250, train accuracy = 99.73%, valid loss = 0.091388, valid accuracy = 97.93%, duration = 1.774744(s)
Epoch [14/20], train loss = 0.007620, train accuracy = 99.75%, valid loss = 0.097334, valid accuracy = 97.88%, duration = 1.729386(s)
Epoch [15/20], train loss = 0.006279, train accuracy = 99.80%, valid loss = 0.089528, valid accuracy = 98.03%, duration = 1.704075(s)
Epoch [16/20], train loss = 0.007313, train accuracy = 99.75%, valid loss = 0.103003, valid accuracy = 98.07%, duration = 1.778698(s)
Epoch [17/20], train loss = 0.005835, train accuracy = 99.83%, valid loss = 0.099318, valid accuracy = 98.00%, duration = 1.575845(s)
Epoch [18/20], train loss = 0.005202, train accuracy = 99.83%, valid loss = 0.100552, valid accuracy = 98.13%, duration = 1.829987(s)
Epoch [19/20], train loss = 0.007005, train accuracy = 99.78%, valid loss = 0.117238, valid accuracy = 97.87%, duration = 1.677559(s)
Epoch [20/20], train loss = 0.006036, train accuracy = 99.78%, valid loss = 0.098431, valid accuracy = 98.18%, duration = 1.707393(s)
Duration for train : 36.481726(s)
<<< Train Finished >>>
Test Accraucy : 98.02%
'''