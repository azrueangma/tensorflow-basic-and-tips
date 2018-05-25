import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-1_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 20
ALPHA = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)

def l1_loss (x, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(x), name = name)
    return output

def linear(x, output_dim, weight_decay = None, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')

        if weight_decay:
            wd = l1_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

def relu_layer(x, output_dim, weight_decay = None, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')

        if weight_decay:
            wd = l1_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(X, 256, ALPHA, 'Relu_Layer1')
h2 = relu_layer(h1, 128, ALPHA, 'Relu_Layer2')
h3 = relu_layer(h2, 64, ALPHA, 'Relu_Layer3')
logits = linear(h1, NCLASS, ALPHA, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+tf.reduce_mean(weight_decay_loss)
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
Epoch [ 1/20], train loss = 0.284560, train accuracy = 93.07%, valid loss = 0.119560, valid accuracy = 96.68%, duration = 3.776719(s)
Epoch [ 2/20], train loss = 0.148584, train accuracy = 96.94%, valid loss = 0.088644, valid accuracy = 97.35%, duration = 3.119532(s)
Epoch [ 3/20], train loss = 0.119609, train accuracy = 97.83%, valid loss = 0.070318, valid accuracy = 97.97%, duration = 3.223342(s)
Epoch [ 4/20], train loss = 0.103857, train accuracy = 98.32%, valid loss = 0.076275, valid accuracy = 97.75%, duration = 3.259548(s)
Epoch [ 5/20], train loss = 0.093769, train accuracy = 98.57%, valid loss = 0.067282, valid accuracy = 97.95%, duration = 3.619057(s)
Epoch [ 6/20], train loss = 0.086590, train accuracy = 98.85%, valid loss = 0.059731, valid accuracy = 98.08%, duration = 3.238629(s)
Epoch [ 7/20], train loss = 0.079225, train accuracy = 99.05%, valid loss = 0.071225, valid accuracy = 97.80%, duration = 4.467795(s)
Epoch [ 8/20], train loss = 0.076121, train accuracy = 99.15%, valid loss = 0.068598, valid accuracy = 98.22%, duration = 6.238947(s)
Epoch [ 9/20], train loss = 0.073133, train accuracy = 99.23%, valid loss = 0.069080, valid accuracy = 97.93%, duration = 3.998061(s)
Epoch [10/20], train loss = 0.070262, train accuracy = 99.30%, valid loss = 0.061635, valid accuracy = 98.35%, duration = 3.167520(s)
Epoch [11/20], train loss = 0.067115, train accuracy = 99.36%, valid loss = 0.060618, valid accuracy = 98.20%, duration = 3.481785(s)
Epoch [12/20], train loss = 0.067333, train accuracy = 99.34%, valid loss = 0.062017, valid accuracy = 98.23%, duration = 3.351747(s)
Epoch [13/20], train loss = 0.063237, train accuracy = 99.46%, valid loss = 0.063113, valid accuracy = 98.22%, duration = 3.396948(s)
Epoch [14/20], train loss = 0.060558, train accuracy = 99.50%, valid loss = 0.065362, valid accuracy = 98.22%, duration = 3.102972(s)
Epoch [15/20], train loss = 0.062099, train accuracy = 99.45%, valid loss = 0.072686, valid accuracy = 98.08%, duration = 3.024870(s)
Epoch [16/20], train loss = 0.058314, train accuracy = 99.57%, valid loss = 0.087702, valid accuracy = 97.75%, duration = 2.349672(s)
Epoch [17/20], train loss = 0.060089, train accuracy = 99.51%, valid loss = 0.075940, valid accuracy = 97.90%, duration = 2.251210(s)
Epoch [18/20], train loss = 0.056575, train accuracy = 99.60%, valid loss = 0.073415, valid accuracy = 98.17%, duration = 2.255103(s)
Epoch [19/20], train loss = 0.057775, train accuracy = 99.52%, valid loss = 0.065279, valid accuracy = 98.22%, duration = 2.269778(s)
Epoch [20/20], train loss = 0.053292, train accuracy = 99.70%, valid loss = 0.073353, valid accuracy = 98.00%, duration = 2.362739(s)
Duration for train : 66.538038(s)
<<< Train Finished >>>
Test Accraucy : 98.05%

'''
