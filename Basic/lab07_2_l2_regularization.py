import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-2_board"
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

def l2_loss (x, name='l2_loss'):
    output = tf.reduce_sum(tf.square(x), name = name)/2
    return output

def linear(x, output_dim, weight_decay = None, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')

        if weight_decay:
            wd = l2_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

def relu_layer(x, output_dim, weight_decay = None, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')

        if weight_decay:
            wd = l2_loss(W)*weight_decay
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
Epoch [ 1/20], train loss = 0.236437, train accuracy = 93.19%, valid loss = 0.121948, valid accuracy = 96.47%, duration = 2.420570(s)
Epoch [ 2/20], train loss = 0.098225, train accuracy = 97.20%, valid loss = 0.083879, valid accuracy = 97.47%, duration = 2.659546(s)
Epoch [ 3/20], train loss = 0.067024, train accuracy = 98.05%, valid loss = 0.069901, valid accuracy = 97.95%, duration = 2.545539(s)
Epoch [ 4/20], train loss = 0.050630, train accuracy = 98.58%, valid loss = 0.078543, valid accuracy = 97.57%, duration = 2.073496(s)
Epoch [ 5/20], train loss = 0.040050, train accuracy = 98.92%, valid loss = 0.066128, valid accuracy = 98.08%, duration = 2.083706(s)
Epoch [ 6/20], train loss = 0.032976, train accuracy = 99.18%, valid loss = 0.065513, valid accuracy = 98.13%, duration = 2.255514(s)
Epoch [ 7/20], train loss = 0.028090, train accuracy = 99.33%, valid loss = 0.076010, valid accuracy = 98.10%, duration = 2.266753(s)
Epoch [ 8/20], train loss = 0.024558, train accuracy = 99.47%, valid loss = 0.070432, valid accuracy = 97.85%, duration = 2.242041(s)
Epoch [ 9/20], train loss = 0.022888, train accuracy = 99.52%, valid loss = 0.086905, valid accuracy = 97.58%, duration = 2.413728(s)
Epoch [10/20], train loss = 0.021237, train accuracy = 99.59%, valid loss = 0.076642, valid accuracy = 97.97%, duration = 2.953831(s)
Epoch [11/20], train loss = 0.019910, train accuracy = 99.64%, valid loss = 0.073018, valid accuracy = 98.03%, duration = 2.357763(s)
Epoch [12/20], train loss = 0.019335, train accuracy = 99.65%, valid loss = 0.080459, valid accuracy = 98.10%, duration = 2.376768(s)
Epoch [13/20], train loss = 0.019118, train accuracy = 99.68%, valid loss = 0.071467, valid accuracy = 98.05%, duration = 2.478708(s)
Epoch [14/20], train loss = 0.018206, train accuracy = 99.72%, valid loss = 0.071253, valid accuracy = 98.03%, duration = 2.439470(s)
Epoch [15/20], train loss = 0.019090, train accuracy = 99.68%, valid loss = 0.086418, valid accuracy = 98.03%, duration = 2.352211(s)
Epoch [16/20], train loss = 0.016028, train accuracy = 99.81%, valid loss = 0.092024, valid accuracy = 97.75%, duration = 2.524159(s)
Epoch [17/20], train loss = 0.020119, train accuracy = 99.64%, valid loss = 0.077886, valid accuracy = 98.33%, duration = 2.180729(s)
Epoch [18/20], train loss = 0.016428, train accuracy = 99.80%, valid loss = 0.070269, valid accuracy = 98.25%, duration = 2.217158(s)
Epoch [19/20], train loss = 0.017123, train accuracy = 99.79%, valid loss = 0.088437, valid accuracy = 98.00%, duration = 2.092017(s)
Epoch [20/20], train loss = 0.017594, train accuracy = 99.75%, valid loss = 0.082152, valid accuracy = 98.02%, duration = 2.053086(s)
Duration for train : 47.566494(s)
<<< Train Finished >>>
Test Accraucy : 97.97%
'''