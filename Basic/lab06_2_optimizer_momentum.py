import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-2_board"
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
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def relu_layer(x, output_dim, name):
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

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.1).minimize(loss)

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
Epoch [ 1/30], train loss = 0.377192, train accuracy = 89.45%, valid loss = 0.198012, valid accuracy = 94.17%, duration = 2.174780(s)
Epoch [ 2/30], train loss = 0.180603, train accuracy = 94.74%, valid loss = 0.148171, valid accuracy = 95.63%, duration = 2.051634(s)
Epoch [ 3/30], train loss = 0.129982, train accuracy = 96.24%, valid loss = 0.109284, valid accuracy = 96.83%, duration = 2.220698(s)
Epoch [ 4/30], train loss = 0.100961, train accuracy = 97.05%, valid loss = 0.100360, valid accuracy = 96.90%, duration = 2.288061(s)
Epoch [ 5/30], train loss = 0.081893, train accuracy = 97.63%, valid loss = 0.089591, valid accuracy = 97.15%, duration = 2.112694(s)
Epoch [ 6/30], train loss = 0.068262, train accuracy = 98.03%, valid loss = 0.079266, valid accuracy = 97.57%, duration = 2.105555(s)
Epoch [ 7/30], train loss = 0.056867, train accuracy = 98.36%, valid loss = 0.077577, valid accuracy = 97.57%, duration = 2.048118(s)
Epoch [ 8/30], train loss = 0.048573, train accuracy = 98.63%, valid loss = 0.072409, valid accuracy = 97.63%, duration = 2.040262(s)
Epoch [ 9/30], train loss = 0.040755, train accuracy = 98.86%, valid loss = 0.073217, valid accuracy = 97.63%, duration = 2.038349(s)
Epoch [10/30], train loss = 0.035572, train accuracy = 99.02%, valid loss = 0.067701, valid accuracy = 97.98%, duration = 2.052685(s)
Epoch [11/30], train loss = 0.029689, train accuracy = 99.25%, valid loss = 0.065687, valid accuracy = 97.85%, duration = 2.081913(s)
Epoch [12/30], train loss = 0.025594, train accuracy = 99.31%, valid loss = 0.063266, valid accuracy = 98.15%, duration = 2.066133(s)
Epoch [13/30], train loss = 0.021596, train accuracy = 99.49%, valid loss = 0.066243, valid accuracy = 97.90%, duration = 2.097233(s)
Epoch [14/30], train loss = 0.018021, train accuracy = 99.60%, valid loss = 0.060682, valid accuracy = 98.07%, duration = 2.117744(s)
Epoch [15/30], train loss = 0.015795, train accuracy = 99.68%, valid loss = 0.061310, valid accuracy = 98.07%, duration = 2.160048(s)
Epoch [16/30], train loss = 0.013002, train accuracy = 99.79%, valid loss = 0.059654, valid accuracy = 98.23%, duration = 2.171897(s)
Epoch [17/30], train loss = 0.011438, train accuracy = 99.82%, valid loss = 0.063169, valid accuracy = 98.05%, duration = 2.670938(s)
Epoch [18/30], train loss = 0.009378, train accuracy = 99.89%, valid loss = 0.066123, valid accuracy = 97.97%, duration = 2.164576(s)
Epoch [19/30], train loss = 0.008247, train accuracy = 99.90%, valid loss = 0.059994, valid accuracy = 98.33%, duration = 2.327201(s)
Epoch [20/30], train loss = 0.007329, train accuracy = 99.91%, valid loss = 0.060208, valid accuracy = 98.12%, duration = 2.078888(s)
Epoch [21/30], train loss = 0.006268, train accuracy = 99.95%, valid loss = 0.060199, valid accuracy = 98.27%, duration = 2.085424(s)
Epoch [22/30], train loss = 0.005500, train accuracy = 99.96%, valid loss = 0.061434, valid accuracy = 98.18%, duration = 2.052067(s)
Epoch [23/30], train loss = 0.004713, train accuracy = 99.97%, valid loss = 0.062151, valid accuracy = 98.35%, duration = 2.029168(s)
Epoch [24/30], train loss = 0.004146, train accuracy = 99.98%, valid loss = 0.061259, valid accuracy = 98.20%, duration = 2.101185(s)
Epoch [25/30], train loss = 0.003684, train accuracy = 99.99%, valid loss = 0.060916, valid accuracy = 98.28%, duration = 2.018828(s)
Epoch [26/30], train loss = 0.003267, train accuracy = 99.99%, valid loss = 0.062404, valid accuracy = 98.33%, duration = 2.039755(s)
Epoch [27/30], train loss = 0.003014, train accuracy = 99.99%, valid loss = 0.062479, valid accuracy = 98.27%, duration = 2.048357(s)
Epoch [28/30], train loss = 0.002745, train accuracy = 99.99%, valid loss = 0.062898, valid accuracy = 98.23%, duration = 2.034076(s)
Epoch [29/30], train loss = 0.002507, train accuracy = 99.99%, valid loss = 0.063209, valid accuracy = 98.25%, duration = 2.018349(s)
Epoch [30/30], train loss = 0.002235, train accuracy = 100.00%, valid loss = 0.064421, valid accuracy = 98.20%, duration = 2.056684(s)
Duration for train : 64.527291(s)
<<< Train Finished >>>
Test Accraucy : 97.95%
'''
