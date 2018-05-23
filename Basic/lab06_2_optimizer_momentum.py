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
    optim = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.1).minimize(loss)

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
Epoch [ 1/20], train loss = 0.411445, train accuracy = 88.96%, valid loss = 0.252972, valid accuracy = 92.75%, duration = 1.668652(s)
Epoch [ 2/20], train loss = 0.231184, train accuracy = 93.48%, valid loss = 0.191194, valid accuracy = 94.87%, duration = 1.596147(s)
Epoch [ 3/20], train loss = 0.180063, train accuracy = 94.95%, valid loss = 0.156737, valid accuracy = 95.67%, duration = 1.567282(s)
Epoch [ 4/20], train loss = 0.147776, train accuracy = 95.91%, valid loss = 0.133995, valid accuracy = 96.25%, duration = 1.556404(s)
Epoch [ 5/20], train loss = 0.125211, train accuracy = 96.51%, valid loss = 0.118796, valid accuracy = 96.52%, duration = 1.605727(s)
Epoch [ 6/20], train loss = 0.108554, train accuracy = 97.05%, valid loss = 0.108229, valid accuracy = 96.80%, duration = 1.499884(s)
Epoch [ 7/20], train loss = 0.095727, train accuracy = 97.37%, valid loss = 0.100256, valid accuracy = 96.90%, duration = 1.546425(s)
Epoch [ 8/20], train loss = 0.085468, train accuracy = 97.67%, valid loss = 0.094070, valid accuracy = 97.12%, duration = 1.637721(s)
Epoch [ 9/20], train loss = 0.077077, train accuracy = 97.94%, valid loss = 0.088996, valid accuracy = 97.27%, duration = 1.562450(s)
Epoch [10/20], train loss = 0.070050, train accuracy = 98.16%, valid loss = 0.084927, valid accuracy = 97.37%, duration = 1.940986(s)
Epoch [11/20], train loss = 0.064011, train accuracy = 98.34%, valid loss = 0.081544, valid accuracy = 97.50%, duration = 1.769515(s)
Epoch [12/20], train loss = 0.058784, train accuracy = 98.51%, valid loss = 0.078784, valid accuracy = 97.60%, duration = 1.694709(s)
Epoch [13/20], train loss = 0.054175, train accuracy = 98.64%, valid loss = 0.076452, valid accuracy = 97.65%, duration = 2.064496(s)
Epoch [14/20], train loss = 0.050082, train accuracy = 98.72%, valid loss = 0.074439, valid accuracy = 97.60%, duration = 1.480584(s)
Epoch [15/20], train loss = 0.046395, train accuracy = 98.82%, valid loss = 0.072759, valid accuracy = 97.65%, duration = 1.626531(s)
Epoch [16/20], train loss = 0.043107, train accuracy = 98.94%, valid loss = 0.071284, valid accuracy = 97.68%, duration = 1.521768(s)
Epoch [17/20], train loss = 0.040119, train accuracy = 99.03%, valid loss = 0.069936, valid accuracy = 97.68%, duration = 1.521353(s)
Epoch [18/20], train loss = 0.037412, train accuracy = 99.11%, valid loss = 0.068778, valid accuracy = 97.70%, duration = 1.595815(s)
Epoch [19/20], train loss = 0.034945, train accuracy = 99.20%, valid loss = 0.067800, valid accuracy = 97.73%, duration = 1.501553(s)
Epoch [20/20], train loss = 0.032674, train accuracy = 99.26%, valid loss = 0.066918, valid accuracy = 97.73%, duration = 1.595445(s)
Duration for train : 35.257759(s)
<<< Train Finished >>>
Test Accraucy : 97.80%
'''