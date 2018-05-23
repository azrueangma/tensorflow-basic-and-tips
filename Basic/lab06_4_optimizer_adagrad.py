import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-4_board"
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
    optim = tf.train.AdagradOptimizer(learning_rate = 0.001).minimize(loss)

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
Epoch [ 1/20], train loss = 0.570723, train accuracy = 86.67%, valid loss = 0.386448, valid accuracy = 90.18%, duration = 1.861321(s)
Epoch [ 2/20], train loss = 0.366375, train accuracy = 90.42%, valid loss = 0.331165, valid accuracy = 91.35%, duration = 1.716519(s)
Epoch [ 3/20], train loss = 0.328563, train accuracy = 91.22%, valid loss = 0.304886, valid accuracy = 91.98%, duration = 2.326459(s)
Epoch [ 4/20], train loss = 0.307325, train accuracy = 91.68%, valid loss = 0.288161, valid accuracy = 92.37%, duration = 1.609893(s)
Epoch [ 5/20], train loss = 0.292703, train accuracy = 92.07%, valid loss = 0.276045, valid accuracy = 92.63%, duration = 1.752660(s)
Epoch [ 6/20], train loss = 0.281562, train accuracy = 92.37%, valid loss = 0.266618, valid accuracy = 92.73%, duration = 1.742044(s)
Epoch [ 7/20], train loss = 0.272565, train accuracy = 92.63%, valid loss = 0.258887, valid accuracy = 92.97%, duration = 1.708101(s)
Epoch [ 8/20], train loss = 0.265024, train accuracy = 92.85%, valid loss = 0.252327, valid accuracy = 93.30%, duration = 1.811798(s)
Epoch [ 9/20], train loss = 0.258522, train accuracy = 93.02%, valid loss = 0.246620, valid accuracy = 93.52%, duration = 1.709159(s)
Epoch [10/20], train loss = 0.252803, train accuracy = 93.16%, valid loss = 0.241600, valid accuracy = 93.63%, duration = 1.713551(s)
Epoch [11/20], train loss = 0.247698, train accuracy = 93.28%, valid loss = 0.237120, valid accuracy = 93.73%, duration = 1.502903(s)
Epoch [12/20], train loss = 0.243088, train accuracy = 93.38%, valid loss = 0.233079, valid accuracy = 93.88%, duration = 1.546989(s)
Epoch [13/20], train loss = 0.238888, train accuracy = 93.46%, valid loss = 0.229383, valid accuracy = 93.95%, duration = 1.570102(s)
Epoch [14/20], train loss = 0.235025, train accuracy = 93.57%, valid loss = 0.225984, valid accuracy = 93.93%, duration = 1.452877(s)
Epoch [15/20], train loss = 0.231451, train accuracy = 93.67%, valid loss = 0.222848, valid accuracy = 94.08%, duration = 1.494789(s)
Epoch [16/20], train loss = 0.228126, train accuracy = 93.74%, valid loss = 0.219933, valid accuracy = 94.17%, duration = 1.596604(s)
Epoch [17/20], train loss = 0.225012, train accuracy = 93.85%, valid loss = 0.217195, valid accuracy = 94.20%, duration = 1.493411(s)
Epoch [18/20], train loss = 0.222083, train accuracy = 93.95%, valid loss = 0.214623, valid accuracy = 94.27%, duration = 1.608969(s)
Epoch [19/20], train loss = 0.219309, train accuracy = 94.03%, valid loss = 0.212189, valid accuracy = 94.33%, duration = 1.578214(s)
Epoch [20/20], train loss = 0.216670, train accuracy = 94.11%, valid loss = 0.209870, valid accuracy = 94.37%, duration = 1.582621(s)
Duration for train : 36.097778(s)
<<< Train Finished >>>
Test Accraucy : 94.05%

'''