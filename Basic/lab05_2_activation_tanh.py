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

def tanh_layer(x, output_dim, name):
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

h1 = tanh_layer(X, 256, 'Tanh_Layer1')
h2 = tanh_layer(h1, 128, 'Tanh_Layer2')
h3 = tanh_layer(h2, 64, 'Tanh_Layer3')
logits = linear(h3, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

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
        loss_per_epoch /= total_step
        acc_per_epoch /= total_step

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va
        epoch_valid_loss = vl

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
    print("Test Accraucy : {:.2%}".format(ta))

'''
Epoch [ 1/30], train loss = 3.951365, train accuracy = 37.07%, valid loss = 2.171855, valid accuracy = 52.77%, duration = 2.070824(s)
Epoch [ 2/30], train loss = 1.715625, train accuracy = 59.15%, valid loss = 1.470108, valid accuracy = 63.15%, duration = 2.046549(s)
Epoch [ 3/30], train loss = 1.237092, train accuracy = 66.90%, valid loss = 1.187219, valid accuracy = 68.18%, duration = 2.062458(s)
Epoch [ 4/30], train loss = 0.999607, train accuracy = 71.19%, valid loss = 1.063133, valid accuracy = 70.10%, duration = 2.021473(s)
Epoch [ 5/30], train loss = 0.850899, train accuracy = 74.58%, valid loss = 0.982187, valid accuracy = 71.75%, duration = 2.076509(s)
Epoch [ 6/30], train loss = 0.738712, train accuracy = 77.27%, valid loss = 0.927562, valid accuracy = 73.27%, duration = 2.095117(s)
Epoch [ 7/30], train loss = 0.657836, train accuracy = 79.46%, valid loss = 0.891331, valid accuracy = 73.72%, duration = 2.109924(s)
Epoch [ 8/30], train loss = 0.597315, train accuracy = 81.23%, valid loss = 0.876271, valid accuracy = 74.53%, duration = 2.355122(s)
Epoch [ 9/30], train loss = 0.550898, train accuracy = 82.67%, valid loss = 0.852535, valid accuracy = 75.08%, duration = 2.180421(s)
Epoch [10/30], train loss = 0.513956, train accuracy = 83.88%, valid loss = 0.842202, valid accuracy = 75.43%, duration = 2.115796(s)
Epoch [11/30], train loss = 0.483316, train accuracy = 85.08%, valid loss = 0.828534, valid accuracy = 75.65%, duration = 2.149671(s)
Epoch [12/30], train loss = 0.457804, train accuracy = 85.91%, valid loss = 0.820290, valid accuracy = 76.22%, duration = 2.066024(s)
Epoch [13/30], train loss = 0.436603, train accuracy = 86.59%, valid loss = 0.809927, valid accuracy = 76.60%, duration = 2.044297(s)
Epoch [14/30], train loss = 0.418242, train accuracy = 87.37%, valid loss = 0.805967, valid accuracy = 76.82%, duration = 2.207166(s)
Epoch [15/30], train loss = 0.401583, train accuracy = 87.91%, valid loss = 0.797417, valid accuracy = 77.12%, duration = 2.240852(s)
Epoch [16/30], train loss = 0.386887, train accuracy = 88.38%, valid loss = 0.794089, valid accuracy = 77.28%, duration = 2.180097(s)
Epoch [17/30], train loss = 0.373958, train accuracy = 88.81%, valid loss = 0.791204, valid accuracy = 77.95%, duration = 2.061044(s)
Epoch [18/30], train loss = 0.362332, train accuracy = 89.22%, valid loss = 0.789362, valid accuracy = 77.83%, duration = 2.200169(s)
Epoch [19/30], train loss = 0.351965, train accuracy = 89.60%, valid loss = 0.786918, valid accuracy = 77.92%, duration = 2.045280(s)
Epoch [20/30], train loss = 0.341537, train accuracy = 89.97%, valid loss = 0.785245, valid accuracy = 78.07%, duration = 2.021092(s)
Epoch [21/30], train loss = 0.332281, train accuracy = 90.26%, valid loss = 0.783675, valid accuracy = 78.12%, duration = 2.021661(s)
Epoch [22/30], train loss = 0.323758, train accuracy = 90.56%, valid loss = 0.778742, valid accuracy = 78.57%, duration = 2.153815(s)
Epoch [23/30], train loss = 0.315873, train accuracy = 90.85%, valid loss = 0.780951, valid accuracy = 78.20%, duration = 2.002639(s)
Epoch [24/30], train loss = 0.308711, train accuracy = 91.11%, valid loss = 0.776980, valid accuracy = 78.65%, duration = 2.297991(s)
Epoch [25/30], train loss = 0.301523, train accuracy = 91.35%, valid loss = 0.778115, valid accuracy = 78.67%, duration = 2.012946(s)
Epoch [26/30], train loss = 0.294765, train accuracy = 91.58%, valid loss = 0.775366, valid accuracy = 78.80%, duration = 1.991870(s)
Epoch [27/30], train loss = 0.288486, train accuracy = 91.77%, valid loss = 0.777212, valid accuracy = 78.73%, duration = 2.098964(s)
Epoch [28/30], train loss = 0.282608, train accuracy = 92.01%, valid loss = 0.779218, valid accuracy = 79.03%, duration = 2.036591(s)
Epoch [29/30], train loss = 0.276731, train accuracy = 92.13%, valid loss = 0.780664, valid accuracy = 78.72%, duration = 1.983704(s)
Epoch [30/30], train loss = 0.271275, train accuracy = 92.32%, valid loss = 0.779481, valid accuracy = 78.95%, duration = 1.995120(s)
Duration for train : 63.972436(s)
<<< Train Finished >>>
Test Accraucy : 79.40%
'''
