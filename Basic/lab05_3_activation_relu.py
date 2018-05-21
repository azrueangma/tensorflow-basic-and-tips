#lab04_4 model
import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test = load_data.load_pendigits(seed = 0, scaling = True)

BOARD_PATH = "./board/lab05-3_board"
NSAMPLES = int(len(x_train)+len(x_test))
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 200

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
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name='h')
        return h

def relu_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name='h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')

h1 = relu_linear(X, 32, 'FC_Layer1')
h2 = relu_linear(h1, 16, 'FC_Layer2')
logits = linear(h2, NCLASS, 'FC_Layer3')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss = -tf.reduce_sum(Y_one_hot * tf.log(hypothesis), name='loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.variable_scope("Pred_and_Acc"):
    predict = tf.argmax(hypothesis, axis=1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_loss = tf.placeholder(tf.float32)
    loss_avg  = tf.summary.scalar('avg_loss', avg_loss)
    avg_acc = tf.placeholder(tf.float32)
    acc_avg = tf.summary.scalar('avg_acc', avg_acc)
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

        s = sess.run(merged, feed_dict = {avg_loss:loss_per_epoch, avg_acc:acc_per_epoch})
        writer.add_summary(s, global_step = epoch)

        if (epoch+1) % 20 == 0:
            va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
            epoch_valid_acc = va/len(x_validation)
            epoch_valid_loss = vl/len(x_validation)
            print("Epoch [{:3d}/{:3d}], train loss = {:.6f}, train accuracy = {:.2%}, valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

'''
Epoch [ 20/200], train loss = 0.104339, train accuracy = 97.51%, valid loss = 0.106230, valid accuracy = 97.27%, duration = 0.082370(s)
Epoch [ 40/200], train loss = 0.067286, train accuracy = 98.42%, valid loss = 0.083363, valid accuracy = 97.91%, duration = 0.103476(s)
Epoch [ 60/200], train loss = 0.052767, train accuracy = 98.89%, valid loss = 0.077430, valid accuracy = 98.36%, duration = 0.095469(s)
Epoch [ 80/200], train loss = 0.044407, train accuracy = 99.09%, valid loss = 0.075799, valid accuracy = 98.36%, duration = 0.082597(s)
Epoch [100/200], train loss = 0.038462, train accuracy = 99.22%, valid loss = 0.077654, valid accuracy = 98.27%, duration = 0.084502(s)
Epoch [120/200], train loss = 0.033949, train accuracy = 99.28%, valid loss = 0.077725, valid accuracy = 98.27%, duration = 0.082612(s)
Epoch [140/200], train loss = 0.030063, train accuracy = 99.43%, valid loss = 0.078515, valid accuracy = 98.27%, duration = 0.104323(s)
Epoch [160/200], train loss = 0.027012, train accuracy = 99.45%, valid loss = 0.079685, valid accuracy = 98.27%, duration = 0.080808(s)
Epoch [180/200], train loss = 0.024462, train accuracy = 99.53%, valid loss = 0.080803, valid accuracy = 98.18%, duration = 0.077320(s)
Epoch [200/200], train loss = 0.022288, train accuracy = 99.56%, valid loss = 0.081579, valid accuracy = 98.18%, duration = 0.082301(s)
Duration for train : 17.508737(s)
<<< Train Finished >>>
'''