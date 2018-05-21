#lab04_4 model
import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test = load_data.load_pendigits(seed = 0, scaling = True)

BOARD_PATH = "./board/lab05-2_board"
NSAMPLES = int(len(x_train)+len(x_test))
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 1000

print("The number of data samples : ", NSAMPLES)
print("The dimension of data samples : ", INPUT_DIM)

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def tanh_linear(x, output_dim, name):
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

h1 = tanh_linear(X, 32, 'FC_Layer1')
h2 = tanh_linear(h1, 16, 'FC_Layer2')
logits = linear(h2, NCLASS, 'FC_Layer3')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = -tf.reduce_sum(Y_one_hot*tf.log(hypothesis), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

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
total_step = int(NSAMPLES/BATCH_SIZE)
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

        if (epoch+1) %100 == 0:
            va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
            epoch_valid_acc = va/len(x_validation)
            epoch_valid_loss = vl/len(x_validation)
            print("Epoch [{:3d}/{:3d}], train loss = {:.6f}, train accuracy = {:.2%}, valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

'''
Total step :  309
Epoch [100/1000], train loss = 0.183622, train accuracy = 72.42%, valid loss = 0.319931, valid accuracy = 90.17%, duration = 0.094493(s)
Epoch [200/1000], train loss = 0.156293, train accuracy = 73.27%, valid loss = 0.245531, valid accuracy = 93.81%, duration = 0.094478(s)
Epoch [300/1000], train loss = 0.125184, train accuracy = 73.50%, valid loss = 0.186327, valid accuracy = 93.81%, duration = 0.096408(s)
Epoch [400/1000], train loss = 0.110476, train accuracy = 74.39%, valid loss = 0.147716, valid accuracy = 96.09%, duration = 0.096418(s)
Epoch [500/1000], train loss = 0.093580, train accuracy = 75.27%, valid loss = 0.165819, valid accuracy = 96.00%, duration = 0.094078(s)
Epoch [600/1000], train loss = 0.108091, train accuracy = 74.29%, valid loss = 0.166259, valid accuracy = 95.81%, duration = 0.098981(s)
Epoch [700/1000], train loss = 0.119325, train accuracy = 73.87%, valid loss = 0.184774, valid accuracy = 94.63%, duration = 0.095848(s)
Epoch [800/1000], train loss = 0.130028, train accuracy = 73.76%, valid loss = 0.199044, valid accuracy = 94.27%, duration = 0.104786(s)
Epoch [900/1000], train loss = 0.092470, train accuracy = 75.02%, valid loss = 0.149739, valid accuracy = 95.81%, duration = 0.097738(s)
Epoch [1000/1000], train loss = 0.090932, train accuracy = 75.27%, valid loss = 0.146308, valid accuracy = 95.81%, duration = 0.103294(s)
Duration for train : 100.050362(s)
<<< Train Finished >>>
'''
