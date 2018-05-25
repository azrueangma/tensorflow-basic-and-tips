import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-1_board"
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
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def sigmoid_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = sigmoid_layer(X, 256, 'Sigmoid_Layer1')
h2 = sigmoid_layer(h1, 128, 'Sigmoid_Layer2')
h3 = sigmoid_layer(h2, 64, 'Sigmoid_Layer3')
logits = linear(h3, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

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
Epoch [ 1/20], train loss = 1.515297, train accuracy = 52.20%, valid loss = 0.911828, valid accuracy = 70.57%, duration = 1.988077(s)
Epoch [ 2/20], train loss = 0.779595, train accuracy = 75.15%, valid loss = 0.660727, valid accuracy = 79.13%, duration = 2.371706(s)
Epoch [ 3/20], train loss = 0.608572, train accuracy = 80.99%, valid loss = 0.558178, valid accuracy = 82.37%, duration = 1.920295(s)
Epoch [ 4/20], train loss = 0.520970, train accuracy = 83.78%, valid loss = 0.493837, valid accuracy = 84.60%, duration = 2.065356(s)
Epoch [ 5/20], train loss = 0.464879, train accuracy = 85.59%, valid loss = 0.449712, valid accuracy = 85.92%, duration = 1.917942(s)
Epoch [ 6/20], train loss = 0.424835, train accuracy = 86.91%, valid loss = 0.420518, valid accuracy = 87.18%, duration = 2.021676(s)
Epoch [ 7/20], train loss = 0.393795, train accuracy = 87.93%, valid loss = 0.396166, valid accuracy = 87.92%, duration = 1.946982(s)
Epoch [ 8/20], train loss = 0.368932, train accuracy = 88.69%, valid loss = 0.379397, valid accuracy = 88.20%, duration = 2.018051(s)
Epoch [ 9/20], train loss = 0.348360, train accuracy = 89.31%, valid loss = 0.371131, valid accuracy = 88.62%, duration = 1.965095(s)
Epoch [10/20], train loss = 0.330704, train accuracy = 89.85%, valid loss = 0.354393, valid accuracy = 89.22%, duration = 1.991610(s)
Epoch [11/20], train loss = 0.315609, train accuracy = 90.34%, valid loss = 0.341141, valid accuracy = 89.63%, duration = 1.967831(s)
Epoch [12/20], train loss = 0.302236, train accuracy = 90.74%, valid loss = 0.329850, valid accuracy = 89.95%, duration = 1.984796(s)
Epoch [13/20], train loss = 0.289712, train accuracy = 91.19%, valid loss = 0.322029, valid accuracy = 90.10%, duration = 1.995432(s)
Epoch [14/20], train loss = 0.278777, train accuracy = 91.43%, valid loss = 0.314941, valid accuracy = 90.40%, duration = 1.948630(s)
Epoch [15/20], train loss = 0.268781, train accuracy = 91.72%, valid loss = 0.307977, valid accuracy = 90.62%, duration = 2.012543(s)
Epoch [16/20], train loss = 0.259341, train accuracy = 92.06%, valid loss = 0.302980, valid accuracy = 90.88%, duration = 1.917241(s)
Epoch [17/20], train loss = 0.250740, train accuracy = 92.37%, valid loss = 0.293449, valid accuracy = 91.07%, duration = 2.032063(s)
Epoch [18/20], train loss = 0.242543, train accuracy = 92.63%, valid loss = 0.290043, valid accuracy = 91.17%, duration = 1.939686(s)
Epoch [19/20], train loss = 0.234988, train accuracy = 92.82%, valid loss = 0.287979, valid accuracy = 91.32%, duration = 2.047833(s)
Epoch [20/20], train loss = 0.227930, train accuracy = 93.00%, valid loss = 0.282882, valid accuracy = 91.58%, duration = 1.879061(s)
Duration for train : 40.557978(s)
<<< Train Finished >>>
Test Accraucy : 90.86%
'''
