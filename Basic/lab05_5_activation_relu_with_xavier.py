import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-5_board"
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
Total step :  1687
Epoch [ 1/20], train loss = 0.424426, train accuracy = 88.61%, valid loss = 0.257967, valid accuracy = 92.67%, duration = 1.576067(s)
Epoch [ 2/20], train loss = 0.241308, train accuracy = 93.13%, valid loss = 0.198232, valid accuracy = 94.53%, duration = 1.448732(s)
Epoch [ 3/20], train loss = 0.190499, train accuracy = 94.64%, valid loss = 0.159842, valid accuracy = 95.58%, duration = 1.565295(s)
Epoch [ 4/20], train loss = 0.157313, train accuracy = 95.58%, valid loss = 0.142428, valid accuracy = 96.05%, duration = 1.574245(s)
Epoch [ 5/20], train loss = 0.133861, train accuracy = 96.27%, valid loss = 0.127927, valid accuracy = 96.32%, duration = 1.742331(s)
Epoch [ 6/20], train loss = 0.116924, train accuracy = 96.78%, valid loss = 0.113794, valid accuracy = 96.62%, duration = 1.579312(s)
Epoch [ 7/20], train loss = 0.103077, train accuracy = 97.18%, valid loss = 0.108140, valid accuracy = 96.80%, duration = 1.428400(s)
Epoch [ 8/20], train loss = 0.092611, train accuracy = 97.47%, valid loss = 0.100026, valid accuracy = 96.88%, duration = 1.553183(s)
Epoch [ 9/20], train loss = 0.083412, train accuracy = 97.72%, valid loss = 0.095319, valid accuracy = 97.07%, duration = 1.606360(s)
Epoch [10/20], train loss = 0.076268, train accuracy = 97.93%, valid loss = 0.086408, valid accuracy = 97.48%, duration = 1.499791(s)
Epoch [11/20], train loss = 0.069869, train accuracy = 98.15%, valid loss = 0.086668, valid accuracy = 97.23%, duration = 1.667992(s)
Epoch [12/20], train loss = 0.064349, train accuracy = 98.29%, valid loss = 0.080551, valid accuracy = 97.58%, duration = 1.473667(s)
Epoch [13/20], train loss = 0.059453, train accuracy = 98.41%, valid loss = 0.079424, valid accuracy = 97.55%, duration = 1.537470(s)
Epoch [14/20], train loss = 0.055007, train accuracy = 98.59%, valid loss = 0.075941, valid accuracy = 97.72%, duration = 1.666847(s)
Epoch [15/20], train loss = 0.051227, train accuracy = 98.64%, valid loss = 0.076240, valid accuracy = 97.52%, duration = 1.472071(s)
Epoch [16/20], train loss = 0.047620, train accuracy = 98.75%, valid loss = 0.071352, valid accuracy = 97.85%, duration = 1.495293(s)
Epoch [17/20], train loss = 0.044595, train accuracy = 98.86%, valid loss = 0.073488, valid accuracy = 97.65%, duration = 1.527431(s)
Epoch [18/20], train loss = 0.041876, train accuracy = 98.94%, valid loss = 0.070750, valid accuracy = 97.63%, duration = 1.418654(s)
Epoch [19/20], train loss = 0.039385, train accuracy = 99.02%, valid loss = 0.067276, valid accuracy = 97.70%, duration = 1.506294(s)
Epoch [20/20], train loss = 0.036682, train accuracy = 99.13%, valid loss = 0.068263, valid accuracy = 97.88%, duration = 1.536289(s)
Duration for train : 31.423383(s)
<<< Train Finished >>>
Test Accraucy : 97.92%
'''
