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

TOTAL_EPOCH = 30
ALPHA = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss (tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name = name)
    return output


def linear(tensor_op, output_dim, weight_decay = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            wd = l1_loss(W)
            tf.add_to_collection("weight_decay", wd)

        return h


def relu_layer(tensor_op, output_dim, weight_decay = False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')

        if weight_decay:
            wd = l1_loss(W)
            tf.add_to_collection("weight_decay", wd)

        return h


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(X, 256, True, 'Relu_Layer1')
h2 = relu_layer(h1, 128, True, 'Relu_Layer2')
logits = linear(h2, NCLASS, True, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+ALPHA*tf.reduce_mean(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

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
        loss_per_epoch /= total_step
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, normal_loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va / len(x_validation)
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
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 1.036185, train accuracy = 91.37%, valid loss = 0.181303, valid accuracy = 95.07%, duration = 4.157356(s)
Epoch [ 2/30], train loss = 0.496710, train accuracy = 94.91%, valid loss = 0.149312, valid accuracy = 96.07%, duration = 3.271718(s)
Epoch [ 3/30], train loss = 0.412126, train accuracy = 95.73%, valid loss = 0.127451, valid accuracy = 96.52%, duration = 2.952794(s)
Epoch [ 4/30], train loss = 0.378731, train accuracy = 96.09%, valid loss = 0.152007, valid accuracy = 95.62%, duration = 2.899299(s)
Epoch [ 5/30], train loss = 0.357253, train accuracy = 96.26%, valid loss = 0.125041, valid accuracy = 96.45%, duration = 2.922307(s)
Epoch [ 6/30], train loss = 0.345643, train accuracy = 96.35%, valid loss = 0.130985, valid accuracy = 95.85%, duration = 2.901881(s)
Epoch [ 7/30], train loss = 0.335967, train accuracy = 96.48%, valid loss = 0.119779, valid accuracy = 96.30%, duration = 2.921278(s)
Epoch [ 8/30], train loss = 0.329156, train accuracy = 96.50%, valid loss = 0.118670, valid accuracy = 96.70%, duration = 2.897367(s)
Epoch [ 9/30], train loss = 0.323447, train accuracy = 96.61%, valid loss = 0.137532, valid accuracy = 95.97%, duration = 2.909123(s)
Epoch [10/30], train loss = 0.318100, train accuracy = 96.70%, valid loss = 0.120556, valid accuracy = 96.42%, duration = 2.937132(s)
Epoch [11/30], train loss = 0.313230, train accuracy = 96.63%, valid loss = 0.121510, valid accuracy = 96.58%, duration = 2.952568(s)
Epoch [12/30], train loss = 0.307786, train accuracy = 96.72%, valid loss = 0.103266, valid accuracy = 97.07%, duration = 3.092739(s)
Epoch [13/30], train loss = 0.305087, train accuracy = 96.77%, valid loss = 0.108941, valid accuracy = 96.80%, duration = 3.018854(s)
Epoch [14/30], train loss = 0.301025, train accuracy = 96.79%, valid loss = 0.114664, valid accuracy = 96.58%, duration = 3.036496(s)
Epoch [15/30], train loss = 0.297327, train accuracy = 96.78%, valid loss = 0.114212, valid accuracy = 96.67%, duration = 2.950222(s)
Epoch [16/30], train loss = 0.292740, train accuracy = 96.88%, valid loss = 0.124984, valid accuracy = 96.33%, duration = 2.898228(s)
Epoch [17/30], train loss = 0.290700, train accuracy = 96.87%, valid loss = 0.114442, valid accuracy = 96.67%, duration = 2.918557(s)
Epoch [18/30], train loss = 0.289460, train accuracy = 96.86%, valid loss = 0.109719, valid accuracy = 96.72%, duration = 2.888240(s)
Epoch [19/30], train loss = 0.288528, train accuracy = 96.87%, valid loss = 0.106606, valid accuracy = 96.88%, duration = 2.892054(s)
Epoch [20/30], train loss = 0.283587, train accuracy = 96.96%, valid loss = 0.114686, valid accuracy = 96.73%, duration = 2.906942(s)
Epoch [21/30], train loss = 0.283997, train accuracy = 96.90%, valid loss = 0.102021, valid accuracy = 96.90%, duration = 2.872222(s)
Epoch [22/30], train loss = 0.283525, train accuracy = 96.91%, valid loss = 0.112170, valid accuracy = 96.52%, duration = 2.929481(s)
Epoch [23/30], train loss = 0.280738, train accuracy = 96.93%, valid loss = 0.114490, valid accuracy = 96.77%, duration = 2.903521(s)
Epoch [24/30], train loss = 0.279182, train accuracy = 96.97%, valid loss = 0.117636, valid accuracy = 96.68%, duration = 2.922241(s)
Epoch [25/30], train loss = 0.277963, train accuracy = 96.94%, valid loss = 0.107209, valid accuracy = 96.67%, duration = 3.052312(s)
Epoch [26/30], train loss = 0.277753, train accuracy = 96.95%, valid loss = 0.100862, valid accuracy = 97.12%, duration = 2.996952(s)
Epoch [27/30], train loss = 0.275652, train accuracy = 97.03%, valid loss = 0.111677, valid accuracy = 96.55%, duration = 2.908656(s)
Epoch [28/30], train loss = 0.274572, train accuracy = 97.06%, valid loss = 0.110290, valid accuracy = 96.82%, duration = 2.885839(s)
Epoch [29/30], train loss = 0.274166, train accuracy = 97.04%, valid loss = 0.111466, valid accuracy = 96.70%, duration = 2.928078(s)
Epoch [30/30], train loss = 0.274030, train accuracy = 96.92%, valid loss = 0.117929, valid accuracy = 96.65%, duration = 2.915659(s)
Duration for train : 90.562652(s)
<<< Train Finished >>>
Test Accraucy : 96.51%
'''
