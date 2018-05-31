import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-3_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
KEEP_PROB = 0.7
ALPHA = 0.0

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss (tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name = name)
    return output


def l2_loss (tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name = name)/2
    return output


def linear(tensor_op, output_dim, weight_decay = False, regularizer = None, with_W = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay = False, regularizer = None, keep_prob = 1.0, with_W = False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')
        dr = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return dr, W
        else:
            return dr


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

h1 = relu_layer(tensor_op=X, output_dim=256, keep_prob=keep_prob, name='Relu_Layer1')
h2 = relu_layer(tensor_op=h1, output_dim=128, keep_prob=keep_prob, name='Relu_Layer2')
logits = linear(tensor_op=h2, output_dim=NCLASS, name = 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + ALPHA*tf.reduce_mean(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X:x_train[mask[s:t],:], Y:y_train[mask[s:t],:], keep_prob:KEEP_PROB})
            acc_per_epoch += a
            loss_per_epoch += l
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        acc_per_epoch /= total_step * BATCH_SIZE
        loss_per_epoch /= total_step

        va, vl = sess.run([accuracy, loss], feed_dict={X:x_validation, Y:y_validation, keep_prob:1.0})
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

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, keep_prob : 1.0})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.312819, train accuracy = 90.35%, valid loss = 0.126521, valid accuracy = 96.17%, duration = 3.710264(s)
Epoch [ 2/30], train loss = 0.149720, train accuracy = 95.47%, valid loss = 0.096940, valid accuracy = 97.02%, duration = 3.258433(s)
Epoch [ 3/30], train loss = 0.121148, train accuracy = 96.38%, valid loss = 0.079734, valid accuracy = 97.57%, duration = 3.221418(s)
Epoch [ 4/30], train loss = 0.100140, train accuracy = 96.95%, valid loss = 0.079703, valid accuracy = 97.53%, duration = 3.137302(s)
Epoch [ 5/30], train loss = 0.090567, train accuracy = 97.17%, valid loss = 0.078638, valid accuracy = 97.73%, duration = 2.998726(s)
Epoch [ 6/30], train loss = 0.080966, train accuracy = 97.48%, valid loss = 0.066623, valid accuracy = 97.90%, duration = 2.969753(s)
Epoch [ 7/30], train loss = 0.072621, train accuracy = 97.69%, valid loss = 0.076788, valid accuracy = 97.87%, duration = 3.018790(s)
Epoch [ 8/30], train loss = 0.066658, train accuracy = 97.88%, valid loss = 0.065931, valid accuracy = 97.98%, duration = 2.955772(s)
Epoch [ 9/30], train loss = 0.061354, train accuracy = 97.99%, valid loss = 0.073321, valid accuracy = 97.92%, duration = 3.021498(s)
Epoch [10/30], train loss = 0.061373, train accuracy = 98.04%, valid loss = 0.059847, valid accuracy = 98.38%, duration = 2.986237(s)
Epoch [11/30], train loss = 0.055376, train accuracy = 98.16%, valid loss = 0.072418, valid accuracy = 98.13%, duration = 2.992047(s)
Epoch [12/30], train loss = 0.052671, train accuracy = 98.34%, valid loss = 0.068329, valid accuracy = 98.12%, duration = 3.002837(s)
Epoch [13/30], train loss = 0.050849, train accuracy = 98.40%, valid loss = 0.080192, valid accuracy = 97.75%, duration = 2.986106(s)
Epoch [14/30], train loss = 0.051021, train accuracy = 98.40%, valid loss = 0.065334, valid accuracy = 98.27%, duration = 2.985971(s)
Epoch [15/30], train loss = 0.047851, train accuracy = 98.44%, valid loss = 0.070276, valid accuracy = 98.02%, duration = 3.028434(s)
Epoch [16/30], train loss = 0.044744, train accuracy = 98.59%, valid loss = 0.077535, valid accuracy = 98.20%, duration = 2.962952(s)
Epoch [17/30], train loss = 0.045575, train accuracy = 98.62%, valid loss = 0.071361, valid accuracy = 98.18%, duration = 3.109983(s)
Epoch [18/30], train loss = 0.042420, train accuracy = 98.68%, valid loss = 0.069403, valid accuracy = 98.20%, duration = 2.978568(s)
Epoch [19/30], train loss = 0.042779, train accuracy = 98.66%, valid loss = 0.061270, valid accuracy = 98.25%, duration = 3.027381(s)
Epoch [20/30], train loss = 0.039934, train accuracy = 98.77%, valid loss = 0.070455, valid accuracy = 98.28%, duration = 3.038568(s)
Epoch [21/30], train loss = 0.038888, train accuracy = 98.77%, valid loss = 0.068841, valid accuracy = 98.30%, duration = 3.006185(s)
Epoch [22/30], train loss = 0.039107, train accuracy = 98.78%, valid loss = 0.075019, valid accuracy = 98.28%, duration = 3.017611(s)
Epoch [23/30], train loss = 0.036888, train accuracy = 98.77%, valid loss = 0.064057, valid accuracy = 98.33%, duration = 3.009549(s)
Epoch [24/30], train loss = 0.035244, train accuracy = 98.94%, valid loss = 0.070238, valid accuracy = 98.37%, duration = 3.124993(s)
Epoch [25/30], train loss = 0.038067, train accuracy = 98.80%, valid loss = 0.071509, valid accuracy = 98.20%, duration = 3.104492(s)
Epoch [26/30], train loss = 0.035442, train accuracy = 98.93%, valid loss = 0.074527, valid accuracy = 98.32%, duration = 3.149876(s)
Epoch [27/30], train loss = 0.035638, train accuracy = 98.96%, valid loss = 0.081290, valid accuracy = 98.05%, duration = 3.076856(s)
Epoch [28/30], train loss = 0.035824, train accuracy = 98.86%, valid loss = 0.067588, valid accuracy = 98.23%, duration = 3.115541(s)
Epoch [29/30], train loss = 0.034478, train accuracy = 98.93%, valid loss = 0.073909, valid accuracy = 98.33%, duration = 3.121961(s)
Epoch [30/30], train loss = 0.034731, train accuracy = 98.94%, valid loss = 0.076397, valid accuracy = 98.27%, duration = 3.056804(s)
Duration for train : 93.506084(s)
<<< Train Finished >>>
Test Accraucy : 98.29%
'''
