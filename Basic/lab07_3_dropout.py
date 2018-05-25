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


def linear(tensor_op, output_dim, weight_decay = None, regularizer = None, with_W = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay = None, regularizer = None, keep_prob = 1.0, with_W = False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')
        h = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

h1 = relu_layer(X, 256, keep_prob = keep_prob, name = 'Relu_Layer1')
h2 = relu_layer(h1, 128, keep_prob = keep_prob, name = 'Relu_Layer2')
logits = linear(h2, NCLASS, name = 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + tf.reduce_sum(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:], keep_prob : KEEP_PROB})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*BATCH_SIZE
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, keep_prob : 1.0})
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

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, keep_prob : 1.0})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.315420, train accuracy = 90.41%, valid loss = 0.117097, valid accuracy = 96.27%, duration = 3.759319(s)
Epoch [ 2/30], train loss = 0.150673, train accuracy = 95.44%, valid loss = 0.089674, valid accuracy = 97.15%, duration = 3.374273(s)
Epoch [ 3/30], train loss = 0.119275, train accuracy = 96.37%, valid loss = 0.079328, valid accuracy = 97.48%, duration = 3.287235(s)
Epoch [ 4/30], train loss = 0.098761, train accuracy = 96.96%, valid loss = 0.084369, valid accuracy = 97.52%, duration = 3.339211(s)
Epoch [ 5/30], train loss = 0.088286, train accuracy = 97.31%, valid loss = 0.070142, valid accuracy = 97.65%, duration = 3.214132(s)
Epoch [ 6/30], train loss = 0.080282, train accuracy = 97.52%, valid loss = 0.067098, valid accuracy = 97.97%, duration = 3.265347(s)
Epoch [ 7/30], train loss = 0.074805, train accuracy = 97.55%, valid loss = 0.070007, valid accuracy = 98.13%, duration = 3.197694(s)
Epoch [ 8/30], train loss = 0.068578, train accuracy = 97.83%, valid loss = 0.065617, valid accuracy = 97.95%, duration = 3.273621(s)
Epoch [ 9/30], train loss = 0.062353, train accuracy = 98.03%, valid loss = 0.059050, valid accuracy = 98.15%, duration = 3.195422(s)
Epoch [10/30], train loss = 0.061114, train accuracy = 98.06%, valid loss = 0.061511, valid accuracy = 98.27%, duration = 3.283611(s)
Epoch [11/30], train loss = 0.055844, train accuracy = 98.25%, valid loss = 0.064871, valid accuracy = 98.27%, duration = 3.262858(s)
Epoch [12/30], train loss = 0.054371, train accuracy = 98.30%, valid loss = 0.060114, valid accuracy = 98.33%, duration = 3.260496(s)
Epoch [13/30], train loss = 0.055946, train accuracy = 98.28%, valid loss = 0.064533, valid accuracy = 98.38%, duration = 3.209184(s)
Epoch [14/30], train loss = 0.047963, train accuracy = 98.43%, valid loss = 0.064354, valid accuracy = 98.43%, duration = 3.247119(s)
Epoch [15/30], train loss = 0.047506, train accuracy = 98.50%, valid loss = 0.065541, valid accuracy = 98.47%, duration = 3.260134(s)
Epoch [16/30], train loss = 0.048143, train accuracy = 98.43%, valid loss = 0.070454, valid accuracy = 98.03%, duration = 3.266058(s)
Epoch [17/30], train loss = 0.042599, train accuracy = 98.65%, valid loss = 0.068914, valid accuracy = 98.32%, duration = 3.322012(s)
Epoch [18/30], train loss = 0.042397, train accuracy = 98.66%, valid loss = 0.068305, valid accuracy = 98.25%, duration = 3.251673(s)
Epoch [19/30], train loss = 0.039463, train accuracy = 98.73%, valid loss = 0.061266, valid accuracy = 98.45%, duration = 3.246617(s)
Epoch [20/30], train loss = 0.042149, train accuracy = 98.72%, valid loss = 0.066077, valid accuracy = 98.37%, duration = 3.264390(s)
Epoch [21/30], train loss = 0.037433, train accuracy = 98.75%, valid loss = 0.065439, valid accuracy = 98.53%, duration = 3.220554(s)
Epoch [22/30], train loss = 0.038973, train accuracy = 98.77%, valid loss = 0.070272, valid accuracy = 98.47%, duration = 3.253425(s)
Epoch [23/30], train loss = 0.038366, train accuracy = 98.77%, valid loss = 0.068326, valid accuracy = 98.30%, duration = 3.289393(s)
Epoch [24/30], train loss = 0.040648, train accuracy = 98.75%, valid loss = 0.063163, valid accuracy = 98.47%, duration = 3.286112(s)
Epoch [25/30], train loss = 0.036811, train accuracy = 98.86%, valid loss = 0.067243, valid accuracy = 98.32%, duration = 3.203410(s)
Epoch [26/30], train loss = 0.034989, train accuracy = 98.91%, valid loss = 0.071312, valid accuracy = 98.35%, duration = 3.274750(s)
Epoch [27/30], train loss = 0.037312, train accuracy = 98.84%, valid loss = 0.065816, valid accuracy = 98.53%, duration = 3.308699(s)
Epoch [28/30], train loss = 0.032378, train accuracy = 98.96%, valid loss = 0.066437, valid accuracy = 98.50%, duration = 3.935454(s)
Epoch [29/30], train loss = 0.036182, train accuracy = 98.91%, valid loss = 0.072966, valid accuracy = 98.38%, duration = 3.603243(s)
Epoch [30/30], train loss = 0.032209, train accuracy = 98.96%, valid loss = 0.069240, valid accuracy = 98.45%, duration = 3.640791(s)
Duration for train : 101.165612(s)
<<< Train Finished >>>
Test Accraucy : 98.25%
'''
