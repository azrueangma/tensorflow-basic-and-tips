import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-2_board"
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

def l2_loss (tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name = name)/2
    return output

def linear(tensor_op, output_dim, weight_decay = None, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            wd = l2_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

def relu_layer(tensor_op, output_dim, weight_decay = None, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')

        if weight_decay:
            wd = l2_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(X, 256, ALPHA, 'Relu_Layer1')
h2 = relu_layer(h1, 128, ALPHA, 'Relu_Layer2')
logits = linear(h2, NCLASS, ALPHA, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + tf.reduce_sum(weight_decay_loss)
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
Epoch [ 1/30], train loss = 0.226197, train accuracy = 93.54%, valid loss = 0.099966, valid accuracy = 96.80%, duration = 3.656134(s)
Epoch [ 2/30], train loss = 0.101443, train accuracy = 97.26%, valid loss = 0.088978, valid accuracy = 97.03%, duration = 3.488001(s)
Epoch [ 3/30], train loss = 0.077703, train accuracy = 98.05%, valid loss = 0.072892, valid accuracy = 97.83%, duration = 3.544199(s)
Epoch [ 4/30], train loss = 0.063307, train accuracy = 98.56%, valid loss = 0.092558, valid accuracy = 97.40%, duration = 3.589426(s)
Epoch [ 5/30], train loss = 0.059579, train accuracy = 98.68%, valid loss = 0.069646, valid accuracy = 97.83%, duration = 3.938110(s)
Epoch [ 6/30], train loss = 0.051846, train accuracy = 98.99%, valid loss = 0.061025, valid accuracy = 98.23%, duration = 3.502334(s)
Epoch [ 7/30], train loss = 0.048884, train accuracy = 99.11%, valid loss = 0.076129, valid accuracy = 97.82%, duration = 3.876294(s)
Epoch [ 8/30], train loss = 0.047975, train accuracy = 99.18%, valid loss = 0.080856, valid accuracy = 97.73%, duration = 3.693492(s)
Epoch [ 9/30], train loss = 0.045384, train accuracy = 99.28%, valid loss = 0.085146, valid accuracy = 97.77%, duration = 3.627302(s)
Epoch [10/30], train loss = 0.043891, train accuracy = 99.35%, valid loss = 0.081083, valid accuracy = 97.90%, duration = 3.574210(s)
Epoch [11/30], train loss = 0.043285, train accuracy = 99.34%, valid loss = 0.083695, valid accuracy = 97.78%, duration = 3.887659(s)
Epoch [12/30], train loss = 0.042956, train accuracy = 99.36%, valid loss = 0.085780, valid accuracy = 97.82%, duration = 4.382089(s)
Epoch [13/30], train loss = 0.040046, train accuracy = 99.46%, valid loss = 0.091242, valid accuracy = 97.77%, duration = 3.547448(s)
Epoch [14/30], train loss = 0.041058, train accuracy = 99.44%, valid loss = 0.084975, valid accuracy = 98.00%, duration = 3.714199(s)
Epoch [15/30], train loss = 0.040997, train accuracy = 99.48%, valid loss = 0.096620, valid accuracy = 97.95%, duration = 3.560516(s)
Epoch [16/30], train loss = 0.038895, train accuracy = 99.57%, valid loss = 0.106020, valid accuracy = 97.58%, duration = 4.189506(s)
Epoch [17/30], train loss = 0.041071, train accuracy = 99.49%, valid loss = 0.100643, valid accuracy = 97.82%, duration = 3.837298(s)
Epoch [18/30], train loss = 0.040091, train accuracy = 99.53%, valid loss = 0.090689, valid accuracy = 98.02%, duration = 3.860063(s)
Epoch [19/30], train loss = 0.039613, train accuracy = 99.54%, valid loss = 0.077624, valid accuracy = 98.12%, duration = 4.662461(s)
Epoch [20/30], train loss = 0.036674, train accuracy = 99.62%, valid loss = 0.105119, valid accuracy = 97.78%, duration = 3.791937(s)
Epoch [21/30], train loss = 0.036790, train accuracy = 99.61%, valid loss = 0.086938, valid accuracy = 98.22%, duration = 3.590448(s)
Epoch [22/30], train loss = 0.037724, train accuracy = 99.62%, valid loss = 0.090642, valid accuracy = 97.90%, duration = 3.564460(s)
Epoch [23/30], train loss = 0.038573, train accuracy = 99.56%, valid loss = 0.100663, valid accuracy = 97.78%, duration = 3.481936(s)
Epoch [24/30], train loss = 0.036503, train accuracy = 99.59%, valid loss = 0.090659, valid accuracy = 98.02%, duration = 3.510936(s)
Epoch [25/30], train loss = 0.037931, train accuracy = 99.56%, valid loss = 0.101707, valid accuracy = 97.82%, duration = 3.540913(s)
Epoch [26/30], train loss = 0.035742, train accuracy = 99.63%, valid loss = 0.072663, valid accuracy = 98.18%, duration = 3.764390(s)
Epoch [27/30], train loss = 0.037134, train accuracy = 99.61%, valid loss = 0.100831, valid accuracy = 97.78%, duration = 3.602890(s)
Epoch [28/30], train loss = 0.035971, train accuracy = 99.63%, valid loss = 0.085062, valid accuracy = 97.80%, duration = 3.632649(s)
Epoch [29/30], train loss = 0.034198, train accuracy = 99.70%, valid loss = 0.107983, valid accuracy = 97.68%, duration = 3.763032(s)
Epoch [30/30], train loss = 0.036366, train accuracy = 99.63%, valid loss = 0.097953, valid accuracy = 97.65%, duration = 3.899226(s)
Duration for train : 113.272060(s)
<<< Train Finished >>>
Test Accraucy : 97.95%
'''
