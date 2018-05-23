import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-3_board"
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

def relu_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
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
logits = linear(h3, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = -tf.reduce_sum(Y_one_hot*tf.log(hypothesis), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)

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
Total step :  1687
Epoch [ 1/20], train loss = 1.515222, train accuracy = 52.21%, valid loss = 0.915102, valid accuracy = 70.53%, duration = 1.964443(s)
Epoch [ 2/20], train loss = 0.784199, train accuracy = 75.15%, valid loss = 0.666037, valid accuracy = 79.08%, duration = 2.042315(s)
Epoch [ 3/20], train loss = 0.613002, train accuracy = 80.71%, valid loss = 0.556254, valid accuracy = 82.72%, duration = 1.933831(s)
Epoch [ 4/20], train loss = 0.525099, train accuracy = 83.56%, valid loss = 0.493048, valid accuracy = 84.63%, duration = 1.898109(s)
Epoch [ 5/20], train loss = 0.468705, train accuracy = 85.44%, valid loss = 0.451515, valid accuracy = 85.90%, duration = 1.991584(s)
Epoch [ 6/20], train loss = 0.428123, train accuracy = 86.82%, valid loss = 0.421495, valid accuracy = 87.02%, duration = 1.864280(s)
Epoch [ 7/20], train loss = 0.396928, train accuracy = 87.83%, valid loss = 0.398343, valid accuracy = 87.70%, duration = 1.751633(s)
Epoch [ 8/20], train loss = 0.371859, train accuracy = 88.61%, valid loss = 0.379755, valid accuracy = 88.37%, duration = 1.772500(s)
Epoch [ 9/20], train loss = 0.351030, train accuracy = 89.24%, valid loss = 0.364411, valid accuracy = 88.65%, duration = 1.918565(s)
Epoch [10/20], train loss = 0.333258, train accuracy = 89.87%, valid loss = 0.351444, valid accuracy = 89.12%, duration = 1.788214(s)
Epoch [11/20], train loss = 0.317770, train accuracy = 90.31%, valid loss = 0.340257, valid accuracy = 89.43%, duration = 1.929739(s)
Epoch [12/20], train loss = 0.304054, train accuracy = 90.72%, valid loss = 0.330438, valid accuracy = 89.77%, duration = 1.808813(s)
Epoch [13/20], train loss = 0.291748, train accuracy = 91.10%, valid loss = 0.321702, valid accuracy = 90.02%, duration = 2.129130(s)
Epoch [14/20], train loss = 0.280590, train accuracy = 91.48%, valid loss = 0.313851, valid accuracy = 90.15%, duration = 1.820798(s)
Epoch [15/20], train loss = 0.270380, train accuracy = 91.82%, valid loss = 0.306741, valid accuracy = 90.28%, duration = 1.952196(s)
Epoch [16/20], train loss = 0.260966, train accuracy = 92.08%, valid loss = 0.300266, valid accuracy = 90.70%, duration = 1.838728(s)
Epoch [17/20], train loss = 0.252226, train accuracy = 92.36%, valid loss = 0.294341, valid accuracy = 90.85%, duration = 1.948066(s)
Epoch [18/20], train loss = 0.244070, train accuracy = 92.61%, valid loss = 0.288899, valid accuracy = 91.10%, duration = 1.986315(s)
Epoch [19/20], train loss = 0.236423, train accuracy = 92.86%, valid loss = 0.283884, valid accuracy = 91.32%, duration = 2.066862(s)
Epoch [20/20], train loss = 0.229229, train accuracy = 93.09%, valid loss = 0.279246, valid accuracy = 91.63%, duration = 2.101965(s)
Duration for train : 41.157188(s)
<<< Train Finished >>>
Test Accraucy : 90.91%
'''
