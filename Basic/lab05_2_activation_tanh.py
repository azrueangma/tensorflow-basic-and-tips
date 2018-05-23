#lab04_4 model
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

h1 = tanh_linear(X, 256, 'Tanh_Layer1')
h2 = tanh_linear(h1, 128, 'Tanh_Layer2')
h3 = tanh_linear(h2, 64, 'Tanh_Layer3')
logits = linear(h3, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = -tf.reduce_sum(Y_one_hot*tf.log(hypothesis), name = 'loss')
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
Epoch [ 1/20], train loss = 2.330511, train accuracy = 53.05%, valid loss = 1.123457, valid accuracy = 67.92%, duration = 2.185681(s)
Epoch [ 2/20], train loss = 0.956510, train accuracy = 71.70%, valid loss = 0.802389, valid accuracy = 75.45%, duration = 2.213448(s)
Epoch [ 3/20], train loss = 0.737402, train accuracy = 77.13%, valid loss = 0.680420, valid accuracy = 78.43%, duration = 2.129831(s)
Epoch [ 4/20], train loss = 0.627928, train accuracy = 80.23%, valid loss = 0.617898, valid accuracy = 80.98%, duration = 2.128162(s)
Epoch [ 5/20], train loss = 0.553161, train accuracy = 82.62%, valid loss = 0.570661, valid accuracy = 82.02%, duration = 2.009247(s)
Epoch [ 6/20], train loss = 0.494826, train accuracy = 84.37%, valid loss = 0.539069, valid accuracy = 83.37%, duration = 1.965461(s)
Epoch [ 7/20], train loss = 0.447566, train accuracy = 86.00%, valid loss = 0.520272, valid accuracy = 83.78%, duration = 1.895197(s)
Epoch [ 8/20], train loss = 0.400437, train accuracy = 87.33%, valid loss = 0.502969, valid accuracy = 84.67%, duration = 1.856750(s)
Epoch [ 9/20], train loss = 0.361615, train accuracy = 88.70%, valid loss = 0.494930, valid accuracy = 84.90%, duration = 2.099399(s)
Epoch [10/20], train loss = 0.326963, train accuracy = 89.77%, valid loss = 0.483621, valid accuracy = 85.17%, duration = 1.868144(s)
Epoch [11/20], train loss = 0.297697, train accuracy = 90.87%, valid loss = 0.478946, valid accuracy = 85.65%, duration = 2.050505(s)
Epoch [12/20], train loss = 0.273342, train accuracy = 91.69%, valid loss = 0.479021, valid accuracy = 85.58%, duration = 1.879003(s)
Epoch [13/20], train loss = 0.253134, train accuracy = 92.52%, valid loss = 0.476423, valid accuracy = 86.03%, duration = 1.895476(s)
Epoch [14/20], train loss = 0.236056, train accuracy = 93.02%, valid loss = 0.479600, valid accuracy = 85.87%, duration = 1.831321(s)
Epoch [15/20], train loss = 0.220769, train accuracy = 93.59%, valid loss = 0.482388, valid accuracy = 86.05%, duration = 1.867083(s)
Epoch [16/20], train loss = 0.207462, train accuracy = 94.07%, valid loss = 0.480867, valid accuracy = 85.97%, duration = 1.872158(s)
Epoch [17/20], train loss = 0.196141, train accuracy = 94.43%, valid loss = 0.485083, valid accuracy = 85.93%, duration = 1.835785(s)
Epoch [18/20], train loss = 0.185797, train accuracy = 94.81%, valid loss = 0.487332, valid accuracy = 86.25%, duration = 1.813670(s)
Epoch [19/20], train loss = 0.177112, train accuracy = 95.08%, valid loss = 0.489973, valid accuracy = 86.15%, duration = 1.850892(s)
Epoch [20/20], train loss = 0.169117, train accuracy = 95.36%, valid loss = 0.493894, valid accuracy = 86.28%, duration = 1.838714(s)
Duration for train : 41.773893(s)
<<< Train Finished >>>
Test Accraucy : 85.94%
'''
