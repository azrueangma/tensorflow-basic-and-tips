import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-3_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30

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

def relu_layer(x, output_dim, name):
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

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 2.324212, train accuracy = 12.94%, valid loss = 2.291710, valid accuracy = 15.33%, duration = 2.878509(s)
Epoch [ 2/30], train loss = 2.256144, train accuracy = 18.65%, valid loss = 2.226120, valid accuracy = 21.82%, duration = 2.586123(s)
Epoch [ 3/30], train loss = 2.194315, train accuracy = 25.67%, valid loss = 2.165385, valid accuracy = 29.47%, duration = 2.513523(s)
Epoch [ 4/30], train loss = 2.135805, train accuracy = 33.14%, valid loss = 2.107035, valid accuracy = 37.03%, duration = 2.793613(s)
Epoch [ 5/30], train loss = 2.079186, train accuracy = 40.11%, valid loss = 2.050205, valid accuracy = 43.25%, duration = 2.783064(s)
Epoch [ 6/30], train loss = 2.023594, train accuracy = 46.27%, valid loss = 1.993949, valid accuracy = 49.35%, duration = 2.755593(s)
Epoch [ 7/30], train loss = 1.968328, train accuracy = 51.50%, valid loss = 1.937953, valid accuracy = 54.55%, duration = 2.616684(s)
Epoch [ 8/30], train loss = 1.912895, train accuracy = 56.04%, valid loss = 1.881596, valid accuracy = 58.45%, duration = 2.399674(s)
Epoch [ 9/30], train loss = 1.857125, train accuracy = 59.63%, valid loss = 1.824895, valid accuracy = 61.52%, duration = 2.493883(s)
Epoch [10/30], train loss = 1.800968, train accuracy = 62.68%, valid loss = 1.767962, valid accuracy = 64.43%, duration = 2.417582(s)
Epoch [11/30], train loss = 1.744744, train accuracy = 65.14%, valid loss = 1.710843, valid accuracy = 66.60%, duration = 2.455332(s)
Epoch [12/30], train loss = 1.688184, train accuracy = 67.29%, valid loss = 1.653516, valid accuracy = 68.27%, duration = 2.411615(s)
Epoch [13/30], train loss = 1.631590, train accuracy = 69.12%, valid loss = 1.596498, valid accuracy = 69.97%, duration = 2.475331(s)
Epoch [14/30], train loss = 1.575233, train accuracy = 70.63%, valid loss = 1.539853, valid accuracy = 71.77%, duration = 2.402578(s)
Epoch [15/30], train loss = 1.519431, train accuracy = 72.08%, valid loss = 1.484279, valid accuracy = 72.83%, duration = 2.512087(s)
Epoch [16/30], train loss = 1.465035, train accuracy = 73.34%, valid loss = 1.430156, valid accuracy = 73.88%, duration = 2.378369(s)
Epoch [17/30], train loss = 1.411922, train accuracy = 74.51%, valid loss = 1.377589, valid accuracy = 74.90%, duration = 2.382713(s)
Epoch [18/30], train loss = 1.360449, train accuracy = 75.48%, valid loss = 1.326653, valid accuracy = 75.90%, duration = 2.428242(s)
Epoch [19/30], train loss = 1.310733, train accuracy = 76.34%, valid loss = 1.277568, valid accuracy = 76.82%, duration = 2.387531(s)
Epoch [20/30], train loss = 1.262815, train accuracy = 77.18%, valid loss = 1.230377, valid accuracy = 77.65%, duration = 2.362875(s)
Epoch [21/30], train loss = 1.216714, train accuracy = 77.96%, valid loss = 1.185209, valid accuracy = 78.40%, duration = 2.388923(s)
Epoch [22/30], train loss = 1.172825, train accuracy = 78.67%, valid loss = 1.142424, valid accuracy = 79.15%, duration = 2.389596(s)
Epoch [23/30], train loss = 1.131260, train accuracy = 79.44%, valid loss = 1.101639, valid accuracy = 79.70%, duration = 2.381375(s)
Epoch [24/30], train loss = 1.091671, train accuracy = 80.03%, valid loss = 1.063140, valid accuracy = 80.13%, duration = 2.491189(s)
Epoch [25/30], train loss = 1.054193, train accuracy = 80.58%, valid loss = 1.026673, valid accuracy = 80.78%, duration = 2.401799(s)
Epoch [26/30], train loss = 1.018845, train accuracy = 81.14%, valid loss = 0.992350, valid accuracy = 81.38%, duration = 2.417252(s)
Epoch [27/30], train loss = 0.985396, train accuracy = 81.65%, valid loss = 0.960025, valid accuracy = 81.97%, duration = 2.340698(s)
Epoch [28/30], train loss = 0.953900, train accuracy = 82.02%, valid loss = 0.929566, valid accuracy = 82.35%, duration = 2.364229(s)
Epoch [29/30], train loss = 0.924172, train accuracy = 82.41%, valid loss = 0.900739, valid accuracy = 82.75%, duration = 2.390223(s)
Epoch [30/30], train loss = 0.896194, train accuracy = 82.81%, valid loss = 0.873651, valid accuracy = 83.00%, duration = 2.384416(s)
Duration for train : 75.372241(s)
<<< Train Finished >>>
Test Accraucy : 83.96%
'''
