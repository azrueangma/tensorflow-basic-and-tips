#lab04_4 model
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

def sigmoid_linear(x, output_dim, name):
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

h1 = sigmoid_linear(X, 256, 'Sigmoid_Layer1')
h2 = sigmoid_linear(h1, 128, 'Sigmoid_Layer2')
h3 = sigmoid_linear(h2, 64, 'Sigmoid_Layer3')
logits = linear(h2, NCLASS, 'Linear_Layer')

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
Epoch [ 1/20], train loss = 1.549676, train accuracy = 58.53%, valid loss = 0.818122, valid accuracy = 74.25%, duration = 2.078436(s)
Epoch [ 2/20], train loss = 0.706034, train accuracy = 77.73%, valid loss = 0.613837, valid accuracy = 80.73%, duration = 1.966403(s)
Epoch [ 3/20], train loss = 0.561565, train accuracy = 82.53%, valid loss = 0.523901, valid accuracy = 83.78%, duration = 2.038207(s)
Epoch [ 4/20], train loss = 0.485609, train accuracy = 85.02%, valid loss = 0.470800, valid accuracy = 85.52%, duration = 1.953612(s)
Epoch [ 5/20], train loss = 0.436054, train accuracy = 86.68%, valid loss = 0.434648, valid accuracy = 86.75%, duration = 2.023713(s)
Epoch [ 6/20], train loss = 0.400169, train accuracy = 87.82%, valid loss = 0.407844, valid accuracy = 87.68%, duration = 1.864963(s)
Epoch [ 7/20], train loss = 0.372363, train accuracy = 88.68%, valid loss = 0.386834, valid accuracy = 88.25%, duration = 1.993824(s)
Epoch [ 8/20], train loss = 0.349782, train accuracy = 89.44%, valid loss = 0.369719, valid accuracy = 88.95%, duration = 1.785211(s)
Epoch [ 9/20], train loss = 0.330828, train accuracy = 90.08%, valid loss = 0.355378, valid accuracy = 89.27%, duration = 1.970941(s)
Epoch [10/20], train loss = 0.314535, train accuracy = 90.53%, valid loss = 0.343103, valid accuracy = 89.72%, duration = 1.893384(s)
Epoch [11/20], train loss = 0.300273, train accuracy = 90.93%, valid loss = 0.332413, valid accuracy = 89.98%, duration = 1.806373(s)
Epoch [12/20], train loss = 0.287602, train accuracy = 91.30%, valid loss = 0.322980, valid accuracy = 90.12%, duration = 1.704559(s)
Epoch [13/20], train loss = 0.276204, train accuracy = 91.65%, valid loss = 0.314570, valid accuracy = 90.30%, duration = 1.913650(s)
Epoch [14/20], train loss = 0.265851, train accuracy = 91.95%, valid loss = 0.307015, valid accuracy = 90.63%, duration = 1.817868(s)
Epoch [15/20], train loss = 0.256372, train accuracy = 92.28%, valid loss = 0.300185, valid accuracy = 90.98%, duration = 1.849123(s)
Epoch [16/20], train loss = 0.247639, train accuracy = 92.55%, valid loss = 0.293977, valid accuracy = 91.07%, duration = 1.744001(s)
Epoch [17/20], train loss = 0.239552, train accuracy = 92.82%, valid loss = 0.288307, valid accuracy = 91.30%, duration = 1.930087(s)
Epoch [18/20], train loss = 0.232033, train accuracy = 93.06%, valid loss = 0.283108, valid accuracy = 91.40%, duration = 1.852700(s)
Epoch [19/20], train loss = 0.225014, train accuracy = 93.29%, valid loss = 0.278323, valid accuracy = 91.43%, duration = 1.870939(s)
Epoch [20/20], train loss = 0.218441, train accuracy = 93.48%, valid loss = 0.273907, valid accuracy = 91.62%, duration = 1.923497(s)
Duration for train : 40.735464(s)
<<< Train Finished >>>
Test Accraucy : 91.55%
'''
