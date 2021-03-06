import tensorflow as tf
import os
import random
import shutil
import time

def search(dirname):
    filelist = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename.replace('\\', '/'))
    return filelist

def read_one_sample(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    tfrecord_features = tf.parse_single_example(serialized=serialized_example, features={
        'features' : tf.FixedLenFeature([16], tf.float32),
        'label' : tf.FixedLenFeature([], tf.int64)
    })

    feature = tfrecord_features['features']
    label = tf.reshape(tfrecord_features['label'], [1])
    return feature, label

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name='h')
        return h

def sigmoid_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name='h')
        return h

DEFAULT_DIR = "D:/STUDY/2018/MYT/tensorflow-basic-and-tips/Basics/data/lab04_7"
BOARD_PATH = "./board/lab04-9_board"
BATCH_SIZE = 32
TOTAL_EPOCH = 30
INPUT_DIM = 16
NCLASS = 10

data_list = search(DEFAULT_DIR)
random.seed(0)
random.shuffle(data_list)

nsamples = len(data_list)
ntrain = int(nsamples*0.7)
nvalidation = int(nsamples*0.1)
ntest = int(nsamples-ntrain-nvalidation)

train_list = data_list[:ntrain]
validation_list = data_list[ntrain:(ntrain+nvalidation)]
test_list = data_list[-ntest:]

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(0)

    with tf.variable_scope("Train_Queue"):
        train_filename_queue = tf.train.string_input_producer(train_list, shuffle=True, seed=0, num_epochs=None)
        x_mini_batch_op, y_mini_batch_op = read_one_sample(train_filename_queue)
        x_mini_batch, y_mini_batch = tf.train.batch([x_mini_batch_op, y_mini_batch_op], batch_size=BATCH_SIZE)

    with tf.variable_scope("Inputs"):
        '''
        #When using feed_dict
        X = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
        '''
        X = x_mini_batch
        Y = y_mini_batch
        Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')

    h1 = sigmoid_layer(X, 32, 'FC_Layer1')
    h2 = sigmoid_layer(h1, 16, 'FC_Layer2')
    logits = linear(h2, NCLASS, 'FC_Layer3')

    with tf.variable_scope("Optimization"):
        hypothesis = tf.nn.softmax(logits, name='hypothesis')
        loss = -tf.reduce_sum(Y_one_hot * tf.log(hypothesis), name='loss')
        optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.variable_scope("Pred_and_Acc"):
        predict = tf.argmax(hypothesis, axis=1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32))

    with tf.variable_scope("Summary"):
        avg_loss = tf.placeholder(tf.float32)
        loss_avg = tf.summary.scalar('avg_loss', avg_loss)
        avg_acc = tf.placeholder(tf.float32)
        acc_avg = tf.summary.scalar('avg_acc', avg_acc)
        merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    total_step = int(nsamples / BATCH_SIZE)
    print("Total step : ", total_step)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if os.path.exists(BOARD_PATH):
            shutil.rmtree(BOARD_PATH)
        writer = tf.summary.FileWriter(BOARD_PATH)
        writer.add_graph(sess.graph)

        sess.run(init_op)
        train_start_time = time.perf_counter()
        for epoch in range(TOTAL_EPOCH):
            loss_per_epoch = 0
            acc_per_epoch = 0

            epoch_start_time = time.perf_counter()
            for step in range(total_step):
                s = BATCH_SIZE * step
                t = BATCH_SIZE * (step + 1)
                '''
                #using feed_dict
                x_batch, y_batch = sess.run([x_mini_batch, y_mini_batch])
                a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_batch, Y: y_batch})
                '''
                #don't use feed_dict
                a, l, _ = sess.run([accuracy, loss, optim])
                loss_per_epoch += l
                acc_per_epoch += a
            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            loss_per_epoch /= total_step * BATCH_SIZE
            acc_per_epoch /= total_step * BATCH_SIZE

            s = sess.run(merged, feed_dict={avg_loss: loss_per_epoch, avg_acc: acc_per_epoch})
            writer.add_summary(s, global_step=epoch)
            if (epoch + 1) % 1 == 0:
                print("Epoch [{:3d}/{:3d}], train loss = {:.6f}, train accuracy = {:.2%}, duration = {:.6f}(s)".
                      format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_duration))

        train_end_time = time.perf_counter()
        train_duration = train_end_time - train_start_time
        print("Duration for train : {:.6f}(s)".format(train_duration))
        print("<<< Train Finished >>>")
        coord.request_stop()
        coord.join(threads)

'''
Epoch [100/1000], train loss = 0.175064, train accuracy = 95.70%, duration = 1.234136(s)
Epoch [200/1000], train loss = 0.088415, train accuracy = 98.06%, duration = 1.491218(s)
Epoch [300/1000], train loss = 0.063637, train accuracy = 98.58%, duration = 1.233490(s)
Epoch [400/1000], train loss = 0.045786, train accuracy = 99.07%, duration = 1.403404(s)
Epoch [500/1000], train loss = 0.036937, train accuracy = 99.32%, duration = 1.429733(s)
Epoch [600/1000], train loss = 0.030345, train accuracy = 99.44%, duration = 1.584678(s)
Epoch [700/1000], train loss = 0.026781, train accuracy = 99.57%, duration = 1.696624(s)
Epoch [800/1000], train loss = 0.023470, train accuracy = 99.63%, duration = 1.308650(s)
Epoch [900/1000], train loss = 0.020309, train accuracy = 99.68%, duration = 1.369758(s)
Epoch [1000/1000], train loss = 0.015571, train accuracy = 99.80%, duration = 1.637397(s)
Duration for train : 1541.970177(s)
<<< Train Finished >>>
'''
