import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', "./lab10-4_model/", """model path""")
flags.DEFINE_string('model_name',"my_model", """the name of model""")
flags.DEFINE_string('board_path', "./board/lab10-4_board", """board path""")
flags.DEFINE_float('init_learning_rate', 0.001, """initial learning rate""")
flags.DEFINE_integer('batch_size', 32, """batch size""")
flags.DEFINE_integer('total_epoch', 5, """number of train epochs""")

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)
else:
    os.mkdir(FLAGS.board_path)

if os.path.exists(FLAGS.model_dir):
    shutil.rmtree(FLAGS.model_dir)
else:
    os.mkdir(FLAGS.model_dir)

input_dim = np.size(x_train,1)
nclass = len(np.unique(y_train))
ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss(tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name=name)
    return output


def l2_loss(tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name=name) / 2
    return output


def linear(tensor_op, output_dim, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')
    return h


def relu_layer(tensor_op, output_dim, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='pre_op')
        h = tf.nn.relu(pre_activation, name='relu_op')
        return h


class Model(object):
    def __init__(self, sess, input_dim, nclass):
        self.sess = sess
        self.input_dim = input_dim
        self.nclass = nclass

    def build_net(self):
        tf.set_random_seed(0)
        with tf.variable_scope(FLAGS.model_name):
            with tf.variable_scope("Inputs"):
                self.X = tf.placeholder(shape=[None, self.input_dim], dtype = tf.float32, name='X')
                self.Y = tf.placeholder(shape=[None, 1], dtype = tf.int32, name='Y')
                Y_one_hot = tf.reshape(tf.one_hot(self.Y, self.nclass), [-1, self.nclass], name='Y_one_hot')
                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            self.h1 = relu_layer(tensor_op=self.X, output_dim=256, name='Relu_Layer1')
            self.h2 = relu_layer(tensor_op=self.h1, output_dim=128, name='Relu_Layer2')
            logits=linear(tensor_op=self.h2, output_dim=self.nclass, name='Linear_Layer')

        with tf.variable_scope("Optimization"):
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope("Prediction"):
            self.prediction = tf.argmax(logits, axis=1, name='prediction')

        with tf.variable_scope("Accuracy"):
            is_correct = tf.cast(tf.equal(self.prediction, tf.argmax(Y_one_hot, axis=1)), tf.float32)
            self.accuracy = tf.reduce_sum(is_correct, name='accuracy')

        with tf.variable_scope("Summary"):
            self.avg_train_loss = tf.placeholder(tf.float32)
            loss_train_avg  = tf.summary.scalar('avg_train_loss', self.avg_train_loss)
            self.avg_train_acc = tf.placeholder(tf.float32)
            acc_train_avg = tf.summary.scalar('avg_train_acc', self.avg_train_acc)
            self.avg_validation_loss = tf.placeholder(tf.float32)
            loss_validation_avg = tf.summary.scalar('avg_validation_loss', self.avg_validation_loss)
            self.avg_validation_acc = tf.placeholder(tf.float32)
            acc_validation_avg = tf.summary.scalar('avg_validation_acc', self.avg_validation_acc)
            self.merged = tf.summary.merge_all()

        with tf.variable_scope("Saver"):
            self.writer = tf.summary.FileWriter(FLAGS.board_path)
            self.writer.add_graph(self.sess.graph)
            self.saver = tf.train.Saver()


    def get_predict(self, x_test):
        p = self.sess.run(self.prediction, feed_dict={self.X:x_test})
        return p


    def get_accuracy(self, x_test, y_test):
        a = self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test})
        return a


    def train(self, x_train, y_train, u):
        return self.sess.run([self.loss, self.accuracy, self.optim], feed_dict={self.X:x_train, self.Y:y_train, self.learning_rate:u})


    def evaluate(self, x_test, y_test):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X:x_test, self.Y:y_test})


    def summary_log(self, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch):
        s = self.sess.run(self.merged, feed_dict={
            self.avg_train_loss: loss_per_epoch,
            self.avg_train_acc: acc_per_epoch,
            self.avg_validation_loss: epoch_valid_loss,
            self.avg_validation_acc: epoch_valid_acc})
        self.writer.add_summary(s, global_step=epoch)


    def save_model(self, epoch):
        self.saver.save(self.sess, FLAGS.model_dir + FLAGS.model_name+"_{}/model".format(epoch))


total_step = int(ntrain/FLAGS.batch_size)
print("Total step : ", total_step)

with tf.Session() as sess:
    m = Model(sess, input_dim, nclass)
    m.build_net()

    sess.run(tf.global_variables_initializer())
    train_start_time = time.perf_counter()
    for epoch in range(FLAGS.total_epoch):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))
        epoch_start_time = time.perf_counter()

        u = FLAGS.init_learning_rate
        for step in range(total_step):
            s = FLAGS.batch_size*step
            t = FLAGS.batch_size*(step+1)
            l, a, _ = m.train(x_train[mask[s:t],:], y_train[mask[s:t],:], u)
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*FLAGS.batch_size
        acc_per_epoch /= total_step*FLAGS.batch_size

        vl, va = m.evaluate(x_validation, y_validation)
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        m.summary_log(loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch)
        m.save_model(epoch)

        u = u*0.95
        if (epoch+1) % 1 == 0:
            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, FLAGS.total_epoch, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = m.get_accuracy(x_test, y_test)
    print("Test Accraucy : {:.2%}".format(ta/ntest))