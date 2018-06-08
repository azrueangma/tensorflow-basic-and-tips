import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time
import functools

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', "./lab10-5_model/", """model path""")
flags.DEFINE_string('model_name',"my_model", """the name of model""")
flags.DEFINE_string('board_path', "./board/lab10-5_board", """board path""")
flags.DEFINE_float('init_learning_rate', 0.001, """initial learning rate""")
flags.DEFINE_integer('batch_size', 32, """batch size""")
flags.DEFINE_integer('total_epoch', 5, """number of train epochs""")

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)
    os.mkdir(FLAGS.board_path)
else:
    os.mkdir(FLAGS.board_path)

if os.path.exists(FLAGS.model_dir):
    shutil.rmtree(FLAGS.model_dir)
    os.mkdir(FLAGS.model_dir)
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


def lazy_property(function):
    attribute='_'+function.__name__
    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Model(object):
    def __init__(self, input_dim, nclass):
        tf.set_random_seed(0)
        self.sess = tf.Session()
        self.input_dim = input_dim
        self.nclass = nclass
        self._build_net()
        self.loss
        self.optimizer
        self.predict
        self.accuracy
        self.summary
        self.writer


    def _build_net(self):
        with tf.variable_scope(FLAGS.model_name):
            with tf.variable_scope("Inputs"):
                self.X = tf.placeholder(shape=[None, self.input_dim], dtype = tf.float32, name='X')
                self.Y = tf.placeholder(shape=[None, 1], dtype = tf.int32, name='Y')
                self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, self.nclass), [-1, self.nclass], name='Y_one_hot')
                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            self.h1 = relu_layer(tensor_op=self.X, output_dim=256, name='Relu_Layer1')
            self.h2 = relu_layer(tensor_op=self.h1, output_dim=128, name='Relu_Layer2')
            self.logits = linear(tensor_op=self.h2, output_dim=self.nclass, name='Linear_Layer')


    def fit(self, x_train, y_train, x_validation, y_validation):
        self.writer.add_graph(self.sess.graph)

        ntrain = len(x_train)
        total_step = int(ntrain / FLAGS.batch_size)
        print("Total step : ", total_step)
        self.sess.run(tf.global_variables_initializer())
        train_start_time = time.perf_counter()
        for epoch in range(FLAGS.total_epoch):
            loss_per_epoch = 0
            acc_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(ntrain)
            epoch_start_time = time.perf_counter()

            u = FLAGS.init_learning_rate
            for step in range(total_step):
                s = FLAGS.batch_size * step
                t = FLAGS.batch_size * (step + 1)
                l, a, _ = self.sess.run([self.loss, self.accuracy, self.optimizer],
                                        feed_dict={self.X:x_train[mask[s:t], :],
                                                   self.Y:y_train[mask[s:t], :],
                                                   self.learning_rate:u})
                loss_per_epoch += l
                acc_per_epoch += a
            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            loss_per_epoch /= total_step * FLAGS.batch_size
            acc_per_epoch /= total_step * FLAGS.batch_size

            vl, va = self.evaluate(x_validation, y_validation)
            epoch_valid_acc = va / len(x_validation)
            epoch_valid_loss = vl / len(x_validation)

            self.summary_log(loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch)
            self.save_model(epoch)

            u = u * 0.95
            if (epoch + 1) % 1 == 0:
                print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                      "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                      .format(epoch + 1, FLAGS.total_epoch, loss_per_epoch, acc_per_epoch, epoch_valid_loss,
                              epoch_valid_acc, epoch_duration))

        train_end_time = time.perf_counter()
        train_duration = train_end_time - train_start_time
        print("Duration for train : {:.6f}(s)".format(train_duration))
        print("<<< Train Finished >>>")


    def evaluate(self, x_test, y_test):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X:x_test, self.Y:y_test})


    def summary_log(self, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch):
        s = self.sess.run(self.summary, feed_dict={
            self.avg_train_loss: loss_per_epoch,
            self.avg_train_acc: acc_per_epoch,
            self.avg_validation_loss: epoch_valid_loss,
            self.avg_validation_acc: epoch_valid_acc})
        self.writer.add_summary(s, global_step=epoch)


    def save_model(self, epoch):
        self.saver.save(self.sess, FLAGS.model_dir + FLAGS.model_name+"_{}/model".format(epoch))


    @lazy_property
    def predict(self):
        return tf.argmax(self.logits, axis=1, name='prediction')


    @lazy_property
    def accuracy(self):
        is_correct = tf.cast(tf.equal(self.predict, tf.argmax(self.Y_one_hot, axis=1)), tf.float32)
        return tf.reduce_sum(is_correct, name='accuracy')


    @lazy_property
    def loss(self):
        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y_one_hot), name='loss')
        return loss


    @lazy_property
    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        opt = opt.minimize(self.loss)
        return opt


    @lazy_property
    def summary(self):
        self.avg_train_loss = tf.placeholder(tf.float32)
        loss_train_avg = tf.summary.scalar('avg_train_loss', self.avg_train_loss)
        self.avg_train_acc = tf.placeholder(tf.float32)
        acc_train_avg = tf.summary.scalar('avg_train_acc', self.avg_train_acc)
        self.avg_validation_loss = tf.placeholder(tf.float32)
        loss_validation_avg = tf.summary.scalar('avg_validation_loss', self.avg_validation_loss)
        self.avg_validation_acc = tf.placeholder(tf.float32)
        acc_validation_avg = tf.summary.scalar('avg_validation_acc', self.avg_validation_acc)
        self.merged = tf.summary.merge_all()
        return self.merged


    @lazy_property
    def writer(self):
        return tf.summary.FileWriter(FLAGS.board_path)


    @lazy_property
    def saver(self):
        return tf.train.Saver()


m = Model(input_dim, nclass)
m.fit(x_train, y_train, x_validation, y_validation)
tl, ta = m.evaluate(x_test, y_test)
print("Test loss : {:.6f} Test Accraucy : {:.2%}".format(tl / ntest, ta / ntest))
