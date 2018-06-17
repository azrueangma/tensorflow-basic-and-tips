import tensorflow as tf
import numpy as np
from load_mnist import save_and_load_mnist
import os
import shutil
from utils import lazy_property

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.001, "learning rate for optimization")

flags.DEFINE_integer("total_epoch", 30, "the number of training")
flags.DEFINE_integer("batch_size", 128, "the number of batch size")
flags.DEFINE_integer("dim_input", 28, "input dimension")
flags.DEFINE_integer("seq_length", 28, "length of sequence")
flags.DEFINE_integer("n_hidden", 128, "the number of hidden units")
flags.DEFINE_integer("n_class", 10, "the number of clases")

flags.DEFINE_string("model_path", './model10_1/', "directory to save model")
flags.DEFINE_string("board_path", './board10_1', "for tensorboard")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if not os.path.exists(FLAGS.board_path):
    os.mkdir(FLAGS.board_path)
else:
    shutil.rmtree(FLAGS.board_path)

dataset = save_and_load_mnist("./data/mnist/")

x_train = dataset['train_data']
x_train = np.reshape(x_train, [-1, 28, 28])
y_train = dataset['train_target']
x_test = dataset['test_data']
x_test = np.reshape(x_test, [-1, 28, 28])
y_test = dataset['test_target']


class RNN_MNIST(object):
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._build_model()
        self.hypothesis
        self.predict
        self.accuracy
        self.cost
        self.train_op
        self.merged


    def _build_model(self):
        with tf.variable_scope("model"):
            self.X = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.dim_input],name='X')
            self.Y = tf.placeholder(tf.int32, [None, 1],name='Y')
            self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, FLAGS.n_class), [-1, FLAGS.n_class], name='Y_one_hot')

            self.W = tf.get_variable(name='W', shape=[FLAGS.n_hidden, FLAGS.n_class], initializer=tf.glorot_uniform_initializer())
            self.b = tf.get_variable(name='b', shape=[FLAGS.n_class], initializer=tf.zeros_initializer())

            cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
            outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=self.X, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            self.outputs = outputs[-1]
            self.logits = tf.nn.bias_add(tf.matmul(self.outputs, self.W), self.b)


    def fit(self, reuse_model=False):
        total_steps = int(len(x_train) / FLAGS.batch_size)
        print(">>> Training Start [total epochs : {}, total step : {}]".format(FLAGS.total_epoch, total_steps))
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            writer = tf.summary.FileWriter(FLAGS.board_path, sess.graph)
            if not reuse_model:
                shutil.rmtree(FLAGS.model_path)
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.total_epoch):
                loss_per_epoch = 0
                acc_per_epoch = 0

                np.random.seed(epoch)
                mask = np.random.permutation(len(x_train))
                for step in range(total_steps):
                    s = step * FLAGS.batch_size
                    t = (step + 1) * FLAGS.batch_size
                    a, l, _ = sess.run([self.accuracy, self.cost, self.train_op],
                                       feed_dict = {self.X: x_train[mask[s:t]], self.Y: y_train[mask[s:t]]})
                    loss_per_epoch += l/total_steps
                    acc_per_epoch += a/total_steps

                print("Global Step : {:5d}, Epoch : [{:3d}/{:3d}], loss : {:.6f}, accuracy : {:.2%}"
                      .format(sess.run(self.global_step), epoch + 1, FLAGS.total_epoch, loss_per_epoch, acc_per_epoch))

                m = sess.run(self.merged, feed_dict={self.avg_acc:acc_per_epoch, self.avg_loss:loss_per_epoch})
                writer.add_summary(m, global_step=epoch)
                saver.save(sess, FLAGS.model_path+"rnn_mnist", global_step=self.global_step)

            te_a = sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})
            print("Test Accuracy : {:.2%}".format(te_a))


    @lazy_property
    def hypothesis(self):
        return tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.0, name='hypothesis')


    @lazy_property
    def predict(self):
        return tf.argmax(self.hypothesis, axis=1, name='predict')


    @lazy_property
    def accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.Y_one_hot, axis=1)), tf.float32),name='accuracy')


    @lazy_property
    def cost(self):
        return -tf.reduce_mean(self.Y_one_hot * tf.log(self.hypothesis), name='cost')


    @lazy_property
    def train_op(self):
        return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cost, global_step=self.global_step)


    @lazy_property
    def merged(self):
        self.avg_loss = tf.placeholder(tf.float32)
        avg_loss_scalar = tf.summary.scalar(name='avg_loss', tensor=self.avg_loss)
        self.avg_acc = tf.placeholder(tf.float32)
        avg_acc_scalar = tf.summary.scalar(name='avg_acc', tensor=self.avg_acc)
        return tf.summary.merge_all()


m = RNN_MNIST()
m.fit()

'''
Global Step :   375, Epoch : [  1/ 30], loss : 0.061568, accuracy : 80.01%
Global Step :   750, Epoch : [  2/ 30], loss : 0.018032, accuracy : 94.53%
Global Step :  1125, Epoch : [  3/ 30], loss : 0.012691, accuracy : 96.17%
Global Step :  1500, Epoch : [  4/ 30], loss : 0.009577, accuracy : 97.07%
Global Step :  1875, Epoch : [  5/ 30], loss : 0.008417, accuracy : 97.46%
Global Step :  2250, Epoch : [  6/ 30], loss : 0.007027, accuracy : 97.80%
Global Step :  2625, Epoch : [  7/ 30], loss : 0.005813, accuracy : 98.19%
Global Step :  3000, Epoch : [  8/ 30], loss : 0.004991, accuracy : 98.43%
Global Step :  3375, Epoch : [  9/ 30], loss : 0.004478, accuracy : 98.63%
Global Step :  3750, Epoch : [ 10/ 30], loss : 0.003979, accuracy : 98.75%
Global Step :  4125, Epoch : [ 11/ 30], loss : 0.003651, accuracy : 98.84%
Global Step :  4500, Epoch : [ 12/ 30], loss : 0.003020, accuracy : 99.06%
Global Step :  4875, Epoch : [ 13/ 30], loss : 0.002789, accuracy : 99.16%
Global Step :  5250, Epoch : [ 14/ 30], loss : 0.002574, accuracy : 99.17%
Global Step :  5625, Epoch : [ 15/ 30], loss : 0.002700, accuracy : 99.16%
Global Step :  6000, Epoch : [ 16/ 30], loss : 0.002221, accuracy : 99.32%
Global Step :  6375, Epoch : [ 17/ 30], loss : 0.002089, accuracy : 99.39%
Global Step :  6750, Epoch : [ 18/ 30], loss : 0.001976, accuracy : 99.38%
Global Step :  7125, Epoch : [ 19/ 30], loss : 0.001784, accuracy : 99.43%
Global Step :  7500, Epoch : [ 20/ 30], loss : 0.001935, accuracy : 99.40%
Global Step :  7875, Epoch : [ 21/ 30], loss : 0.001340, accuracy : 99.55%
Global Step :  8250, Epoch : [ 22/ 30], loss : 0.001647, accuracy : 99.47%
Global Step :  8625, Epoch : [ 23/ 30], loss : 0.001360, accuracy : 99.56%
Global Step :  9000, Epoch : [ 24/ 30], loss : 0.001219, accuracy : 99.62%
Global Step :  9375, Epoch : [ 25/ 30], loss : 0.001327, accuracy : 99.58%
Global Step :  9750, Epoch : [ 26/ 30], loss : 0.001217, accuracy : 99.62%
Global Step : 10125, Epoch : [ 27/ 30], loss : 0.001212, accuracy : 99.61%
Global Step : 10500, Epoch : [ 28/ 30], loss : 0.001071, accuracy : 99.67%
Global Step : 10875, Epoch : [ 29/ 30], loss : 0.001181, accuracy : 99.64%
Global Step : 11250, Epoch : [ 30/ 30], loss : 0.000743, accuracy : 99.79%

Test Accuracy : 98.60%
'''
