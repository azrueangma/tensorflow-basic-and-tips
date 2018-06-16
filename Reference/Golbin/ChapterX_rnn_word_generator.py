import tensorflow as tf
import numpy as np
import shutil
import os
from utils import lazy_property

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q','r','s','t','u','v','w','x','y','z']

num_dic = {n:i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = {'word', 'wood', 'deep', 'dive', 'pose', 'cold', 'cool', 'load', 'love', 'kiss', 'kind',
            'load', 'soup', 'less', 'door', 'bowl', 'flow', 'wolf', 'code', 'seek', 'fool', 'geek'}


def make_batch(seq_data):
    input_batch = []
    target_batch = []
    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)
    return input_batch, target_batch


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('n_hidden', 128, 'number of rnn units')
flags.DEFINE_integer('total_epoch', 50, 'number of training epochs')
flags.DEFINE_integer('seq_len', 3, 'sequence length')
flags.DEFINE_integer('dim_input', dic_len, 'dimension of input')
flags.DEFINE_integer('n_class', dic_len, 'number of classes')

flags.DEFINE_string("model_path", './model10_2/', "directory to save model")
flags.DEFINE_string("board_path", './board10_2', "for tensorboard")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if not os.path.exists(FLAGS.board_path):
    os.mkdir(FLAGS.board_path)
else:
    shutil.rmtree(FLAGS.board_path)


class Word_generator(object):
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.sess = tf.Session()
        self._build_net()
        self.hypothesis
        self.cost
        self.train_op
        self.predict
        self.accuracy
        self.merged


    def _build_net(self):
        with tf.variable_scope('model'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.seq_len, FLAGS.dim_input], name='X')
            self.Y = tf.placeholder(dtype=tf.int32, shape=[None], name='Y')
            self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, FLAGS.n_class), [-1, FLAGS.n_class], name='Y_one_hot')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            self.W = tf.get_variable(name='W', shape=[FLAGS.n_hidden, FLAGS.n_class], initializer=tf.glorot_uniform_initializer())
            self.b = tf.get_variable(name='b', shape=[FLAGS.n_class], initializer=tf.zeros_initializer())

            cell1 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=self.keep_prob)
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)

            multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
            outputs, states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            self.logits = tf.nn.bias_add(tf.matmul(outputs, self.W), self.b)


    @lazy_property
    def hypothesis(self):
        return tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.0)


    @lazy_property
    def cost(self):
        return -tf.reduce_mean(self.Y_one_hot*tf.log(self.hypothesis))


    @lazy_property
    def train_op(self):
        return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cost, global_step=self.global_step)


    @lazy_property
    def predict(self):
        return tf.argmax(self.hypothesis, axis=1)


    @lazy_property
    def accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.Y_one_hot, axis=1)), tf.float32))


    @lazy_property
    def merged(self):
        loss_scalar = tf.summary.scalar(name='loss_scalar', tensor=self.cost)
        acc_scalar = tf.summary.scalar(name='acc_scalar', tensor=self.accuracy)
        return tf.summary.merge_all()


    def fit(self, reuse_model=False):
        saver = tf.train.Saver(tf.global_variables())
        writer = tf.summary.FileWriter(FLAGS.board_path, self.sess.graph)
        if not reuse_model:
            shutil.rmtree(FLAGS.model_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        input_batch, target_batch = make_batch(seq_data)

        for epoch in range(FLAGS.total_epoch):
            a, l, m, _ = self.sess.run([self.accuracy, self.cost, self.merged, self.train_op],
                               feed_dict={self.X: input_batch, self.Y: target_batch, self.keep_prob:0.8})
            print('Global step : {:5d} Epoch : {:4d}, cost : {:.6f}, accuracy : {:.2%}'.
                  format(self.sess.run(self.global_step),epoch+1, l, a))

            writer.add_summary(m, global_step=epoch)
            saver.save(self.sess, FLAGS.model_path + "rnn_mnist", global_step=self.global_step)


    def evaluate(self, x_test, y_test):
        predict, accuracy_val = self.sess.run([self.predict, self.accuracy], feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob: 1.0})
        predict_words = []
        for idx, val in enumerate(seq_data):
            last_char = char_arr[predict[idx]]
            predict_words.append(val[:3]+last_char)

        print("\n=== 예측 결과 ===")
        print('입력값 : ', [w[:3] + ' ' for w in seq_data])
        print('예측값 : ', predict_words)
        print('정확도 : {:.2%}'.format(accuracy_val))


m = Word_generator()
m.fit()

input_batch, target_batch = make_batch(seq_data)
m.evaluate(input_batch, target_batch)

'''

=== 예측 결과 ===
입력값 :  ['flo ', 'see ', 'wor ', 'les ', 'loa ', 'pos ', 'cod ', 'kin ', 'bow ', 'col ', 
           'div ', 'coo ', 'sou ', 'lov ', 'foo ', 'doo ', 'gee ', 'wol ', 'woo ', 'dee ', 'kis ']

예측값 :  ['flow', 'seek', 'word', 'less', 'load', 'pose', 'code', 'kind', 'bowl', 'cold', 
           'dive', 'cool', 'soup', 'love', 'fool', 'door', 'geek', 'wolf', 'wood', 'deep', 'kiss']
정확도 : 100.00%

'''
