import tensorflow as tf
import numpy as np
from utils import lazy_property
import shutil
import os

seq_data = [["word", "단어"],["wood", "나무"],["game", "놀이"],["girl", "소녀"],["kiss","키스"],["love","사랑"]]
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S'+seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, "learning rate")
flags.DEFINE_integer('dim_hidden', 128, "dimension of lstm hidden units")
flags.DEFINE_integer('total_epoch', 100, "total number of training epochs")
flags.DEFINE_integer('n_class', dic_len, "number of classes")
flags.DEFINE_integer('dim_input', dic_len, "dimension of input")

flags.DEFINE_string("model_path", './model10_2/', "directory to save model")
flags.DEFINE_string("board_path", './board10_2', "for tensorboard")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if not os.path.exists(FLAGS.board_path):
    os.mkdir(FLAGS.board_path)
else:
    shutil.rmtree(FLAGS.board_path)


class Model(object):
    def __init__(self, sess, seq_data, char_arr):
        self.sess = sess
        self.seq_data = seq_data
        self.char_arr = char_arr
        self._set_dataset()
        self._set_params()
        self._build_net()
        self.logits
        self.cost
        self.optimizer


    def _set_params(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')


    def _set_dataset(self):
        self.input_batch, self.output_batch, self.target_batch = make_batch(self.seq_data)


    def _build_net(self):
        self.enc_input = tf.placeholder(dtype=tf.float32, shape=[None, None, FLAGS.dim_input], name='X')
        self.dec_input = tf.placeholder(dtype=tf.float32, shape=[None, None, FLAGS.dim_input], name='Y')
        self.targets = tf.placeholder(dtype=tf.int64, shape=[None, None], name='T')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.dim_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=self.keep_prob, seed=0)
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        #decode의 state에 주목
        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.dim_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=self.keep_prob, seed=0)
            self.outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, initial_state=enc_states, dtype=tf.float32)


    @lazy_property
    def logits(self):
        return tf.layers.dense(self.outputs, FLAGS.n_class, activation=None)


    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))


    @lazy_property
    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost, global_step=self.global_step)


    def train(self, reuse_model=False):
        saver = tf.train.Saver(tf.global_variables())
        writer = tf.summary.FileWriter(FLAGS.board_path, self.sess.graph)
        if not reuse_model:
            shutil.rmtree(FLAGS.model_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.total_epoch):
            l, _ = self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.enc_input: self.input_batch,
                                            self.dec_input: self.output_batch,
                                            self.targets: self.target_batch,
                                            self.keep_prob: 0.5})

            print("Global_step : {:4d}, Epoch : {:4d}, Cost : {:.6f}".format(self.sess.run(self.global_step), epoch+1, l))

        print("Training Done")


    def translate(self, word):
        seq_data = [word, 'P'*len(word)]
        input_batch, output_batch, target_batch = make_batch([seq_data])
        prediction = tf.argmax(self.logits, axis=2)
        result = self.sess.run(prediction, feed_dict={self.enc_input: input_batch,
                                                      self.dec_input: output_batch,
                                                      self.targets: target_batch,
                                                      self.keep_prob: 1.0})

        decoded = [self.char_arr[i] for i in result[0]]
        if not 'E' in decoded:
            return 'None'
        end = decoded.index('E')
        translated = ''.join(decoded[:end])
        return translated


    def test(self, word_list):
        print("\n=== Translation Test ===")
        for i in  word_list:
            print("{} -> {}".format(i, self.translate(i)))


sess = tf.Session()
m = Model(sess, seq_data, char_arr)
m.train()
m.test(['word', 'wood', 'game', 'kiss', 'girl', 'love', 'abcd'])


'''
=== Translation Test ===
word -> 단어
wood -> 나무
game -> 놀이
kiss -> 키스
girl -> 소녀
love -> 사랑
abcd -> None
'''
