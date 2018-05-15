import tensorflow as tf
import shutil
import os

BOARD_PATH = './board/lab03_1_board'

a = tf.constant(value = 3, dtype = tf.float32, shape = [], name = 'a')
b = tf.constant(value = 5, dtype = tf.float32, shape = [], name = 'b')
c = tf.add(a,b, name = 'c')


if os.path.exists(BOARD_PATH):
    shutil.rmtree(BOARD_PATH)
writer = tf.summary.FileWriter(BOARD_PATH)

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    print(sess.run(c))

'''
tensorboard --logdir=lab03_1_board

'''
