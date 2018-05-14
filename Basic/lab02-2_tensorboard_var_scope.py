import tensorflow as tf
import shutil
import os

BOARD_PATH = './board/lab02_2_board'

a = tf.constant(value = 3, dtype = tf.float32, shape = [], name = 'a')
b = tf.constant(value = 5, dtype = tf.float32, shape = [], name = 'b')

with tf.variable_scope("Add"):
    c = tf.add(a,b, name = 'c')

if os.path.exists(BOARD_PATH):
    shutil.rmtree(BOARD_PATH)
writer = tf.summary.FileWriter(BOARD_PATH)

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    print(sess.run(c))