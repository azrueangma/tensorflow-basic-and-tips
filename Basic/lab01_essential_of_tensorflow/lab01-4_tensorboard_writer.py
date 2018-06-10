#-*- coding: utf-8 -*-
import tensorflow as tf
import shutil
import os

BOARD_PATH = './board/lab01-2_board'
if os.path.exists(BOARD_PATH):
    shutil.rmtree(BOARD_PATH)

g = tf.Graph()


with g.as_default():
    # create tensor op
    a = tf.constant(value=3, dtype=tf.float32, shape=[], name='a')
    b = tf.constant(value=5, dtype=tf.float32, shape=[], name='b')
    c = tf.add(a,b, name='c')

    writer = tf.summary.FileWriter(BOARD_PATH)


with tf.Session(graph=g) as sess:
    writer.add_graph(sess.graph)
    print(sess.run(c))


'''
8.0

tensorboard --logdir=lab01_2_board
'''
