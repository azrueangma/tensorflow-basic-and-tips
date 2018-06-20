#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import shutil

#constant
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('total_epoch', 100, "the number of train epochs")
flags.DEFINE_integer('cpu_device', 0, 'the number of cpu device ')
flags.DEFINE_string('board_path', "./board/lab01-8_board", "board directory")
flags.DEFINE_string('model_path', "./model/", "model directory")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)

#load data
x_train = np.array([[3.5], [4.2], [3.8], [5.0], [5.1], [0.2], [1.3], [2.4], [1.5], [0.5]])
y_train = np.array([[3.3], [4.4], [4.0], [4.8], [5.6], [0.1], [1.2], [2.4], [1.7], [0.6]])

g = tf.Graph()


def linear_v1(input_op, output_dim):
    input_shape = input_op.get_shape().as_list()
    W = tf.Variable(tf.truncated_normal(shape=[input_shape[-1], output_dim]), name='W')
    b = tf.Variable(tf.zeros(shape=[output_dim]), name='b')
    output = tf.nn.bias_add(tf.multiply(input_op, W), b, name='output')
    return W, b, output


def linear_v2(input_op, output_dim):
    input_shape = input_op.get_shape().as_list()
    W = tf.get_variable(name='W', shape=[input_shape[-1], output_dim], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name='b', shape=[output_dim], initializer=tf.zeros_initializer())
    output = tf.nn.bias_add(tf.multiply(input_op, W), b, name='output')
    return W, b, output


def my_model1(input_op):
    with tf.variable_scope("layer1"):
        W1, b1, output1 = linear_v1(input_op, 1)
    with tf.variable_scope("layer1"):
        W2, b2, output2 = linear_v1(output1, 1)
    return W1, W2


def my_model2(input_op):
    with tf.variable_scope("layer2"):
        W1, b1, output1 = linear_v2(input_op, 1)
    with tf.variable_scope("layer2"):
        W2, b2, output2 = linear_v2(output1, 1)
    return W1, W2


X = tf.constant(x_train, dtype=tf.float32)

#Step 1 tf.Variable
W11, W12 = my_model1(X)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(W11)
print(W12)
print(sess.run(W11))
print(sess.run(W12))

'''
#When we call linear_v1 function twice, we get two different variables
<tf.Variable 'layer1/W:0' shape=(1, 1) dtype=float32_ref>
<tf.Variable 'layer1_1/W:0' shape=(1, 1) dtype=float32_ref>

[[0.7115113]]
[[-0.79587203]]
'''


#Step 2 tf.get_variable without defining reuse
#W21, W22 = my_model2(X)
'''
ValueError: 
Variable layer2/W already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? 
'''


#Step 3 add reuse=True
def modified_my_model2(input_op):
    with tf.variable_scope("layer2"):
        W1, b1, output1 = linear_v2(input_op, 1)
    with tf.variable_scope("layer2", reuse=True):
        W2, b2, output2 = linear_v2(output1, 1)

    return W1, W2


W31, W32 = modified_my_model2(X)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(W31)
print(W32)
print(sess.run(W31))
print(sess.run(W32))

'''
#We can reuse same variable
<tf.Variable 'layer2/W:0' shape=(1, 1) dtype=float32_ref>
<tf.Variable 'layer2/W:0' shape=(1, 1) dtype=float32_ref>

[[-0.8412895]]
[[-0.8412895]]
'''
