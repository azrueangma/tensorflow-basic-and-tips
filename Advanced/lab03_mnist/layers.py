import tensorflow as tf

def linear(input_op, output_dim, name):
    with tf.variable_scope(name):
        input_shape = input_op.get_shape().as_list()
        W = tf.get_variable(name='W', shape=[input_shape[-1],output_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(tf.matmul(input_op, W), b, name='output')
    return output


def relu_layer(input_op, output_dim, name):
    with tf.variable_scope(name):
        input_shape = input_op.get_shape().as_list()
        W = tf.get_variable(name='W', shape=[input_shape[-1],output_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(tf.matmul(input_op, W), b, name='output')
        output = tf.nn.relu(output)
    return output
