#-*- coding: utf-8 -*-
import tensorflow as tf

#create graph object
g = tf.Graph()


with g.as_default():

    #create tensor op
    hello_str_tensor = tf.constant(dtype=tf.string, value="Hello", shape = [], name="hello_str_tensor")


#session
with tf.Session(graph=g) as sess:
    hello_str_value = sess.run(hello_str_tensor)
    print(hello_str_tensor)
    print("{:s}".format(hello_str_value.decode("utf-8")))

'''
Tensor("hello_str_tensor:0", shape=(), dtype=string)

Hello
'''
