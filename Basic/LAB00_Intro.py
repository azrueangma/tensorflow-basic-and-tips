#-*- coding: utf-8 -*-
import tensorflow as tf

#create tensor op
hello_str_tensor = tf.constant(dtype = tf.string, value = "Hello", name = "hello_str_tensor")

#session
with tf.Session() as sess:
    hello_str_value = sess.run(hello_str_tensor)
    print("{:s}".format(hello_str_value.decode("utf-8")))

'''
Hello
'''