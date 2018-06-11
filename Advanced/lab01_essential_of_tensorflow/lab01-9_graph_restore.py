#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

#load data
x_test = np.array([[0.7], [1.5], [8.0], [-1.0], [-0.4]])

#We restore our saved model to predict class of samples in test set
model_path = "./model"
model_list = os.listdir(model_path)

def path_join(p1, p2):
    return os.path.join(p1, p2).replace('\\','/')

#select last model
print(model_list)
selected_model_path = path_join(model_path, model_list[-1])
ckpt = tf.train.get_checkpoint_state(selected_model_path)

g = tf.Graph()
with tf.Session(graph=g) as sess:
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
    saver.restore(sess, ckpt.model_checkpoint_path)
    predict = sess.graph.get_tensor_by_name("Output/output:0")

    p = sess.run(predict, feed_dict={"Inputs/X:0":x_test})
    print(p)

'''
model list

['my_model_10', 'my_model_100', 'my_model_20', 'my_model_30', 'my_model_40', 'my_model_50', 
 'my_model_60', 'my_model_70', 'my_model_80', 'my_model_90']

 
predictions
 
 [[ 0.7713588 ]
 [ 1.4871737 ]
 [ 7.3031693 ]
 [-0.7497477 ]
 [-0.21288657]]
'''