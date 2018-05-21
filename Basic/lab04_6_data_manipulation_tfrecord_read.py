import tensorflow as tf
import numpy as np

tfrecords_filename = "D:/STUDY/2018/MYT/tensorflow-basic-and-tips/Basics/data/pendigits.tfrecords"
filename_queue = tf.train.string_input_producer([tfrecords_filename])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

tfrecord_features = tf.parse_single_example(serialized=serialized_example, features={
    'features': tf.VarLenFeature(tf.float32),
    'label': tf.VarLenFeature(tf.int64),
})

feature = tfrecord_features['features']
label = tfrecord_features['label']

feature_value = tf.reshape(feature.values, [-1, 64])
label_value = tf.reshape(label.values, [-1, 1])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x_data, y_data = sess.run([feature_value, label_value])
    print(x_data)
    print(y_data)
    coord.request_stop()
    coord.join(threads)
    
'''
[[0.   0.96 0.52 ... 0.86 0.39 1.  ]
[0.   0.65 0.43 ... 0.07 1.   0.08]
[0.37 0.71 0.79 ... 0.37 0.   0.22]
...
[0.98 0.89 0.48 ... 0.   0.   0.19]
[0.3  0.97 0.81 ... 0.45 0.   0.69]
[0.   1.   0.15 ... 0.4  0.   0.35]]
[[7]
 [2]
 [0]
 ...
 [9]
 [0]
 [6]]
'''
