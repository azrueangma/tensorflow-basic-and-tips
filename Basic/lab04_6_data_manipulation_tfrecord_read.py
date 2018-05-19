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
    coord.request_stop()
    coord.join(threads)

