import tensorflow as tf
import os

def search(dirname):
    filelist = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename.replace('\\', '/'))
    return filelist

def read_one_sample(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    tfrecord_features = tf.parse_single_example(serialized=serialized_example, features={
        'features' : tf.VarLenFeature(tf.float32),
        'label' : tf.FixedLenFeature([], tf.int64)
    })

    feature = tfrecord_features['features']
    feature = feature.values
    label = tfrecord_features['label']
    return feature, label

DEFAULT_DIR = "D:/STUDY/2018/MYT/tensorflow-basic-and-tips/Basics/data/lab04_7"

data_list = search(DEFAULT_DIR)

train_filename_queue = tf.train.string_input_producer(data_list)
x_mini_batch_op, y_mini_batch_op = read_one_sample(train_filename_queue)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(5):
        x_data, y_data = sess.run([x_mini_batch_op, y_mini_batch_op])
        print("x data : ", x_data)
        print("y_data : ",y_data)
        print()
    coord.request_stop()
    coord.join(threads)

'''

x
data: [0.   0.85 0.47 0.93 1.   1.   0.66 0.69 0.37 0.35 0.16 0.   0.   0.17
       0.41 0.26]
y_data: 7

x
data: [0.14 0.86 0.5  1.   0.73 0.8  0.39 0.53 0.   0.28 0.   0.04 0.58 0.01
       1.   0.]
y_data: 2

x
data: [0.68 0.66 1.   0.93 0.48 1.   0.   0.78 0.47 0.68 0.9  0.56 0.67 0.26
       0.29 0.]
y_data: 9

x
data: [0.33 0.83 0.01 0.71 0.   0.13 0.62 0.   1.   0.48 0.73 1.   0.16 0.77
       0.08 0.2]
y_data: 0

x
data: [0.   0.76 0.31 1.   0.67 0.82 0.47 0.44 0.13 0.13 0.05 0.01 0.52 0.04
       1.   0.]
y_data: 2

'''
