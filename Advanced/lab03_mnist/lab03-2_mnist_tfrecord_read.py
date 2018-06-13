import tensorflow as tf
import os

DEFAULT_SAVE_PATH = './data/mnist/'
filename = os.path.join(DEFAULT_SAVE_PATH, "train.tfrecords")
filename_queue = tf.train.string_input_producer([filename])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
)

image = tf.decode_raw(features['image'], tf.float32)
image.set_shape([784])
label = tf.cast(features['label'], tf.int32)
label = tf.reshape(label, [1])
images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=32, capacity=2000, min_after_dequeue=1000)



coord = tf.train.Coordinator()

with tf.Session() as sess:
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.local_variables_initializer())
    print(sess.run(labels_batch))

    coord.join(threads)
    sess.close()