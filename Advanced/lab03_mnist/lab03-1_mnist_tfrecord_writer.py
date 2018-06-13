import tensorflow as tf
import load_mnist
import os

DEFAULT_SAVE_PATH = './data/mnist/'
mnist = input_mnist.save_and_load_mnist(DEFAULT_SAVE_PATH, as_image=False, seed =0, fraction_of_validation=0.2)

data_splits = ["train", "test", "validation"]
for d in range(len(data_splits)):
    print(">>> saving " + data_splits[d])
    images = mnist[data_splits[d]+"_data"]
    labels = mnist[data_splits[d]+"_target"]
    filename = os.path.join(DEFAULT_SAVE_PATH, data_splits[d]+".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(len(images)):
        image = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[28])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels[index])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
