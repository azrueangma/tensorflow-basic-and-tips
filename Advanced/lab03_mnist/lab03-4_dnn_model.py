#-*- coding: utf-8 -*-
import tensorflow as tf
import os
import shutil
from utils import lazy_property
from layers import linear
from layers import relu_layer

#constant
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('total_epoch', 3, "the number of train epochs")
flags.DEFINE_integer('cpu_device', 0, 'the number of cpu device ')
flags.DEFINE_integer('batch_size', 32, "the number of batch size")
flags.DEFINE_integer('nclass', 10, "the number of classes")
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for optimization')
flags.DEFINE_string('board_path', "./board/lab03-4_board", "board directory")
flags.DEFINE_string('model_path', "./model/", "model directory")
flags.DEFINE_string('data_path', "./data/mnist/", "data directory")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)


class Model(object):
    def __init__(self, g, seed):
        self.graph = g
        self.seed = seed
        with self.graph.as_default(), self.graph.device('/cpu:{}'.format(FLAGS.cpu_device)):
            tf.set_random_seed(self.seed)
            self.test_queue
            self.train_queue
            self.test_select
            self._build_model()
            self.loss
            self.predict
            self.accuracy
            self.optim
            self.writer
            self.saver
            self.merged


    def _build_model(self):
        with tf.variable_scope("Inputs"):
            #create placeholder for input
            self.use_test_set = tf.placeholder(tf.bool, name='use_test_set')
            [self.X, self.Y] = tf.cond(tf.equal(self.use_test_set, tf.constant(True)), lambda:self.test_select, lambda:self.train_queue)
            self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, FLAGS.nclass), [-1, FLAGS.nclass], name='Y_one_hot')

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.board_path = tf.placeholder(tf.string, name='board_path')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.l1 = relu_layer(input_op=self.X, output_dim=256, name="Layer1")
        self.l2 = relu_layer(input_op=self.l1, output_dim=128, name="Layer2")
        self.output = linear(input_op=self.l2, output_dim=10, name="Linear")


    @lazy_property
    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.Y_one_hot))


    @lazy_property
    def predict(self):
        return tf.argmax(self.output, 1)


    @lazy_property
    def accuracy(self):
        is_correct = tf.equal(self.predict, tf.argmax(self.Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        return accuracy


    @lazy_property
    def optim(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optim = optimizer.minimize(self.loss, global_step=self.global_step)
        return optim


    @lazy_property
    def writer(self):
        return tf.summary.FileWriter(FLAGS.board_path)


    @lazy_property
    def saver(self):
        return tf.train.Saver(tf.global_variables())


    @lazy_property
    def merged(self):
        self.avg_loss = tf.placeholder(tf.float32)
        self.avg_loss_scalar = tf.summary.scalar('loss_scalar', self.avg_loss)
        self.avg_acc = tf.placeholder(tf.float32)
        self.avg_acc_scalar = tf.summary.scalar('acc_scalar', self.avg_acc)
        return tf.summary.merge_all()

    @lazy_property
    def train_queue(self):
        filename = os.path.join(FLAGS.data_path, "train.tfrecords")
        self.n_train = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
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
        images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=FLAGS.batch_size,
                                                            capacity=2000, min_after_dequeue=1000)
        return images_batch, labels_batch


    @lazy_property
    def validation_queue(self):
        filename = os.path.join(FLAGS.data_path, "validation.tfrecords")
        self.n_validation = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
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
        images_batch, labels_batch = tf.train.batch([image, label], batch_size=FLAGS.batch_size, allow_smaller_final_batch=True)
        return images_batch, labels_batch

    @lazy_property
    def test_queue(self):
        filename = os.path.join(FLAGS.data_path, "test.tfrecords")
        self.n_test = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
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
        images_batch, labels_batch = tf.train.batch([image, label], batch_size=self.n_test)
        return images_batch, labels_batch


    @lazy_property
    def test_select(self):
        self.use_placeholder = tf.placeholder(tf.bool, name='use_placeholder')
        return tf.cond(tf.equal(self.use_placeholder, tf.constant(True)), self.external_input_placeholder,
                       lambda: self.test_queue)


    def external_input_placeholder(self):
        self.X2 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X2')
        self.Y2 = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='Y2')
        return self.X2, self.Y2


    def fit(self, model_reuse=False):
        with tf.Session(graph=self.graph) as sess:
            self.writer.add_graph(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.local_variables_initializer())

            if model_reuse == False:
                shutil.rmtree(FLAGS.model_path)
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            total_steps = int(self.n_train/FLAGS.batch_size)
            for epoch in range(FLAGS.total_epoch):
                loss_per_loss = 0
                loss_per_acc = 0
                for step in range(total_steps):
                    l, a, _ = sess.run([self.loss, self.accuracy, self.optim],
                                       feed_dict={self.use_test_set:False, self.use_placeholder:False, self.learning_rate:FLAGS.learning_rate})
                    loss_per_loss+=l/total_steps
                    loss_per_acc+=a/total_steps

                m = sess.run(self.merged, feed_dict={self.avg_loss:loss_per_loss, self.avg_acc:loss_per_acc})
                self.writer.add_summary(m, global_step=epoch)
                if (epoch+1) % 1 == 0:
                    print("Epoch [{:3d}/{:3d}], Global_step : {}, accuracy : {:.2%}, loss : {:.6f}".
                          format(epoch+1, FLAGS.total_epoch, sess.run(self.global_step), loss_per_acc, loss_per_loss))
                    # save model per epoch
                    self.saver.save(sess, FLAGS.model_path+"mnist_dnn.ckpt", global_step=self.global_step)

            '''
            test_total_steps = int(self.n_test/FLAGS.batch_size)+1
            test_loss = 0
            test_acc = 0
            '''
            l, a = sess.run([self.loss, self.accuracy], feed_dict={self.use_test_set:True, self.use_placeholder:False})

            print("Test accuracy : {:.2%}, loss : {:.6f}".format(a/self.n_test, l/self.n_test))

            coord.request_stop()
            coord.join(threads)


    def evaluation(self, x_test, y_test):
        with tf.Session(graph=self.graph) as sess:
            a, c = sess.run([self.accuracy, self.loss],
                            feed_dict={self.use_placeholder:True, self.X2:x_test, self.Y2:y_test})
            return a, c


    def prediction(self, x_test):
        with tf.Session(graph=self.graph) as sess:
            p = sess.run(self.predict, feed_dict={self.use_placeholder:True, self.X2:x_test})
            return p


g = tf.Graph()
m = Model(g, seed=0)
m.fit()
