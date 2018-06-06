import tensorflow as tf
import numpy as np
import os
import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', "./model/", "model path")
flags.DEFINE_integer('input_dim', np.size(x_train, 1), "dimension of input")
flags.DEFINE_integer('nclass', len(np.unique(y_train)), "number of classes")
flags.DEFINE_integer('batch_size', 32, "batch size")
flags.DEFINE_integer('total_epoch', 5, "number of train epochs")
flags.DEFINE_integer('ntrain', len(x_train), "number of train samples")
flags.DEFINE_integer('nvalidation', len(x_validation), "number of validation samples")
flags.DEFINE_integer('ntest', len(x_test), "number of test samples")

print("The number of train samples : ", FLAGS.ntrain)
print("The number of validation samples : ", FLAGS.nvalidation)
print("The number of test samples : ", FLAGS.ntest)


def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name='h')
        return h


def relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name='h')
        return h


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, FLAGS.input_dim], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, FLAGS.nclass), [-1, FLAGS.nclass], name='Y_one_hot')

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, FLAGS.nclass, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1, name='predict')

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(input_tensor=tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32), name='accuracy')


#create saver
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
total_step = int(FLAGS.ntrain/FLAGS.batch_size)

with tf.Session() as sess:
    sess.run(init_op)
    print("<<< Train Start >>>")
    for epoch in range(FLAGS.total_epoch):
        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))
        for step in range(total_step):
            s = FLAGS.batch_size*step
            t = FLAGS.batch_size*(step+1)
            _ = sess.run(optim, feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:]})

        #save model per epoch
        if not os.path.exists(FLAGS.model_dir):
            os.mkdir(FLAGS.model_dir)
        saver.save(sess, FLAGS.model_dir + "my_model_{}/model".format(epoch))

    print("<<< Train Finished >>>")