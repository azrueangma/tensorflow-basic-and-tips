import tensorflow as tf
import numpy as np
import os
import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 5

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


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
    X = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

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
total_step = int(ntrain/BATCH_SIZE)

with tf.Session() as sess:
    sess.run(init_op)
    print("<<< Train Start >>>")
    for epoch in range(TOTAL_EPOCH):
        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))
        for step in range(total_step):
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            _ = sess.run(optim, feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:]})

        #save model per epoch
        saver.save(sess, MODEL_DIR + "my_model_{}/model".format(epoch))

    print("<<< Train Finished >>>")
