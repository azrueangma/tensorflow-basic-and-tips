import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import save_and_load_mnist
import shutil

dataset = save_and_load_mnist("./data/mnist/")
reuse_model = False

x_train = dataset['train_data']
y_train = dataset['train_target']
x_test = dataset['test_data']
y_test = dataset['test_target']

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')

with tf.variable_scope("encoder") as scope:
    W_encode = tf.get_variable(name='W_encode', shape=[784, 256], initializer=tf.glorot_uniform_initializer())
    b_encode = tf.get_variable(name='b_encode', shape=[256], initializer=tf.zeros_initializer())
    encoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X, W_encode), b_encode), name='encoder')

#골빈의 코드에서 이 부분이 잘못되었다. 
#decoder에 사용되는 weight는 encoder에서 사용되는 weight의 transpose형태여야 한다. 
with tf.variable_scope("decoder") as scope:
    b_decode = tf.get_variable(name='b_decode', shape=[784], initializer=tf.zeros_initializer())
    scope.reuse_variables()
    W_decode = tf.transpose(W_encode, name='W_decode')

decoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(encoder, W_decode), b_decode), name='decoder')

cost = tf.reduce_mean(tf.square(X-decoder), name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)


total_epochs = 10
batch_size = 32
total_steps = int(len(x_train)/batch_size)
print(">>> Training Start [total epochs : {}, total step : {}]".format(total_epochs, total_steps))
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    if not reuse_model:
        shutil.rmtree('./model8')
    ckpt = tf.train.get_checkpoint_state('./model8')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        loss_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))
        for step in range(total_steps):
            s = step*batch_size
            t = (step+1)*batch_size
            c, _= sess.run([cost, train_op], feed_dict={X:x_train[mask[s:t]]})

            loss_per_epoch += c/total_steps

        print("Global Step : {:5d}, Epoch : [{:3d}/{:3d}], cost : {:.6f}"
              .format(sess.run(global_step), epoch+1, total_epochs, loss_per_epoch))

        saver.save(sess, "./model8/dnn.ckpt", global_step=global_step)


    #임의로 10개만 택하여 확인해본다.
    sample_size = 10
    np.random.seed(0)
    mask = np.random.choice(a=len(x_test), size=10, replace=False)
    samples = sess.run(decoder, feed_dict={X:x_test[mask]})

    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(x_test[mask[i]], (28, 28)), cmap=plt.cm.gray_r)
        ax[1][i].imshow(np.reshape(samples[i], (28, 28)), cmap=plt.cm.gray_r)

    plt.show()
