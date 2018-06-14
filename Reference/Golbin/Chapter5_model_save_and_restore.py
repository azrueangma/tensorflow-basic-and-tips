import tensorflow as tf
import numpy as np
from load_mnist import save_and_load_mnist

dataset = save_and_load_mnist("./data/mnist/")

x_train = dataset['train_data']
y_train = dataset['train_target']
x_test = dataset['test_data']
y_test = dataset['test_target']

#global step의 경우, 0으로 초기화하고 train가능하지 않게 설정한다.
#global step은 optimizer가 학습한 횟수를 의미한다. 변수로서 계속 변한다.
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
Y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='Y')
Y_one_hot = tf.reshape(tf.one_hot(Y, 10), [-1, 10], name='Y_one_hot')

W1 = tf.get_variable(name='W1', shape=[784,256], initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable(name='b1', shape=[256], initializer=tf.zeros_initializer())
h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W1), b1), name='h1')

W2 = tf.get_variable(name='W2', shape=[256,128], initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable(name='b2', shape=[128], initializer=tf.zeros_initializer())
h2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h1, W2), b2), name='h2')

W3 = tf.get_variable(name='W3', shape=[128, 10], initializer=tf.glorot_uniform_initializer())
b3 = tf.get_variable(name='b3', shape=[10], initializer=tf.zeros_initializer())

output = tf.nn.bias_add(tf.matmul(h2, W3), b3, name='output')

#hypothesis에서 확률이 0이 되는 것을 방지하기 위하여 1e-10이하의 값은 1e-10으로 대체한다.
hypothesis = tf.clip_by_value(tf.nn.softmax(output), 1e-10, 1.0)
cost = -tf.reduce_mean(Y_one_hot*tf.log(hypothesis))

predict = tf.argmax(hypothesis, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

total_epochs = 10
batch_size = 32
total_steps = int(len(x_train)/batch_size)
print(">>> Training Start [total epochs : {}, total step : {}]".format(total_epochs, total_steps))
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    #model5 path안의 ckpt파일들을 체크한다.
    #만일, 모델을 다시 학습하고 싶으면
    #shutil.rmtree를 이용하여 path안의 내용을 모두 지운다.
    ckpt = tf.train.get_checkpoint_state('./model5')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #이렇게 restore할 경우, ckpt.model_checkpoint_path는 가장 마지막으로 저장된 모델의 path를 의미하여
        #가장 마지막 학습된 모델이 restore된다.
        #특정 global step의 모델을 되살리고 싶으면 print(ckpt)를 하여 모델을 확인하고, 모델 경로를 넣어주면 된다.
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))
        for step in range(total_steps):
            s = step*batch_size
            t = (step+1)*batch_size
            c, a, _ = sess.run([cost, accuracy, train_op], feed_dict={X:x_train[mask[s:t]], Y:y_train[mask[s:t]]})

            loss_per_epoch += c/total_steps
            acc_per_epoch += a/total_steps

        print("Global Step : {:5d}, Epoch : [{:3d}/{:3d}], cost : {:.6f}, accuracy : {:.2%}"
              .format(sess.run(global_step), epoch, total_epochs, loss_per_epoch, acc_per_epoch))

        saver.save(sess, "./model5/dnn.ckpt", global_step=global_step)

    print("Done.")

    te_a = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("Test Accuracy : {:.2%}".format(te_a))