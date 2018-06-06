import tensorflow as tf
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 10000

dataX, dataY = load_data.generate_data_for_two_class_classification(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

W = tf.Variable(tf.truncated_normal(shape = [1]), name = 'W')
b = tf.Variable(tf.zeros([1]), name = 'b')
logits = tf.nn.bias_add(tf.multiply(X, W), b, name = 'logits')

hypothesis = tf.nn.sigmoid(logits, name = 'hypothesis')
loss = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.cast(hypothesis>0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    arg_ = 0
    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %1000 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [1000/10000], loss = 0.477491, accuracy = 82.45%
Epoch [2000/10000], loss = 0.424556, accuracy = 86.30%
Epoch [3000/10000], loss = 0.385662, accuracy = 88.75%
Epoch [4000/10000], loss = 0.354860, accuracy = 90.25%
Epoch [5000/10000], loss = 0.329532, accuracy = 91.70%
Epoch [6000/10000], loss = 0.308235, accuracy = 92.70%
Epoch [7000/10000], loss = 0.290052, accuracy = 93.50%
Epoch [8000/10000], loss = 0.274342, accuracy = 94.20%
Epoch [9000/10000], loss = 0.260637, accuracy = 94.50%
Epoch [10000/10000], loss = 0.248580, accuracy = 94.70%
'''
