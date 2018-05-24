import tensorflow as tf
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 10000
NCLASS = 2

dataX, dataY = load_data.generate_data_for_two_class_classification(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

W = tf.Variable(tf.truncated_normal(shape = [1, 2]), name = 'W')
b = tf.Variable(tf.zeros([2]), name = 'b')
logits = tf.nn.bias_add(tf.multiply(X, W), b, name = 'logits')

hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
loss = -tf.reduce_mean(Y_one_hot*tf.log(hypothesis)+(1-Y_one_hot)*tf.log(1-hypothesis))
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.argmax(hypothesis, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    arg_ = 0
    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %1000 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [1000/10000], loss = 0.727547, accuracy = 50.00%
Epoch [2000/10000], loss = 0.493531, accuracy = 73.80%
Epoch [3000/10000], loss = 0.399124, accuracy = 86.40%
Epoch [4000/10000], loss = 0.339412, accuracy = 90.80%
Epoch [5000/10000], loss = 0.297381, accuracy = 92.90%
Epoch [6000/10000], loss = 0.266228, accuracy = 94.30%
Epoch [7000/10000], loss = 0.242277, accuracy = 94.95%
Epoch [8000/10000], loss = 0.223322, accuracy = 95.45%
Epoch [9000/10000], loss = 0.207958, accuracy = 95.70%
Epoch [10000/10000], loss = 0.195255, accuracy = 96.10%
'''
