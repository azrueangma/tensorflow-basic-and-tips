import tensorflow as tf
import numpy as np
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 20
NCLASS = 2
C = 10.0

def svm_target(y):
    tmp = []
    for i in range(len(y)):
        if y[i]==0:
            tmp.append(1)
        else:
            tmp.append(-1)
    return np.expand_dims(np.array(tmp), axis=1)

dataX, dataY = load_data.generate_data_for_two_class_classification(NPOINTS)
dataY = svm_target(dataY)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

W = tf.Variable(tf.truncated_normal(shape = [1]), name = 'W')
b = tf.Variable(tf.zeros([1]), name = 'b')
logits = tf.tanh(tf.nn.bias_add(tf.multiply(X, W), b, name = 'logits'))

l2_norm = tf.reduce_sum(tf.square(W),axis=0, name = 'l2_norm')

const = tf.constant(C,name="c")
tmp =  tf.subtract(1., tf.multiply(logits, Y))
hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros_like(tmp),tmp), name = 'l2_hinge_loss')
loss = tf.add(tf.multiply(const,hinge_loss),0.5*l2_norm,'loss')
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.cast(logits>0.0, tf.int32)
is_correct = tf.cast(tf.multiply(logits, Y)>0, tf.float32)
accuracy = tf.reduce_mean(is_correct)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    arg_ = 0
    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %2 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [  2/ 20], loss = 6293.590820, accuracy = 84.60%
Epoch [  4/ 20], loss = 4138.598633, accuracy = 89.80%
Epoch [  6/ 20], loss = 2738.566650, accuracy = 93.50%
Epoch [  8/ 20], loss = 1899.675171, accuracy = 95.45%
Epoch [ 10/ 20], loss = 1270.909668, accuracy = 97.00%
Epoch [ 12/ 20], loss = 965.788818, accuracy = 98.05%
Epoch [ 14/ 20], loss = 937.067688, accuracy = 98.05%
Epoch [ 16/ 20], loss = 934.864502, accuracy = 98.15%
Epoch [ 18/ 20], loss = 934.588196, accuracy = 98.15%
Epoch [ 20/ 20], loss = 934.490845, accuracy = 98.15%
'''
