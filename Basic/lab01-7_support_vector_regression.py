import matplotlib.pyplot as plt
import tensorflow as tf
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 1000
EPSILON = 1e-3
C = 10.0

def create_weight_variable(shape):
    W = tf.get_variable(name = 'W', shape = shape, dtype = tf.float32, initializer= tf.truncated_normal_initializer())
    b = tf.get_variable(name = 'b', shape = shape, dtype = tf.float32, initializer= tf.constant_initializer(0.0))
    return W, b

dataX, dataY = load_data.generate_data_for_linear_regression(NPOINTS)

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

W, b = create_weight_variable([1])
hypothesis = tf.nn.bias_add(tf.multiply(X, W), b, name = 'hypothesis')

l2_norm = tf.add_n([tf.square(v) for v in tf.trainable_variables() if 'W' in v.name])[0]

loss = 0.5*l2_norm+C*tf.reduce_mean(tf.maximum(tf.zeros_like(hypothesis),tf.abs(tf.subtract(hypothesis,Y))-EPSILON))
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

plt.figure(num=None, figsize=(8, 14), dpi=60, facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace = 0.4, top = 0.9, bottom = 0.05)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    arg_ = 0
    for epoch in range(TOTAL_EPOCH):
        l, W_value, b_value, _ = sess.run([loss, W, b, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %100 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, TOTAL_EPOCH, l))

            #for plot graph
            arg_+=1
            plt.subplot(5, 2, arg_)
            plt.scatter(dataX, dataY, marker='.')
            plt.plot(dataX, dataX * W_value + b_value, c='r')
            plt.title('Epoch {}'.format(epoch+1))
            plt.grid()
            plt.xlim(-2, 2)

plt.suptitle('Support_Vector_Regression', fontsize=20)
plt.savefig('./image/lab01-7_support_vector_regression.jpg')
plt.show()

'''
Epoch [100/1000], loss = 0.399017
Epoch [200/1000], loss = 0.397028
Epoch [300/1000], loss = 0.397028
Epoch [400/1000], loss = 0.397027
Epoch [500/1000], loss = 0.397027
Epoch [600/1000], loss = 0.397027
Epoch [700/1000], loss = 0.397027
Epoch [800/1000], loss = 0.397027
Epoch [900/1000], loss = 0.397027
Epoch [1000/1000], loss = 0.397027
'''
