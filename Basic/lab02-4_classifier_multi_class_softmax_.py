import tensorflow as tf
import numpy as np
import load_data

SEED = 0
dataX, dataY = load_data.generate_data_for_multi_class_classification(seed = SEED, scaling = True)

NSAMPLES = np.size(dataX, 0)
INPUT_DIM = np.size(dataX,1)
NCLASS = len(np.unique(dataY))
TOTAL_EPOCH = 50000

print("The number of data samples : ", NSAMPLES)
print("The dimension of data samples : ", INPUT_DIM)

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, 
                            initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def sigmoid_linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, 
                            initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = sigmoid_linear(X, 128, 'FC_Layer1')
h2 = sigmoid_linear(h1, 256, 'FC_Layer2')
logits = linear(h2, NCLASS, 'FC_Layer3')

hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
loss = -tf.reduce_mean(Y_one_hot*tf.log(hypothesis), name = 'loss')
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

predict = tf.argmax(hypothesis, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(TOTAL_EPOCH):
        a, l, _ = sess.run([ accuracy, loss, optim], feed_dict={X: dataX, Y: dataY})
        if (epoch+1) %1000 == 0:
            print("Epoch [{:3d}/{:3d}], loss = {:.6f}, accuracy = {:.2%}".format(epoch + 1, TOTAL_EPOCH, l, a))

'''
Epoch [1000/50000], loss = 0.816317, accuracy = 13.36%
Epoch [2000/50000], loss = 0.665140, accuracy = 14.75%
Epoch [3000/50000], loss = 0.557679, accuracy = 17.58%
Epoch [4000/50000], loss = 0.469572, accuracy = 20.09%
Epoch [5000/50000], loss = 0.399450, accuracy = 22.59%
Epoch [6000/50000], loss = 0.342672, accuracy = 26.60%
Epoch [7000/50000], loss = 0.301455, accuracy = 31.00%
Epoch [8000/50000], loss = 0.274866, accuracy = 34.11%
Epoch [9000/50000], loss = 0.256859, accuracy = 36.51%
Epoch [10000/50000], loss = 0.242351, accuracy = 38.95%
Epoch [11000/50000], loss = 0.229465, accuracy = 40.85%
Epoch [12000/50000], loss = 0.217710, accuracy = 43.18%
Epoch [13000/50000], loss = 0.206913, accuracy = 44.85%
Epoch [14000/50000], loss = 0.196962, accuracy = 46.47%
Epoch [15000/50000], loss = 0.187767, accuracy = 48.02%
Epoch [16000/50000], loss = 0.179246, accuracy = 49.47%
Epoch [17000/50000], loss = 0.171334, accuracy = 50.64%
Epoch [18000/50000], loss = 0.163976, accuracy = 51.75%
Epoch [19000/50000], loss = 0.157121, accuracy = 53.20%
Epoch [20000/50000], loss = 0.150720, accuracy = 53.98%
Epoch [21000/50000], loss = 0.144741, accuracy = 55.37%
Epoch [22000/50000], loss = 0.139151, accuracy = 56.87%
Epoch [23000/50000], loss = 0.133921, accuracy = 58.49%
Epoch [24000/50000], loss = 0.129023, accuracy = 60.16%
Epoch [25000/50000], loss = 0.124433, accuracy = 61.27%
Epoch [26000/50000], loss = 0.120127, accuracy = 62.55%
Epoch [27000/50000], loss = 0.116089, accuracy = 63.72%
Epoch [28000/50000], loss = 0.112298, accuracy = 64.94%
Epoch [29000/50000], loss = 0.108738, accuracy = 66.33%
Epoch [30000/50000], loss = 0.105388, accuracy = 67.00%
Epoch [31000/50000], loss = 0.102235, accuracy = 67.72%
Epoch [32000/50000], loss = 0.099265, accuracy = 68.89%
Epoch [33000/50000], loss = 0.096467, accuracy = 69.67%
Epoch [34000/50000], loss = 0.093826, accuracy = 70.45%
Epoch [35000/50000], loss = 0.091330, accuracy = 71.12%
Epoch [36000/50000], loss = 0.088969, accuracy = 72.01%
Epoch [37000/50000], loss = 0.086734, accuracy = 72.68%
Epoch [38000/50000], loss = 0.084615, accuracy = 73.51%
Epoch [39000/50000], loss = 0.082605, accuracy = 73.96%
Epoch [40000/50000], loss = 0.080696, accuracy = 74.62%
Epoch [41000/50000], loss = 0.078881, accuracy = 75.40%
Epoch [42000/50000], loss = 0.077154, accuracy = 75.90%
Epoch [43000/50000], loss = 0.075510, accuracy = 76.74%
Epoch [44000/50000], loss = 0.073942, accuracy = 77.35%
Epoch [45000/50000], loss = 0.072445, accuracy = 77.69%
Epoch [46000/50000], loss = 0.071015, accuracy = 77.96%
Epoch [47000/50000], loss = 0.069647, accuracy = 78.46%
Epoch [48000/50000], loss = 0.068337, accuracy = 78.80%
Epoch [49000/50000], loss = 0.067083, accuracy = 79.08%
Epoch [50000/50000], loss = 0.065880, accuracy = 79.30%
'''
