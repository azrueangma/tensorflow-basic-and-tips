import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import save_and_load_mnist
import os
import shutil

if not os.path.exists('./samples9_1'):
    os.mkdir('./samples9_1')
dataset = save_and_load_mnist("./data/mnist/")

x_train = dataset['train_data']
y_train = dataset['train_target']
x_test = dataset['test_data']
y_test = dataset['test_target']


class GAN(object):
    def __init__(self):
        self._build_net()


    def _build_net(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # 여기서 100은 noise의 dimension을 의미한다.

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
        self.Z = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='Z')

        # 골빈의 책에서 GAN의 첫 번째 코드에서 이 부분 reuse를 안하고 있다. 심각한 오류!!!
        self.G = self.generator(self.Z)
        self.D_gen = self.discriminator(self.G, reuse=False)
        self.D_real = self.discriminator(self.X, reuse=True)

        self.loss_D = -tf.reduce_mean(tf.log(self.D_real)+tf.log(1-self.D_gen))
        self.loss_G = -tf.reduce_mean(tf.log(self.D_gen))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        self.train_D = tf.train.AdamOptimizer(0.0001).minimize(self.loss_D, var_list=d_vars, global_step=self.global_step)
        self.train_G = tf.train.AdamOptimizer(0.0001).minimize(self.loss_G, var_list=g_vars)


    def generator(self, noise_z):
        # hidden unit을 하나만 두는 것을 말하는데 여기서는 256이다.
        # cnn이 아닌 dnn을 이용하여 생성하는 코드이다.
        with tf.variable_scope('generator') as scope:
            g_W1 = tf.get_variable(name='g_W1', shape=[100, 256], initializer=tf.glorot_uniform_initializer())
            g_b1 = tf.get_variable(name='g_b1', shape=[256], initializer=tf.zeros_initializer())
            g_W2 = tf.get_variable(name='g_W2', shape=[256, 784], initializer=tf.glorot_uniform_initializer())
            g_b2 = tf.get_variable(name='g_b2', shape=[784], initializer=tf.zeros_initializer())
            hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(noise_z, g_W1), g_b1))
            output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden, g_W2), g_b2))
        return output


    def discriminator(self, inputs, reuse):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            d_W1 = tf.get_variable(name='d_W1', shape=[784, 256], initializer=tf.glorot_uniform_initializer())
            d_b1 = tf.get_variable(name='d_b1', shape=[256], initializer=tf.zeros_initializer)
            d_W2 = tf.get_variable(name='d_W2', shape=[256, 1], initializer=tf.glorot_uniform_initializer())
            d_b2 = tf.get_variable(name='d_b2', shape=[1], initializer=tf.zeros_initializer())
            hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, d_W1), d_b1))
            output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden, d_W2), d_b2))
        return output


    def get_noise(self, batch_size, n_noise):
        return np.random.normal(size=(batch_size, n_noise))


    def get_noise_with_seed(self, batch_size, n_noise, seed):
        np.random.seed(0)
        return np.random.normal(size=(batch_size, n_noise))


    def fit(self, reuse_model=False):
        total_epochs = 20
        batch_size = 32
        total_steps = int(len(x_train) / batch_size)
        print(">>> Training Start [total epochs : {}, total step : {}]".format(total_epochs, total_steps))
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            if not reuse_model:
                shutil.rmtree('./model9_1')
            ckpt = tf.train.get_checkpoint_state('./model9_1')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(total_epochs):
                g_loss_per_epoch = 0
                d_loss_per_epoch = 0

                np.random.seed(epoch)
                mask = np.random.permutation(len(x_train))
                for step in range(total_steps):
                    s = step * batch_size
                    t = (step + 1) * batch_size
                    noise = self.get_noise(batch_size, 100)
                    c, _ = sess.run([self.loss_D, self.train_D], feed_dict={self.X: x_train[mask[s:t]], self.Z:noise})
                    d_loss_per_epoch += c / total_steps
                    cc, _ = sess.run([self.loss_G, self.train_G], feed_dict={self.Z:noise})
                    g_loss_per_epoch += cc / total_steps

                print("Global Step : {:5d}, Epoch : [{:3d}/{:3d}], d_cost : {:.6f}, g_cost : {:.6f}"
                      .format(sess.run(self.global_step), epoch+1, total_epochs, d_loss_per_epoch, g_loss_per_epoch))
                saver.save(sess, "./model9/dnn.ckpt", global_step=self.global_step)

                #plot하여 저장하기 위한 단계
                sample_size = 100
                noise = self.get_noise_with_seed(sample_size, 100, seed=0)
                samples = sess.run(self.G, feed_dict={self.Z:noise})

                images = np.reshape(samples, [len(samples), 28, 28])
                fig = plt.figure()
                plt.gca().set_axis_off()
                h_num = int(np.sqrt(sample_size))
                v_num = int(np.sqrt(sample_size))
                v_list = []
                count = 0
                for j in range(v_num):
                    h_list = []
                    for i in range(h_num):
                        h_list.append(images[count])
                        count += 1
                    tmp = np.hstack(h_list)
                    v_list.append(tmp)
                im = np.vstack(v_list)
                plt.imshow(im, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.savefig('./samples9_1/{}.png'.format(str(epoch + 1).zfill(3)))
            plt.close(fig)
            print("Done.")


m = GAN()
m.fit()