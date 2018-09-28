#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def check_args(args):
    check_folder(args.board_path)
    assert args.total_epoch >= 1, 'number of total epoch must be larger than or equal to one'
    assert args.cpu_device >= 0, 'number of cpu device must be larger than or equal to one'
    return args

def parse_args():
    desc = "Model parameters"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('total_epoch', type=int, default=30, help='The number of epochs')
    parser.add_argument('cpu_device', type=int, default=0, help='the number of cpu device')
    parser.add_argument('board_path', type=str, default="./board/lab01-x_board", help="board directory")
    return check_args(parser.parse_args())

def main():
    args = parse_args()
    print(args)
    if args is None:
      exit()

    #load data
    x_train = np.array([[3.5], [4.2], [3.8], [5.0], [5.1], [0.2], [1.3], [2.4], [1.5], [0.5]])
    y_train = np.array([[3.3], [4.4], [4.0], [4.8], [5.6], [0.1], [1.2], [2.4], [1.7], [0.6]])

    g = tf.Graph()
    with g.as_default(), tf.device('/cpu:{}'.format(args.cpu_device)):
        tf.set_random_seed(0)
        with tf.variable_scope("Inputs"):
            #create placeholder for input
            X = tf.placeholder(shape=[None,1], dtype=tf.float32, name='X')
            Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name='Y')

        with tf.variable_scope("Weight_and_Bias"):
            #weigth and bias
            W = tf.Variable(tf.truncated_normal(shape=[1]), name='W')
            b = tf.Variable(tf.zeros([1]), name='b')

        with tf.variable_scope("Output"):
            output = tf.nn.bias_add(tf.multiply(X, W), b, name='output')

        with tf.variable_scope("Loss"):
            loss = tf.reduce_mean(tf.square(Y-output), name='loss')

        with tf.variable_scope("Optimization"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            optim = optimizer.minimize(loss)

        with tf.variable_scope("Summary"):
            W_hist = tf.summary.histogram('W_hist', W)
            b_hist = tf.summary.histogram('b_hist', b)
            loss_scalar = tf.summary.scalar('loss_scalar', loss)
            merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(args.board_path)

        init_op = tf.global_variables_initializer()
        with tf.Session(graph=g) as sess:
            writer.add_graph(sess.graph)
            sess.run(init_op)

            for epoch in range(args.total_epoch):
                m, l, _ = sess.run([merged, loss, optim], feed_dict={X:x_train, Y:y_train})
                writer.add_summary(m, global_step=epoch)
                if (epoch+1) % 10 == 0:
                    print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, args.total_epoch, l))

if __name__ == '__main__':
    main()
