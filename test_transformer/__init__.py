#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from transformer import transformer
from skimage import transform
import matplotlib.pyplot as plt
import os
import random

w_size = 299
h_size = 299
d_size = 3
out_size = (w_size, h_size, d_size)
dir = os.path.join(os.path.curdir, '../', 'train')


def get_file(old_dir, batch_size):
    if not os.path.exists(old_dir):
        raise SyntaxError('输入的图片保存路径不存在!')

    dirs = os.listdir(old_dir)[1:]
    file_list = []
    ground_truth = []
    for dir in dirs:
        new_dir = os.path.join(old_dir, dir)
        file_list.extend(os.listdir(new_dir))

    random_list = random.sample(file_list, batch_size)
    for idx, old_name in enumerate(random_list):
        for i, dir in enumerate(dirs):
            new_dir = os.path.join(old_dir, dir)
            if old_name in os.listdir(new_dir):
                random_list[idx] = os.path.join(old_dir, dir, old_name)
                ground_truth.append(i)

    return random_list, ground_truth


def load_img(sess, file_list):
    img_list_size = len(file_list)
    img_arr = np.zeros((img_list_size, w_size, h_size, d_size))

    for idx, path in enumerate(file_list):
        if not os.path.exists(path):
            raise SyntaxError('图片文件对应路径%s不存在' % path)

        image_data = tf.gfile.FastGFile(path, 'r').read()
        img_data_jpg = sess.run(tf.image.decode_jpeg(image_data))
        # img_arr[idx] = np.resize(img_data_jpg, (w_size * h_size * d_size))
        img_arr[idx][:, :, :] = transform.resize(img_data_jpg, (299, 299, 3))

    return img_arr


def run_net(sess, batch_xs, batch_ys, keep=1.0):
    x = tf.placeholder(tf.float32, [None, w_size * h_size * d_size])
    y = tf.placeholder(tf.float32, [None, 3])

    x_tensor = tf.reshape(x, [-1, w_size, h_size, d_size])
    W_fc_loc1 = weight_variable([w_size * h_size * d_size, 20])
    b_fc_loc1 = bias_variable([20])

    W_fc_loc2 = weight_variable([20, 6])
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

    h_trans = transformer(x_tensor, h_fc_loc2, out_size)

    sess.run(tf.global_variables_initializer())
    return sess.run(h_trans, feed_dict={
        x: batch_xs,
        y: batch_ys,
        keep_prob: keep
    })


def draw_image(xout, xtrans, batch_ys):
    fig = plt.figure(figsize=(10, 2))

    for idx in range(xtrans.shape[0]):
        ax1 = fig.add_subplot(2, 10, idx + 1)
        ax2 = fig.add_subplot(2, 10, idx + 11)

        ax1.imshow(xout[idx, :, :, :])
        ax2.imshow(xtrans[idx, :, :, :])
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1.set_title(np.argmax(batch_ys, axis=1)[idx])
        ax2.set_title(np.argmax(batch_ys, axis=1)[idx])

    plt.show()

if __name__ == '__main__':
    with tf.Session() as sess:
        file_list, ground_truth = get_file(dir, 10)
        # change to one hot
        batch_ys = dense_to_one_hot(ground_truth, 3)
        xout = load_img(sess, file_list)
        batch_xs = np.resize(xout, (-1, w_size * h_size * d_size))

        xtransOut = run_net(sess, batch_xs, batch_ys, 1.0)
        draw_image(xout, xtransOut, batch_ys)
