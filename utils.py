#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import glob
from tensorflow.python.platform import gfile

train_name = 'train'
validation_name = 'validation'


def get_nb_files(directory):
    """获取样本个数"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def read_img(sess, img_path):
    if os.path.exists(img_path):
        img_data = gfile.FastGFile(img_path).read()
        img_data = tf.image.decode_jpeg(img_data)
        return img_data


def flip_img(sess, img_data, type='up_down'):
    if type == 'up_down':
        fliped_data = tf.image.flip_up_down(img_data)
    elif type == 'left_right':
        fliped_data = tf.image.flip_left_right(img_data)
    else:
        fliped_data = tf.image.transpose_image(img_data)
    return fliped_data


def write_img(sess, img_data, path):
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.FastGFile(path, 'wb') as f:
        f.write(sess.run(encoded_image))


def rewrite_img(sess, dir_name):
    sub_dirs = [x[0] for x in os.walk(dir_name)]
    sub_dirs = sub_dirs[1:]

    for i, sub_dir in enumerate(sub_dirs):
        file_glob = os.path.join(sub_dir, '*.jpg')
        for file_path in glob.glob(file_glob):
            base_name = file_path[:-4]
            img_data = read_img(sess, file_path)
            flip_up_down = flip_img(sess, img_data, 'up_down')
            print flip_up_down
            write_img(sess, flip_up_down, base_name + '_up_down.jpg')
            flip_left_right = flip_img(sess, img_data, 'left_right')
            print flip_left_right
            write_img(sess, flip_left_right, base_name + '_left_right.jpg')
            flip_transpose = flip_img(sess, base_name, 'transpose')
            print flip_transpose
            write_img(sess, flip_transpose, base_name + '_transpose.jpg')
    print 'success!'


def main(_):
    with tf.Session() as sess:
        rewrite_img(sess, train_name)
        rewrite_img(sess, validation_name)


if __name__ == '__main__':
    tf.app.run()