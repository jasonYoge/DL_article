import tensorflow as tf
import numpy as np
import os
import glob
import math
from tensorflow.python.platform import gfile

dir_name = './train'
file_count = 500


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
        return sess.run(img_data).astype(np.float32)


def find_examples(sess, dir_name):
    sub_dirs = [x[0] for x in os.walk(dir_name)]
    sub_dirs = sub_dirs[1:]
    total_count = math.ceil(get_nb_files(dir_name) / 500.)

    image_list = []
    ground_truth = []
    examples_count = 0
    tf_name = dir_name + ('.tfrecords-%.5d-of-%.5d' % (examples_count / 500, total_count))
    writer = tf.python_io.TFRecordWriter(tf_name)

    for i, sub_dir in enumerate(sub_dirs):
        file_glob = os.path.join(sub_dir, '*.jpg')
        for file_path in glob.glob(file_glob):
            image_list.append(read_img(sess, file_path))
            ground_truth_data = np.zeros(len(sub_dir), dtype=np.float32)
            ground_truth_data[i] = 1.0
            ground_truth.append(ground_truth_data)
            examples_count = examples_count + 1
            if examples_count - 500 >= 0:
                



with tf.Session() as sess:
    image_list, ground_truth = find_examples(sess, dir_name)
    print image_list
    print ground_truth