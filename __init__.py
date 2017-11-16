#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import os
import shutil
import pandas as pd  # 处理csv文件
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

train_dir_name = 'train'
validation_dir_name = 'validation'
superpixels_name = 'ISIC_*_superpixels.png'
train_csv_name = 'ISIC-2017_Training_Part3_GroundTruth.csv'
validation_csv_name = 'ISIC-2017_Validation_Part3_GroundTruth.csv'
data_list = {}
data_type = ['melanoma', 'nevus', 'seborrheic_keratosis']
nb_classes = len(data_type)
nb_epoch = 3
batch_size = 50

IM_WIDTH = 299
IM_HEIGHT = 299
FC_SIZE = 1024

# 删除超像素文件
def delete_superpixels_from_data():
    train_dir_path = os.path.join(os.curdir, train_dir_name)
    validation_dir_path = os.path.join(os.curdir, validation_dir_name)

    file_path = os.path.join(train_dir_path, superpixels_name)
    if os.path.exists(train_dir_path):
        file_list = glob.glob(file_path)
        for path in file_list:
            os.remove(path)
        print 'Removed superpixel files from train directory.'

    file_path = os.path.join(validation_dir_path, superpixels_name)
    if os.path.exists(validation_dir_path):
        file_list = glob.glob(file_path)
        for path in file_list:
            os.remove(path)
        print 'Removed superpixel files from validation directory.'


def create_train_dir():
    # 初始化图像数据字典
    # for type_name in data_type:
    #     data_list[type_name] = {
    #         'images': [],
    #         'validation': [],
    #         'test': []
    #     }

    # 获取ground_truth数据的路径
    dir_path = os.path.join(os.curdir, train_dir_name)
    file_path = os.path.join(dir_path, train_csv_name)

    # 构造数据字典
    if os.path.exists(file_path):
        csv_info = pd.read_csv(file_path)
        for index, row in csv_info.iterrows():
            file_name = os.path.join(dir_path, row[0] + '.jpg')
            # 训练集
            if row[1] == 1.0:
                des_file_name = os.path.join(dir_path, data_type[0], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[0])
            elif row[2] == 1.0:
                des_file_name = os.path.join(dir_path, data_type[2], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[2])
            else:
                des_file_name = os.path.join(dir_path, data_type[1], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[1])

            if not os.path.exists(des_dir_name):
                os.mkdir(des_dir_name)
            shutil.move(file_name, des_file_name)
        print 'Create training category dir success.'
    else:
        raise EnvironmentError('Train CSV file path is wrong!')


def create_validation_dir():
    # 初始化图像数据字典
    # for type_name in data_type:
    #     data_list[type_name] = {
    #         'images': [],
    #         'validation': [],
    #         'test': []
    #     }

    # 获取ground_truth数据的路径
    dir_path = os.path.join(os.curdir, validation_dir_name)
    file_path = os.path.join(dir_path, validation_csv_name)

    # 构造数据字典
    if os.path.exists(file_path):
        csv_info = pd.read_csv(file_path)
        for index, row in csv_info.iterrows():
            file_name = os.path.join(dir_path, row[0] + '.jpg')
            # 训练集
            if row[1] == 1.0:
                des_file_name = os.path.join(dir_path, data_type[0], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[0])
            elif row[2] == 1.0:
                des_file_name = os.path.join(dir_path, data_type[2], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[2])
            else:
                des_file_name = os.path.join(dir_path, data_type[1], row[0] + '.jpg')
                des_dir_name = os.path.join(dir_path, data_type[1])

            if not os.path.exists(des_dir_name):
                os.mkdir(des_dir_name)
            shutil.move(file_name, des_file_name)
        print 'Create validation category dir success.'
    else:
        raise EnvironmentError('Validation CSV file path is wrong!')


def get_nb_files(directory):
    """获取样本个数"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# 添加新层
def add_new_last_layer(base_model, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    # 数据预处理
    # delete_superpixels_from_data()
    # create_train_dir()
    # create_validation_dir()
    train_dir = os.path.join(os.path.curdir, 'train')
    val_dir = os.path.join(os.path.curdir, 'validation')

    nb_train_samples = get_nb_files(train_dir)
    nb_classes = len(glob.glob(train_dir + '/*'))
    nb_val_samples = get_nb_files(val_dir)
    nb_epoch = int(nb_epoch)
    batch_size = int(batch_size)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(None, None),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(None, None),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # 定义网络框架
    base_model = InceptionV3(weights='imagenet', include_top=False)  # 预先要下载no_top模型
    model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
    setup_to_transfer_learn(model, base_model)  # 冻结base_model所有层

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')
