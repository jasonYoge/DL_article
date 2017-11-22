#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from utils import get_nb_files

train_dir_name = 'train'
validation_dir_name = 'validation'
data_list = {}
data_type = ['melanoma', 'nevus', 'seborrheic_keratosis']
nb_classes = len(data_type)
nb_epoch = 1
batch_size = 10

IM_WIDTH = 299
IM_HEIGHT = 299
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 8


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


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
    model: keras model
    """
    for layer in model.layers:
        layer.trainable = False
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == '__main__':
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
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # 定义网络框架
    base_model = InceptionV3(weights='imagenet', include_top=False)  # 预先要下载no_top模型
    setup_to_finetune(base_model)  # 冻结base_model所有层
    model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=[tb])
