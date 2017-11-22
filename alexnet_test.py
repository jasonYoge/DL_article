#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout
from keras import regularizers

from keras.optimizers import SGD
from utils import get_nb_files
import os

train_dir = os.path.join(os.curdir, 'train')
val_dir = os.path.join(os.curdir, 'validation')

nb_train_examples = get_nb_files(train_dir)
nb_val_examples = get_nb_files(val_dir)

IM_WIDTH = 227
IM_HEIGHT = 227
batch_size = 16
nb_epoch = 3


def get_file_iterator(train_dir, validation_dir):
    train_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    validation_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    validation_data = validation_gen.flow_from_directory(
        validation_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_data, validation_data


def create_graph():
    input_tensor = Input(shape=(227, 227, 3))

    # part one
    x = Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation='relu',
               use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros',
               kernel_regularizer=regularizers.l2(0.0001), padding='valid')(input_tensor)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = BatchNormalization(axis=-1)(x)

    # part two
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu',
               use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros',
               kernel_regularizer=regularizers.l2(0.0001), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # part three
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu',
               use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros',
               kernel_regularizer=regularizers.l2(0.0001), padding='same')(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu',
               use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros',
               kernel_regularizer=regularizers.l2(0.0001), padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu',
               use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros',
               kernel_regularizer=regularizers.l2(0.0001), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # part four
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    # part five
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(3, activation='softmax')(x)

    return input_tensor, predictions


if __name__ == '__main__':
    input_tensor, predictions = create_graph()
    model = Model(inputs=input_tensor, outputs=predictions)
    train_data, val_data = get_file_iterator(train_dir, val_dir)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
        train_data,
        nb_epoch=nb_epoch,
        steps_per_epoch=(nb_train_examples / batch_size),
        validation_data=val_data,
        validation_steps=nb_val_examples,
        class_weight='auto'
    )