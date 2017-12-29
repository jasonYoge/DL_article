#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from utils import get_nb_files
import os

train_dir = os.path.join(os.curdir, 'train')
val_dir = os.path.join(os.curdir, 'validation')
test_dir = os.path.join(os.curdir, 'test')

nb_train_examples = get_nb_files(train_dir)
nb_val_examples = get_nb_files(val_dir)
nb_test_examples = get_nb_files(test_dir)



IM_WIDTH = 500
IM_HEIGHT = 500
batch_size = 10
FC_SIZE = 1024
nb_epoch = 30


def get_file_iterator(train_dir, validation_dir, test_dir):
    train_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
    )
    validation_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
    )
    test_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
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
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_data, validation_data, test_data


def freeze_old_model(base_model):
    for layer in base_model.layers[:-12]:
        layer.trainable = False


def get_preprocessing_model():
    model = ResNet50(weights='imagenet', include_top=False)
    freeze_old_model(model)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model


if __name__ == '__main__':
    train_data, val_data, test_data = get_file_iterator(train_dir, val_dir, test_dir)
    model = get_preprocessing_model()

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit_generator(
        train_data,
        nb_epoch=nb_epoch,
        steps_per_epoch=(nb_train_examples / batch_size),
        validation_data=val_data,
        validation_steps=(nb_val_examples / batch_size),
        class_weight='auto',
        callbacks=[tb]
    )
    scoreSeg = model.evaluate_generator(test_data, steps=20, max_queue_size=20)
    print('Accuracy = ', scoreSeg[1])
