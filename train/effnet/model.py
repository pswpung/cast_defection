from typing import Tuple

import tensorflow as tf
import tensorflow.keras.applications.efficientnet as backbone
from tensorflow.keras.applications.efficientnet import EfficientNetB4 as EffNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_train(img):
    img = tf.image.random_hue(img, max_delta=0.5)
    img = tf.image.random_contrast(img, lower=0.75, upper=1.5)
    img = tf.clip_by_value(img, 0., 255.)
    img = backbone.preprocess_input(img)
    return img


def get_model(input_shape):
    # Declare Base Model
    base_model = EffNet(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def split_data(train_path: str, test_path: str, validation_split: float, input_size: int, train_batch_size: int):
    input_height: int = input_size
    input_width: int = input_size
    imageGenerator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.9, 1.1],
        shear_range=5,
        zoom_range=0.1,
        fill_mode='constant',
        horizontal_flip=True,
        vertical_flip=True,
        dtype=tf.float32,
        validation_split=validation_split,
        preprocessing_function=preprocess_train)

    train_gen = imageGenerator.flow_from_directory(
        train_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=train_batch_size,
        subset="training"
    )

    val_gen = imageGenerator.flow_from_directory(
        train_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=train_batch_size,
        subset="validation"
    )

    test_gen = ImageDataGenerator(dtype=tf.float32, preprocessing_function=backbone.preprocess_input).flow_from_directory(
        test_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=1
    )
    validation_steps: float = len(val_gen) // train_batch_size
    steps_per_epoch: float = len(train_gen) // train_batch_size
    return train_gen, val_gen, test_gen, validation_steps, steps_per_epoch


def train_model(model, train_gen, val_gen, n_epochs, steps_per_epoch, validation_steps):
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
