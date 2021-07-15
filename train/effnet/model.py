from typing import Tuple

import tensorflow as tf
import tensorflow.keras.applications.efficientnet as backbone
from numpy import ndarray
from tensorflow.keras.applications.efficientnet import EfficientNetB4 as EffNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import EagerTensor, Tensor
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.preprocessing.image import DirectoryIterator


def preprocess_train(img: ndarray) -> EagerTensor:
    img: EagerTensor = tf.image.random_hue(img, max_delta=0.5)
    img: EagerTensor = tf.image.random_contrast(img, lower=0.75, upper=1.5)
    img: EagerTensor = tf.clip_by_value(img, 0., 255.)
    img: EagerTensor = backbone.preprocess_input(img)
    return img


def get_model(input_shape: Tuple[int, int, int]) -> Functional:
    # Declare Base Model
    base_model: Functional = EffNet(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )

    inputs = tf.keras.Input(shape=input_shape)
    x: Tensor = inputs
    x: Tensor = base_model(x, training=False)
    x: Tensor = tf.keras.layers.GlobalAveragePooling2D()(x)
    x: Tensor = tf.keras.layers.Dense(1024, activation="relu")(x)
    x: Tensor = tf.keras.layers.Dropout(0.2)(x)
    outputs: Tensor = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def split_data(train_path: str, test_path: str, validation_split: float, input_size: int, train_batch_size: int) \
        -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, int, int]:
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

    train_gen: DirectoryIterator = imageGenerator.flow_from_directory(
        train_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=train_batch_size,
        subset="training"
    )

    val_gen: DirectoryIterator = imageGenerator.flow_from_directory(
        train_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=train_batch_size,
        subset="validation"
    )

    test_gen: DirectoryIterator = ImageDataGenerator(dtype=tf.float32, preprocessing_function=backbone.preprocess_input).flow_from_directory(
        test_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=1
    )
    validation_steps: int = len(val_gen) // train_batch_size
    steps_per_epoch: int = len(train_gen) // train_batch_size
    return train_gen, val_gen, test_gen, validation_steps, steps_per_epoch


def train_model(model: Functional, train_gen: DirectoryIterator, val_gen: DirectoryIterator, n_epochs: int, steps_per_epoch: int, validation_steps: int) -> None:
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
