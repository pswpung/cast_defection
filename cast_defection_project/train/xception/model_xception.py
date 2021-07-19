from typing import Tuple

import tensorflow as tf
import tensorflow.keras.applications.nasnet as backbone
from numpy import ndarray
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import EagerTensor, Tensor
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.preprocessing.image import DirectoryIterator


def preprocess_train(img: ndarray) -> EagerTensor:
    """
    Preprocess input image

    Arguments
    ----------
    img : ndarray
        image array

    Return
    ----------
    img : EagerTensor
        apply preprocess image and return 

    """
    img: EagerTensor = tf.clip_by_value(img, 0., 255.)
    img: EagerTensor = backbone.preprocess_input(img)
    return img


def split_data(train_path: str, test_path: str, validation_split: float, input_size: int, train_batch_size: int) \
        -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, int, int]:
    """
    Split data into train_gen, val_gen, test_gen and calculate validation_steps, steps_per_epoch

    Arguments
    ----------
    train_path: str
        path of train dataset
    test_path: str
        path of test dataset
    validation_split: float
        propotion for validation generator
    input_size: int
        image size
    train_batch_size: int
        Batch size

    Return
    ----------
    train_gen: DirectoryIterator
        preprocessed dataset that used to train model 
    val_gen: DirectoryIterator
        preprocessed dataset that used to validate model 
    test_gen: DirectoryIterator
        preprocessed dataset that used to test model 
    validation_steps: int
        number of validation steps per epochs
    steps_per_epoch: int
        number of training steps per epochs

    """
    input_height: int = input_size
    input_width: int = input_size
    imageGenerator = ImageDataGenerator(
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

    test_gen: DirectoryIterator = imageGenerator.flow_from_directory(
        test_path,
        target_size=(input_height, input_width),
        classes=["def_front", "ok_front"],
        class_mode="binary",
        batch_size=1
    )

    validation_steps: int = len(val_gen) // train_batch_size
    steps_per_epoch: int = len(train_gen) // train_batch_size
    return train_gen, val_gen, test_gen, validation_steps, steps_per_epoch


def get_model(input_shape: Tuple[int, int, int]) -> Functional:
    """
    Declare base model then modified.

    Arguments
    ----------
    input_shape : tuple(int, int, int)

    Return
    ----------
    tf.keras.Model(inputs, outputs) : Functional
        retun keras model

    """
    # Declare Base Model
    base_model: Functional = Xception(
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


def train_model(model: Functional, train_gen: DirectoryIterator, val_gen: DirectoryIterator, n_epochs: int, steps_per_epoch: int, validation_steps: int, model_name: str) -> None:
    '''
    summary, train and save model.

    Arguments
    ----------
    model: Functional
        modified model architecture
    train_gen: DirectoryIterator
        preprocessed dataset that used to train model 
    val_gen: DirectoryIterator
        preprocessed dataset that used to validate model 
    n_epochs: int
        Number of Epochs
    steps_per_epoch: int
        number of training steps per epochs
    validation_steps: int
        number of validation steps per epochs

    '''
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    model.save_weights(model_name)
