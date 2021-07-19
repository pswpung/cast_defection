import argparse
import sys
from argparse import ArgumentParser, Namespace
import os

import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.preprocessing.image import DirectoryIterator

from model_effnet import get_model, split_data, train_model

sys.path.append("../")
from evaluation import model_evaluation, predict, visualize_model


# constants variable
static_path: str = "../../static/trained_model/"
model_name: str = "EffNet.h5"
model_save: str = os.path.join(static_path, model_name)

def get_static(static_path: str) -> None:
    """
    create static/trained model to save trained model (h5 file)

    Arguments
    ----------
    static_path : str
        path of static folder
    """
    try:
        os.makedirs(static_path)
    except OSError as err:
        print(err.strerror)
    else:
        print('Successfully created directory')

def parser() -> Namespace:
    """
    Run argument parsers

    Return
    ------
    args: argparse.Namespace
    """
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of Epochs")
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="propotion for validation generater [0<= x <=1]")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="threshold value [0<= x <=1]")
    parser.add_argument("--input_size", type=int, default=512,
                        help="image size")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="casting_data path on your local machine")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # read from parser
    n_epochs: int = args.n_epochs
    train_batch_size: int = args.train_batch_size
    validation_split: float = args.validation_split
    thresh: float = args.thresh
    input_size: int = args.input_size
    dataset_path: str = args.dataset_path
    # create input shape
    input_shape: tuple(int, int, int) = (input_size, input_size, 3)
    # create train_path and test_path
    train_path: str = os.path.join(dataset_path, "train/")
    test_path: str = os.path.join(dataset_path, "test/")

    # create static directory
    get_static(static_path)

    # split data
    train_gen, val_gen, test_gen, validation_steps, steps_per_epoch = split_data(
        train_path, test_path, validation_split, input_size, train_batch_size)

    tf.keras.backend.clear_session()

    # declare and train model
    model: Functional = get_model(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    train_model(model, train_gen, val_gen, n_epochs,
                steps_per_epoch, validation_steps)

    # evaluate model by test_gen
    model_evaluation(model, test_gen)
    y_true, y_score = predict(model, test_gen)

    # visualize the result
    visualize_model(thresh, y_true, y_score)


if __name__ == "__main__":
    args: Namespace = parser()
    main(args)
