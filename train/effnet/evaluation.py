from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             recall_score, roc_curve)
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tqdm import tqdm


def model_evaluation(model: Functional, test_gen: DirectoryIterator) -> None:
    """
    evaluate trained model

    Arguments
    ----------
    model: Functional
        trained model
    test_gen: DirectoryIterator
        preprocessed dataset that used to test model

    """
    loss, accuracy = model.evaluate(test_gen)
    print(f"Loss :{loss:.4f} Accuracy:{accuracy*100:.4f}%")


def predict(model: Functional, test_gen: DirectoryIterator) -> Tuple[ndarray, ndarray]:
    """
    predict result by trained model

    Arguments
    ----------
    model: Functional
        trained model
    test_gen: DirectoryIterator
        preprocessed dataset that used to test model

    Return
    ----------
    y_true: ndarray
        array of answer for individual image
    y_score: ndarray
        array of predicted for individual image

    """
    y_true: List = []
    y_score: List = []
    for i, (x, y) in tqdm(enumerate(test_gen), total=len(test_gen)):
        y_true.append(y[0])
        y_score.append(model(x, training=False)[0][0].numpy())
        if i+1 > len(test_gen):
            break
    y_true: ndarray = np.array(y_true)
    y_score: ndarray = np.array(y_score)
    return y_true, y_score


def visualize_model(thresh: float, y_true: ndarray, y_score: ndarray) -> None:
    """
    visualize model
        - ROC curve
        - confusion Matrix
        - Accuracy, Recall, F1-score

    Arguments
    ----------
    thresh: float
        threshold value for interpretation and seperate y_score to 'ok' or 'defect' 
    y_true: ndarray
        array of answer for individual image
    y_score: ndarray
        array of predicted for individual image

    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc: float = auc(fpr, tpr)

    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    y_pred: ndarray = np.array([1. if yi > thresh else 0. for yi in y_score])

    acc: float = accuracy_score(y_true, y_pred)
    f1: float = f1_score(y_true, y_pred)
    recall: float = recall_score(y_true, y_pred)
    cm: ndarray = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax[0])
    sns.heatmap(cm / np.sum(cm, axis=0), annot=True, ax=ax[1])
    ax[0].set_title("Confusion Matrix")
    ax[1].set_title("Normalized Confusion Matrix")
    plt.show()
    print(f"Threshold = {thresh}")
    print(
        f"Accuracy: {acc*100:.2f}%\nRecall: {recall*100:.2f}%\nF1-score: {f1:.2f}")
