import os
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_plot(path: str, history: Any, acc_or_loss: str = "accuracy"):
    """Plot the given history.

    Args:
        path (str): Path to save the plot.
        history (Any): History object. After .fit method.
        acc_or_loss (str): accuracy or loss. Defaults to accuracy.
    """
    plt.figure()
    plt.plot(history[acc_or_loss])
    plt.plot(history['val_'+acc_or_loss])
    plt.title('Model_'+acc_or_loss)
    plt.ylabel(acc_or_loss)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, acc_or_loss+".png"))


def save_confusion_matrix(original_data: np.ndarray, maxed_predictions: np.ndarray, path: str):
    """Create confusion matrix.

    Args:
        original_data (np.ndarray): Original OE data.
        maxed_predictions (np.ndarray): Predicted OE data.
        path (str): Path to save the confusion matrix.
    """
    confusion = confusion_matrix(original_data, maxed_predictions)
    plt.figure()
    heatmap = sns.heatmap(confusion, annot=True)
    heatmap.figure.savefig(path)


def plot_data(original_inverted: np.ndarray, predicted_inverted: np.ndarray, path: str, nb_of_data: int = 100):
    """Plot the first nb_of_data.

    Args:
        original_inverted (np.ndarray): Original database.
        predicted_inverted (np.ndarray): Prediction.
        nb_of_data (int, optional): Number of data points. Defaults to 100.
        path (str): Path to save the plot.
    """
    plt.figure()
    plt.plot(original_inverted[:nb_of_data])
    plt.plot(predicted_inverted[:nb_of_data])
    plt.savefig(path)


def get_mse(original_inverted: np.ndarray, predicted_inverted: np.ndarray) -> float:
    """Calculates MSE.

    Args:
        original_inverted (np.ndarray): Original database.
        predicted_inverted (np.ndarray): Prediction.

    Returns:
        float: MSE
    """
    return np.square(original_inverted-predicted_inverted).mean()
