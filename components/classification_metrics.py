from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    """Recall: (TP)/(TP+FN).

    Args:
        y_true (tensor): Original values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: Calculated recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """Precision: (TP)/(TP+FP).

    Args:
        y_true (tensor): Original values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: Calculated precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """F1 score: (precision * recall)/(precision+recall) * w

    Args:
        y_true (tensor): Original values.
        y_pred (tensor): Predicted values.

    Returns:
        tensor: Calculated F1 score
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
