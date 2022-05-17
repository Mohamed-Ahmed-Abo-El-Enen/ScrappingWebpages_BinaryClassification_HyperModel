import keras.backend as K
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score


def F1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


tf.config.run_functions_eagerly(True)
@tf.function
def Precision_macro_score(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    return precision_score(y_true, (y_pred >= 0.5).astype(int), average="macro", zero_division=1)


tf.config.run_functions_eagerly(True)
@tf.function
def Recall_macro_score(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    return recall_score(y_true, (y_pred >= 0.5).astype(int), average="macro", zero_division=1)


def F1_macro_score(y_true, y_pred):
    precision_val = Precision_macro_score(y_true, y_pred)
    recall_val = Recall_macro_score(y_true, y_pred)
    f1_val = 2*((precision_val*recall_val)/(precision_val+recall_val))
    return f1_val

