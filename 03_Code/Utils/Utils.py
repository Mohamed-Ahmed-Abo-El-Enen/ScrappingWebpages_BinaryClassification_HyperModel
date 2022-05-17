from scipy.sparse import hstack, csr_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, \
    classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score as f1_score_rep
import keras.backend as K
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.utils.np_utils import to_categorical
from numpy import unique, newaxis
from Visualization.Visualization import ROC_plot
import joblib
import os
from numpy import log1p, concatenate
from FileHandler import FileHandler as Fh


from sklearn.model_selection import train_test_split


def split_dataset(df, y_col="", test_size=0.20, with_stratify=True, shuffle=True):
    if with_stratify:
        train, val = train_test_split(df,
                                      test_size=test_size,
                                      random_state=1,
                                      stratify=df[y_col],
                                      shuffle=shuffle)
    else:
        train, val = train_test_split(df,
                                      test_size=test_size,
                                      random_state=1,
                                      stratify=df[y_col],
                                      shuffle=shuffle)
    return train, val


def get_label_encoder_obj(y):
    label_encoder = LabelBinarizer()
    return label_encoder.fit(y)


def get_y_label_encoder(label_encoder, y):
    return label_encoder.transform(y)


def get_PCA_obj(X, n_components=5):
    pca_obj = PCA(n_components=n_components)
    return pca_obj.fit(X)


def get_PCA_components(pca_obj, X):
    return pca_obj.transform(X)


def get_nb_classes(y):
    return len(unique(y))


def one_hot_encode(y, num_classes):
    return to_categorical(y, num_classes=num_classes)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def get_class_weights(y):
    class_weights = compute_class_weight('balanced',
                                         classes=unique(y.ravel()),
                                         y=y.ravel())
    return {k: v for k, v in enumerate(class_weights)}


def get_sample_weight(y):
    return compute_sample_weight('balanced',
                                 y=y.ravel())


def print_score(y_pred, y_real, label_encoder):
    print("Accuracy: ", accuracy_score(y_real, y_pred))
    print("Precision:: ", precision_score(y_real, y_pred, average="micro"))
    print("Recall:: ", recall_score(y_real, y_pred, average="micro"))
    print("F1_Score:: ", f1_score_rep(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (macro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="macro"))

    print()
    print("Macro precision_recall_fscore_support (micro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (weighted) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="weighted"))

    print()
    print("Confusion Matrix")
    cm = confusion_matrix(y_real, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in label_encoder.classes_],
                         columns=[i for i in label_encoder.classes_])
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_cm, annot=True)
    plt.show()
    print()
    print("Classification Report")
    print(classification_report(y_real, y_pred, target_names=label_encoder.classes_))


def get_prediction_results(y_true, y_hat, label_encoder, num_classes):
    y_train_ohe = one_hot_encode(y_true, num_classes)
    y_hat_ohe = one_hot_encode(y_hat, num_classes)
    ROC_plot(y_train_ohe, y_hat_ohe, label_encoder, num_classes)
    print_score(y_hat, y_true, label_encoder)


def predict(model, X_val):
    return model.predict(X_val)


def save_model_pkl(model, path_directory, file_name):
    joblib.dump(model, os.path.join(path_directory, file_name))


def load_model_pkl(file_directory):
    return joblib.load(file_directory)


def load_model_weights(model, file_directory):
    model.load_weights(file_directory)
    return model


def check_file_exists(file_path):
    if os.path.exists(file_path):
        return file_path
    raise Exception("Sorry, No file exists with this path")


def ohe_features_fit(train_df, cat_fet_cols_name):
    cat_fet_encode_dict = {}
    X_train_ohe = []
    for index, col in enumerate(cat_fet_cols_name):
        cat_fet_label_encoder = get_label_encoder_obj(train_df[col].values)
        cat_fet_encode_dict[col] = cat_fet_label_encoder
        train_res = csr_matrix(cat_fet_label_encoder.transform(train_df[col].values))
        if index == 0:
            X_train_ohe = train_res
        else:
            X_train_ohe = hstack((X_train_ohe, train_res))

    return cat_fet_encode_dict, X_train_ohe


def ohe_features_transform(test_df, cat_fet_encode_dict):
    X_test_ohe = []
    index = 0
    for key in cat_fet_encode_dict.keys():
        test_res = cat_fet_encode_dict[key].transform(test_df[key].values)
        if index == 0:
            X_test_ohe = test_res
        else:
            X_test_ohe = concatenate((X_test_ohe, test_res), axis=1)
        index += 1

    return X_test_ohe


def horizontal_feat_concatenate_csr(X_train_tfidf, x_ohe_train):
    return hstack((csr_matrix(X_train_tfidf), csr_matrix(x_ohe_train)))


def minmax_scaler_fit(X_train):
    return MinMaxScaler().fit(X_train)


def minmax_scaler_transform(scaler, X):
    return scaler.transform(X)


def log_transform(X):
    return log1p(X)


def save_features_encoder(cat_fet_encode_dict, directory):
    for key in cat_fet_encode_dict.keys():
        encoder_name = "{}.pkl".format(key)
        save_model_pkl(cat_fet_encode_dict[key], directory, encoder_name)


def load_features_encoder(feat_encoder_directory):
    cat_fet_encode_dict = {}
    paths = Fh.read_all_directory(feat_encoder_directory, extensions='pkl')
    for p in paths:
        p = p.replace('\\', '/')
        f = p.split("/")[-1]
        f = f.split(".")[0]
        cat_fet_encode_dict[f] = load_model_pkl(p)
    return cat_fet_encode_dict

