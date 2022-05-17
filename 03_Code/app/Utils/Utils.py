import numpy as np
from tensorflow.nn import softmax
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack, csr_matrix


cat_fet_cols_name = ["campaign_name", "content_name"]

numeric_col_name = ['number_of_episodes',
 'response_status',
 'domain_name_status',
 'content_name_freq',
 'trans_content_name_freq',
 'alt_content_names_0',
 'alt_content_names_1',
 'alt_content_names_2',
 'alt_content_names_3',
 'alt_content_names_4',
 'campaign_name_freq']


def padding_sequences(x_arr, max_len):
    x_arr = pad_sequences(x_arr, maxlen=max_len, value=0, padding='post')
    return x_arr


def tokenizer_encode(sentences, labels, max_length):
    input_ids = []
    attention_masks = []
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for sentence in sentences:
        bert_inp = bert_tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length,
                                              pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    if labels is not None:
        labels = np.array(labels)
    return input_ids, attention_masks, labels


def preprocessing_sgd_svm_xgboost(X_text_samples, preprocess, numeric_feat=csr_matrix([])):
    X_samples = preprocess.transform(X_text_samples)
    if numeric_feat.shape[1] != 0 and numeric_feat.shape[0] == X_samples.shape[0]:
        X_samples = hstack((csr_matrix(X_samples), csr_matrix(numeric_feat)))
    return X_samples


def predict_sgd_svm_xgboost_class(X_samples, model, label_encoder, ):
    y_hat = model.predict(X_samples)
    y_hat = label_encoder.inverse_transform(y_hat)
    return y_hat


def preprocessing_lstm_attention(X_text_samples, preprocess, numeric_feat=csr_matrix([]), model_params=None):
    X_samples = preprocess.texts_to_sequences(X_text_samples)
    X_samples = padding_sequences(X_samples, model_params["max_text_length"])

    if numeric_feat.shape[1] != 0 and numeric_feat.shape[0] == X_samples.shape[0]:
        X_samples = [X_samples, numeric_feat.todense()]
    return X_samples


def predict_lstm_attention_class(X_samples, LSTM_attention_model, label_encoder, ):
    y_hat = np.argmax(LSTM_attention_model.predict(X_samples), axis=1)
    y_hat = label_encoder.inverse_transform(y_hat)
    return y_hat


def predict_bert_class(X, label_encoder, ber_model, max_text_length=50):
    val_inp, val_mask, val_label = tokenizer_encode(X["content"].values, None, max_text_length)
    y_hat = ber_model.predict([val_inp, val_mask])
    y_hat = softmax(y_hat.logits, axis=-1)
    y_hat = np.argmax(y_hat, axis=-1)
    y_hat = label_encoder.inverse_transform(y_hat)
    return y_hat


def log_transform(X):
    return np.log1p(X)


def horizontal_feat_concatenate_csr(X_feats_1, X_feats_2):
    return hstack((csr_matrix(X_feats_1), csr_matrix(X_feats_2)))


def ohe_features_transform(df, cat_fet_encode_dict):
    X_ohe = []
    index = 0
    for key in cat_fet_encode_dict.keys():
        res = cat_fet_encode_dict[key].transform(df[key].values)
        if index == 0:
            X_ohe = res
        else:
            X_ohe = np.concatenate((X_ohe, res), axis=1)
        index += 1

    return X_ohe


def prepare_samples_features(df, using_numeric_feats, using_cat_feats, categorical_feats_encoder_dict,
                             numeric_feats_scaler):

    global cat_fet_cols_name
    global numeric_col_name

    samples_numeric_feat = csr_matrix([])
    if using_numeric_feats and using_cat_feats:

        samples_ohe_feat = ohe_features_transform(df[cat_fet_cols_name], categorical_feats_encoder_dict)
        samples_numeric_feat = log_transform(df[numeric_col_name].astype(float))
        samples_numeric_feat = numeric_feats_scaler.transform(samples_numeric_feat)
        samples_numeric_feat = horizontal_feat_concatenate_csr(samples_ohe_feat, samples_numeric_feat)

    elif using_numeric_feats:
        samples_numeric_feat = log_transform(df[numeric_col_name].astype(float))
        samples_numeric_feat = numeric_feats_scaler.transform(samples_numeric_feat)

    elif using_cat_feats:
        samples_numeric_feat = ohe_features_transform(df[cat_fet_cols_name], categorical_feats_encoder_dict)

    return samples_numeric_feat

