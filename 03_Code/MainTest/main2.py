from FileHandler import FileHandler as Fh
from FeaturesExtraction import FeatureExtraction as Fe
from Utils import Utils as Utl
from Models import lstm_attention as LSTM_clf

if __name__ == '__main__':
    csv_file_path = "../Dataset/preprocessed_df.csv"
    df = Fh.read_arabic_csv(csv_file_path)

    values = {"content": "<NONE>",
              "top_5_word": "<NONE>"}

    df = df.fillna(values)

    train, val = Utl.split_dataset(df, y_col="class", test_size=0.2, with_stratify=True, shuffle=True)
    # train, test = Utl.split_dataset(train, y_col="class", test_size=0.1, with_stratify=True, shuffle=True)

    cat_fet_cols_name = ["campaign_name", "content_name"]
    cat_fet_encode_dict, X_train_ohe = Utl.ohe_features_fit(train, cat_fet_cols_name)
    feat_directory = "../Weights/FeaturesEncoder"
    Utl.save_features_encoder(cat_fet_encode_dict, feat_directory)
    cat_fet_encode_dict = Utl.load_features_encoder(feat_directory)
    X_val_ohe = Utl.ohe_features_transform(val, cat_fet_encode_dict)
    using_cat_feats = True

    numeric_col_name = ["number_of_episodes"]
    X_train_numeric = Utl.log_transform(train[numeric_col_name].values)
    X_val_numeric = Utl.log_transform(val[numeric_col_name].values)
    minmax_scaler = Utl.minmax_scaler_fit(X_train_numeric)
    X_train_numeric = Utl.minmax_scaler_transform(minmax_scaler, X_train_numeric)
    X_val_numeric = Utl.minmax_scaler_transform(minmax_scaler, X_val_numeric)
    path_directory = "../Weights"
    file_name = "minmax_scaler.pkl"
    Utl.save_model_pkl(minmax_scaler, path_directory, file_name)
    file_name = "../Weights/minmax_scaler.pkl"
    minmax_scaler = Utl.load_model_pkl(file_name)
    using_numeric_feats = True

    val_numeric_feat = None
    train_numeric_feat = None
    if using_numeric_feats and using_cat_feats:
        train_numeric_feat = Utl.horizontal_feat_concatenate_csr(X_train_ohe, X_train_numeric)
        val_numeric_feat = Utl.horizontal_feat_concatenate_csr(X_val_ohe, X_val_numeric)

    elif using_numeric_feats:
        train_numeric_feat = X_train_numeric
        val_numeric_feat = X_val_numeric

    elif using_cat_feats:
        train_numeric_feat = X_train_ohe
        val_numeric_feat = X_val_ohe

    #max_statment_len = Fe.get_max_statment_len(train, "content")
    max_statment_len = 2000

    tokenizer, vocab_size = Fe.get_tokenizer_obj(train["content"].values)
    directory = "../Weights/"
    tokenizer_file = "tokenizer.pkl"
    Utl.save_model_pkl(tokenizer, directory, tokenizer_file)
    tokenizer_file = "../Weights/tokenizer.pkl"
    tokenizer = Utl.load_model_pkl(tokenizer_file)

    X_train = Fe.tokenize_texts_to_sequences(tokenizer, train["content"].values)
    X_train = Fe.padding_sequences(X_train, max_statment_len)

    X_val = Fe.tokenize_texts_to_sequences(tokenizer, val["content"].values)
    X_val = Fe.padding_sequences(X_val, max_statment_len)

    label_encoder = Utl.get_label_encoder_obj(train["class"])
    path_directory = "../Weights"
    file_name = "label_encoder.pkl"
    Utl.save_model_pkl(label_encoder, path_directory, file_name)
    file_name = "../Weights/label_encoder.pkl"
    label_encoder = Utl.load_model_pkl(file_name)
    train["class"] = Utl.get_y_label_encoder(label_encoder, train["class"])
    val["class"] = Utl.get_y_label_encoder(label_encoder, val["class"])

    num_classes = Utl.get_nb_classes(train["class"])

    y_train = train["class"]
    y_val = val["class"]

    class_weight = Utl.get_class_weights(y_train)

    max_text_length = X_train.shape[1]
    directory = "../Weights/"
    LSTM_clf_params = "lstm_clf_params.json"

    numeric_input_shape = train_numeric_feat.shape[1:]
    params_dict = {
        "vocab_size": vocab_size,
        "max_text_length": max_text_length,
        "numeric_input_shape": numeric_input_shape,
        "num_classes": 1
    }
    Fh.save_json_file(params_dict, directory, LSTM_clf_params)
    LSTM_clf_params = "../Weights/lstm_clf_params.json"
    params_dict = Fh.load_json_file(LSTM_clf_params)

    model = LSTM_clf.build_model(MAX_NB_WORDS=params_dict["vocab_size"],
                                 MAX_TEXT_LEN=params_dict["max_text_length"],
                                 input_shape=numeric_input_shape,
                                 use_only_lstm=False,
                                 learning_rate=0.001)
    weights_path = "../Weights"

    if train_numeric_feat is not None and val_numeric_feat is not None:
        X_train = [X_train, train_numeric_feat.todense()]
        X_val = [X_val, val_numeric_feat.todense()]

    model, history = LSTM_clf.train_model(model,
                                          X_train, y_train,
                                          X_val, y_val,
                                          class_weight,
                                          weights_path)

    model.load_weights("../Weights/lstm_attention_model_weights.h5")
    y_hat = model.predict(X_val)
    y_hat = (y_hat >= 0.5).astype(int)
    Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
