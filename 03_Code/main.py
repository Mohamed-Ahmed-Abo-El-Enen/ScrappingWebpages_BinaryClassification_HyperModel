from app.app import *
from Utils.Utils import *
from Models import lstm_attention as LSTM_clf
from Models import basic_bert_model as BERT_CLF
from FileHandler.FileHandler import load_json_file

if __name__ == "__main__":
    app.config["using_cat_feats"] = True
    app.config["using_numeric_feats"] = True

    # preprocessing (scaler, encoder) pkl
    app.config["encoded_class"] = load_model_pkl(check_file_exists("Weights/label_encoder.pkl"))
    app.config["numeric_feats_scaler"] = load_model_pkl(check_file_exists("Weights/minmax_scaler.pkl"))
    app.config["categorical_feats_encoder_dict"] = load_features_encoder("Weights/FeaturesEncoder")

    # Ml Models (model, pipline) pkl
    app.config["preprocessing_pipeline"] = load_model_pkl(check_file_exists("Weights/preprocessing_pipeline.pkl"))
    app.config["SGD_clf"] = load_model_pkl(check_file_exists("Weights/SGD_clf.pkl"))
    app.config["SVM_clf"] = load_model_pkl(check_file_exists("Weights/SVM_clf.pkl"))
    app.config["XGBoost_clf"] = load_model_pkl(check_file_exists("Weights/XGBoost_clf.pkl"))

    # lstm Model (weights and parameters) pkl
    app.config["tokenizer"] = load_model_pkl(check_file_exists("Weights/tokenizer.pkl"))
    LSTM_clf_params = load_json_file(check_file_exists("Weights/lstm_clf_params.json"))
    app.config["LSTM_clf_params"] = LSTM_clf_params
    lstm_model = LSTM_clf.build_model(MAX_NB_WORDS=LSTM_clf_params["vocab_size"],
                                      MAX_TEXT_LEN=LSTM_clf_params["max_text_length"],
                                      input_shape=LSTM_clf_params["numeric_input_shape"],
                                      use_only_lstm=True)
    lstm_model = load_model_weights(lstm_model, check_file_exists("Weights/lstm_attention_model_weights.h5"))
    app.config["lstm_model"] = lstm_model

    # bert model
    bert_model = BERT_CLF.build_bert_model(2)
    bert_model = load_model_weights(bert_model, check_file_exists("Weights/bert_model.h5"))
    app.config["bert_model"] = bert_model

    app.run()