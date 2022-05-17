from flask import Flask, render_template, request
from .Utils.Utils import *
from .Utils.Preprocess import *


def get_model_transformers(classifier_type):
    if classifier_type == "LSTM":
        label_encoder = app.config["encoded_class"]
        preprocess = app.config["tokenizer"]
        model = app.config["lstm_model"]
    elif classifier_type == "BERT":
        label_encoder = app.config["encoded_class"]
        preprocess = app.config["tokenizer"]
        model = app.config["bert_model"]
    else:
        label_encoder = app.config["encoded_class"]
        preprocess = app.config["preprocessing_pipeline"]
        if classifier_type == "SGD":
            model = app.config["SGD_clf"]
        elif classifier_type == "SGD":
            model = app.config["SVM_clf"]
        else:
            model = app.config["XGBoost_clf"]

    return label_encoder, preprocess, model


def predict_samples_class(df, label_encoder, preprocess, model, classifier_type, index=None):
    df = generate_clean_df(df)
    if classifier_type == "LSTM":
        samples_numeric_feat = prepare_samples_features(df,
                                                        model_dict["numeric_features"],
                                                        model_dict["categorical_features"],
                                                        app.config["categorical_feats_encoder_dict"],
                                                        app.config["numeric_feats_scaler"])

        X_samples = preprocessing_lstm_attention(df["content"], preprocess, numeric_feat=samples_numeric_feat,
                                                 model_params=app.config["LSTM_clf_params"])
        res = predict_lstm_attention_class(X_samples, model, label_encoder).ravel().astype(str)
        if index is not None:
            return res[index]
        return res

    elif classifier_type == "BERT":
        res = predict_bert_class(df, label_encoder, model, 50).astype(str)
        if index is not None:
            return res[index]
        return res
    else:
        samples_numeric_feat = prepare_samples_features(df,
                                                        model_dict["numeric_features"],
                                                        model_dict["categorical_features"],
                                                        app.config["categorical_feats_encoder_dict"],
                                                        app.config["numeric_feats_scaler"])

        X_samples = preprocessing_sgd_svm_xgboost(df["content"], preprocess, numeric_feat=samples_numeric_feat)
        res = predict_sgd_svm_xgboost_class(X_samples, model, label_encoder).astype(str)
        if index is not None:
            return res[index]
        return res


app = Flask(__name__)

df = pd.DataFrame()
sample_dict = [{"page_source_path":"",
               "link_url":"",
               "link_domain_name":"",
               "content_name":"",
               "trans_content_name":"",
               "alt_content_names":"",
               "campaign_name":"",
               "number_of_episodes":0,
               "class":""
            }]

model_dict = {
    "classifier_type": "SGD",
    "categorical_features": "",
    "numeric_features": ""
}


@app.route('/', methods=["GET", "POST"])
def run():
    request_type_str = request.method
    global sample_dict, model_dict
    if request_type_str == "POST":
        sample_dict[0]["page_source_path"] = request.form['page_source_path']
        sample_dict[0]["link_url"] = request.form['link_url']
        sample_dict[0]["link_domain_name"] = request.form['link_domain_name']
        sample_dict[0]["content_name"] = request.form['content_name']
        sample_dict[0]["trans_content_name"] = request.form['trans_content_name']
        sample_dict[0]["alt_content_names"] = request.form['alt_content_names']
        sample_dict[0]["campaign_name"] = request.form['campaign_name']
        sample_dict[0]["number_of_episodes"] = request.form['number_of_episodes']

        model_dict["classifier_type"] = request.form['classifier_type']

        if "categorical_features" in request.form:
            model_dict["categorical_features"] = True
        else:
            model_dict["categorical_features"] = False

        if "numeric_features" in request.form:
            model_dict["numeric_features"] = True
        else:
            model_dict["numeric_features"] = False

        label_encoder, pipeline, model = get_model_transformers(model_dict["classifier_type"])
        df = json_2_df(sample_dict)
        sample_dict[0]["class"] = predict_samples_class(df, label_encoder, pipeline, model,
                                                     model_dict["classifier_type"], index=0)
    return render_template("index.html",
                           page_source_path_val=sample_dict[0]["page_source_path"],
                           link_url_val=sample_dict[0]["link_url"],
                           link_domain_name_val=sample_dict[0]["link_domain_name"],
                           content_name_val=sample_dict[0]["content_name"],
                           trans_content_name_val=sample_dict[0]["trans_content_name"],
                           alt_content_names_val=sample_dict[0]["alt_content_names"],
                           campaign_name_val=sample_dict[0]["campaign_name"],
                           number_of_episodes_val=sample_dict[0]["number_of_episodes"],
                           res=sample_dict[0]["class"],
                           classifier_type_val=model_dict["classifier_type"],
                           categorical_features_val=model_dict["categorical_features"],
                           numeric_features_val=model_dict["numeric_features"])


@app.route("/json", methods=["POST"])
def json_example():
    global sample_dict, model_dict
    if request.is_json:
        label_encoder, pipeline, model = get_model_transformers(model_dict["classifier_type"])
        sample_dict = request.get_json()
        df = json_2_df(sample_dict)
        df["class"] = predict_samples_class(df, label_encoder, pipeline, model,
                                            model_dict["classifier_type"])
        print(df)
        df.to_csv("../Results/json_result.csv", index=False)
        return "JSON received!", 200
    else:
        return "Request was not JSON", 400


