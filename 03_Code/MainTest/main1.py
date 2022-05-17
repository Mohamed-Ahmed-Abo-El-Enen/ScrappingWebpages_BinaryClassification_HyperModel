from FileHandler import FileHandler as Fh
from FeaturesExtraction import FeatureExtraction as Fe
from Utils import Utils as Utl
from Models import SGDClassifier_Model as SGD_clf
from Models import SVMClassifier_Model as SVM_clf
from Models import XGBoostClassifier_Model as XGBoost_clf


if __name__ == '__main__':
    csv_file_path = "..\Dataset\preprocessed_df.csv"
    df = Fh.read_arabic_csv(csv_file_path)

    values = {"content": "<NONE>",
              'top_5_word': "<NONE>"}
    df = df.fillna(values)

    #df["content"] = df[["campaign_name", "content_name", "content"]].agg(' '.join, axis=1)

    train, val = Utl.split_dataset(df, y_col="class", test_size=0.2, with_stratify=True, shuffle=True)
    X_train = train["content"].values
    y_train = train["class"].values
    X_val = val["content"].values
    y_val = val["class"].values

    num_classes = Utl.get_nb_classes(y_train)
    label_encoder = Utl.get_label_encoder_obj(y_train)
    y_train = Utl.get_y_label_encoder(label_encoder, y_train)
    y_val = Utl.get_y_label_encoder(label_encoder, y_val)

    #count_vect = Fe.CountVectorizer_fit(X_train, ngram_range=(1, 1))
    #X_train_counts = Fe.CountVectorizer_transform(count_vect, X_train)
    #X_val_counts = Fe.CountVectorizer_transform(count_vect, X_val)
    #tf_transformer = Fe.TfidfTransformer_fit(X_train_counts)
    #X_train_tfidf = Fe.TfidfTransformer_transform(tf_transformer, X_train_counts)
    #X_val_tfidf = Fe.TfidfTransformer_transform(tf_transformer, X_val_counts)

    #tf_vectorizer = Fe.TfidfVectorizer_fit(X_train, max_df=0.5, min_df=10, max_features=5000)
    #X_train_tfidf = Fe.TfidfVectorizer_transform(tf_vectorizer, X_train)
    #X_val_tfidf = Fe.TfidfVectorizer_transform(tf_vectorizer, X_val)

    preprocessing_pipeline = Fe.fit_preprocessing_pipeline(X_train, using_tfidf_vec=True,
                                                           ngram_range=(1, 1), use_idf=True,
                                                           max_df=0.5, min_df=10, max_features=5000)
    directory = "../Weights"
    preprocessing_pipeline_file = "preprocessing_pipeline.pkl"
    Utl.save_model_pkl(preprocessing_pipeline, directory, preprocessing_pipeline_file)
    preprocessing_pipeline_file = "../Weights/preprocessing_pipeline.pkl"
    preprocessing_pipeline = Utl.load_model_pkl(preprocessing_pipeline_file)
    X_train_tfidf = Fe.transform_preprocessing_pipeline(preprocessing_pipeline, X_train)
    X_val_tfidf = Fe.transform_preprocessing_pipeline(preprocessing_pipeline, X_val)

    cat_fet_cols_name = ["campaign_name", "content_name"]
    cat_fet_encode_dict, X_train_ohe = Utl.ohe_features_fit(train, cat_fet_cols_name)
    X_val_ohe = Utl.ohe_features_transform(val, cat_fet_encode_dict)

    X_train_tfidf = Utl.horizontal_feat_concatenate_csr(X_train_tfidf, X_train_ohe)
    X_val_tfidf = Utl.horizontal_feat_concatenate_csr(X_val_tfidf, X_val_ohe)

    model, _ = SGD_clf.fit_SGDClassifier(X_train_tfidf, y_train)
    directory = "../Weights"
    SGD_clf_file = "SGD_clf.pkl"
    Utl.save_model_pkl(model, directory, SGD_clf_file)
    SGD_clf_file = "../Weights/SGD_clf.pkl"
    model = Utl.load_model_pkl(SGD_clf_file)
    y_hat = model.predict(X_val_tfidf)
    Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
    y_hat = Utl.predict(model, X_val_tfidf)

    #model, _ = SVM_clf.fit_SVMClassifier(X_train_tfidf, y_train)
    #directory = "../Weights"
    #SVM_clf_file = "SVM_clf.pkl"
    #Utl.save_model_pkl(model, directory, SVM_clf_file)
    #SVM_clf_file = "../Weights/SVM_clf.pkl"
    #model = Utl.load_model_pkl(SVM_clf_file)
    #y_hat = model.predict(X_val_tfidf)
    #Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
    #y_hat = Utl.predict(model, X_val_tfidf)

    #model, _ = XGBoost_clf.fit_XGBClassifier(X_train_tfidf, y_train)
    #directory = "../Weights"
    #XGBoost_clf_file = "XGBoost_clf.pkl"
    #Utl.save_model_pkl(model, directory, XGBoost_clf_file)
    #XGBoost_clf_file = "../Weights/XGBoost_clf.pkl"
    #model = Utl.load_model_pkl(XGBoost_clf_file)
    #y_hat = model.predict(X_val_tfidf)
    #Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
    #y_hat = Utl.predict(model, X_val_tfidf)