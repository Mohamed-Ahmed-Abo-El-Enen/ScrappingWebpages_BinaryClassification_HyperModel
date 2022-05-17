from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import nltk

nltk.download('punkt')
nltk.download('wordnet')

oov_tok = "<oov_tok>"


def CountVectorizer_fit(X_train, ngram_range=(1,1)):
    count_vect = CountVectorizer(ngram_range=ngram_range)
    return count_vect.fit(X_train)


def CountVectorizer_transform(count_vect, X):
    return count_vect.transform(X)


def TfidfTransformer_fit(X_train_counts, use_idf=True):
    tf_transformer = TfidfTransformer(use_idf=use_idf)
    return tf_transformer.fit(X_train_counts)


def TfidfTransformer_transform(tf_transformer, X_counts):
    return tf_transformer.transform(X_counts)


def TfidfVectorizer_fit(X_train_text, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=1000):
    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                max_features=max_features)
    return tfidf_vec.fit(X_train_text)


def TfidfVectorizer_transform(tf_transformer, X_train_text):
    return tf_transformer.transform(X_train_text)


def fit_preprocessing_pipeline(X_train, using_tfidf_vec=True, ngram_range=(1, 1), use_idf=True, max_df=1.0, min_df=1,
                               max_features=1000):
    if using_tfidf_vec:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                      max_features=max_features))])
        pipeline.fit(X_train)
    else:
        pipeline = Pipeline([
            ("vect", CountVectorizer(ngram_range=ngram_range)),
            ("tfidf", TfidfTransformer(use_idf=use_idf))])
        pipeline.fit(X_train)
    return pipeline


def transform_preprocessing_pipeline(pipeline, X):
    return pipeline.transform(X)


def get_max_sequences_len(df, col):
    return max([len(x.split()) for x in df[col].values])


def get_tokenizer_obj(text_list):
    tokenizer = Tokenizer(lower=True, split=" ", oov_token=oov_tok)
    tokenizer.fit_on_texts(text_list)
    return tokenizer, len(tokenizer.word_index)


def tokenize_texts_to_sequences(tokenizer, text_list):
    return tokenizer.texts_to_sequences(text_list)


def padding_sequences(x_arr, max_len):
    x_arr = pad_sequences(x_arr, maxlen=max_len, value=0, padding='post')
    return x_arr


def get_max_statment_len(df, col):
    return max([len(text.split()) for text in df[col]])

