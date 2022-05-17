import numpy as np
import time
import pickle
import os
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from livelossplot import PlotLossesKeras
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig


pickle_inp_path = "Weights\\bert_inp.pkl"
pickle_mask_path = "Weights\\bert_mask.pkl"
pickle_label_path = "Weights\\bert_label.pkl"


def tokenizer_decode(bert_tokenizer, tokenized_sequence):
    bert_tokenizer.decode(tokenized_sequence['input_ids'])


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
    labels = np.array(labels)
    return input_ids, attention_masks, labels


def save_model_pkl(input_ids, attention_masks, labels):
    pickle.dump((input_ids), open(pickle_inp_path, 'wb'))
    pickle.dump((attention_masks), open(pickle_mask_path, 'wb'))
    pickle.dump((labels), open(pickle_label_path, 'wb'))

    print('Pickle files saved as ', pickle_inp_path, pickle_mask_path, pickle_label_path)


def load_model_pkl():
    print('Loading the saved pickle files..')
    input_ids = pickle.load(open(pickle_inp_path, 'rb'))
    attention_masks = pickle.load(open(pickle_mask_path, 'rb'))
    labels = pickle.load(open(pickle_label_path, 'rb'))
    print('Input shape {} Attention mask shape {} Input label shape {}'.format(input_ids.shape, attention_masks.shape,
                                                                               labels.shape))


def build_bert_model(num_classes):
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    print('\nBert Model', bert_model.summary())
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return bert_model


def fit_bert_model(bert_model, train_inp, train_mask, train_label,
                   val_inp, val_mask, val_label, weights_dir):
    log_dir = 'tb_bert'
    model_save_path = os.path.join(weights_dir, 'bert_model.h5')
    callbacks = [ModelCheckpoint(filepath=model_save_path,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1,
                                 save_best_only=True),
                 EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5),
                 PlotLossesKeras(),
                 keras.callbacks.TensorBoard(log_dir=log_dir)]

    start_time = time.time()
    history = bert_model.fit([train_inp, train_mask], train_label,
                             batch_size=32, epochs=20,
                             validation_data=([val_inp, val_mask], val_label),
                             callbacks=callbacks)
    duration = time.time() - start_time
    print("Model take {} S to train ".format(duration))
    return bert_model, history