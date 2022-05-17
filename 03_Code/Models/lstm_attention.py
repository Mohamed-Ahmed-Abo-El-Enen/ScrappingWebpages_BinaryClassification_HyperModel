from tensorflow.keras.optimizers import Adam
from keras.layers import Layer
from keras.layers import LSTM, Embedding, Dense, Input, Dropout, Bidirectional, concatenate, Reshape, Permute, Lambda
from keras.layers import Flatten
from keras.models import Model
from keras.metrics import Recall, Precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from livelossplot import PlotLossesKeras
from os.path import join
import keras.backend as K
import time
from Utils.Evaluation import *


class attention(Layer):
    def __init__(self, return_sequences=True, layer_name=""):
        self.layer_name = layer_name
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name=self.layer_name+"_att_weight", 
                               shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name=self.layer_name+"_att_bias", 
                               shape=(input_shape[1],1),
                               initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


def numeric_model(input_shape):
    input = Input(shape=input_shape)
    x = Dense(64, activation="relu")(input)
    x = Dense(32, activation="relu")(x)
    x = Model(inputs=input, outputs=x)
    return x


def lstm_model(MAX_NB_WORDS,
                MAX_TEXT_LEN):
    EMBEDDING_DIM = 128
    inputs = Input(name='inputs', shape=[MAX_TEXT_LEN])
    layer = Embedding(MAX_NB_WORDS+1, EMBEDDING_DIM, input_length=MAX_TEXT_LEN)(inputs)
    units = 512
    layer = Bidirectional(LSTM(units, return_sequences=True))(layer)
    layer = attention(layer_name="attention_"+str(units),
                      return_sequences=False)(layer)
    #layer = Dropout(0.3)(layer)

    layer = Flatten()(layer)
    #layer = Dropout(0.1)(layer)
    #layer = Dense(512, activation="relu")(layer)
    model = Model(inputs=[inputs], outputs=layer)
    return model


def build_model(MAX_NB_WORDS, MAX_TEXT_LEN, input_shape,
                use_only_lstm,
                learning_rate=2e-5,
                epsilon=1e-08):

    m1 = lstm_model(MAX_NB_WORDS, MAX_TEXT_LEN)

    if not use_only_lstm:
        m2 = numeric_model(input_shape)
        combined = concatenate([m1.output, m2.output])
        layer = Dense(256, activation="relu")(combined)
        output = Dense(1, activation='sigmoid')(layer)  # ,
                    # kernel_regularizer='l1')(layer)
        model = Model(inputs=[m1.input, m2.input], outputs=output)

    else:
        output = Dense(1, activation='sigmoid')(m1.output)
        model = Model(inputs=m1.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon),
                  loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), F1_score,
                                                       Precision_macro_score, Recall_macro_score, F1_macro_score])
    print(model.summary())
    plot_model(model, to_file="../Figures/lstm_model.jpg", show_shapes=True)
    return model


def train_model(model, X_train, y_train, X_val, y_val,
                class_weight,
                weights_dir,
                epochs=20,
                mini_batch_size=32,
                ):
    model_weights_file_path = join(weights_dir, "lstm_attention_model_weights.h5")
    checkpoint = ModelCheckpoint(filepath=model_weights_file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
    plotlosses = PlotLossesKeras()
    call_backs = [checkpoint, early_stopping, lr_reduce, plotlosses]
    start_time = time.time()

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=mini_batch_size,
                        callbacks=call_backs,
                        class_weight=class_weight,
                        verbose=1)

    duration = time.time() - start_time
    print("Model take {} S to train ".format(duration))
    return model, history
