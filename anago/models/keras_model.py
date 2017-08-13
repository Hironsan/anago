"""
A Keras implementation of LSTM-CRF for named-entity recognition.

References
--
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
"Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
https://arxiv.org/abs/1603.01360
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import to_categorical

from anago.models.base_model import BaseModel


class LSTMCrf(BaseModel):

    def predict(self, X):
        scores = self.model.predict(X, batch_size=1)
        y_pred = []
        for score in scores:
            pred, _ = tf.contrib.crf.viterbi_decode(score, self.transition_matrix)
            y_pred.append(to_categorical(pred, num_classes=self.ntags))

        return np.asarray(y_pred)

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_length = tf.constant(shape=(self.config.batch_size,), value=self.config.num_steps, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        self.transition_matrix = K.eval(transition_params)

        return loss

    def build(self):
        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=self.config.char_vocab_size,
                                    output_dim=self.config.char_dim,
                                    mask_zero=True
                                    )(char_ids)
        s = K.shape(char_embeddings)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config.char_dim)))(char_embeddings)
        char_embeddings = Bidirectional(LSTM(units=self.config.char_lstm_size,
                                             ))(char_embeddings)
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.config.char_lstm_size]))(char_embeddings)

        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        word_embeddings = Embedding(input_dim=self.embeddings.shape[0],
                                    output_dim=self.embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[self.embeddings])(word_ids)
        # combine characters and word
        x = Concatenate(axis=-1)([char_embeddings, word_embeddings])

        x = Bidirectional(LSTM(units=self.config.lstm_size, return_sequences=True))(x)
        x = Dropout(self.config.dropout)(x)
        pred = TimeDistributed(Dense(self.ntags))(x)

        model = Model(inputs=[word_ids, char_ids], outputs=[pred])
        model.summary()
        return model
