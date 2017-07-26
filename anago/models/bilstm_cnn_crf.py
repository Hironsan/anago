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
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Conv2D, MaxPool2D, Reshape
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

from anago.models.base_model import BaseModel


class BiLSTMCNNCrf(BaseModel):

    def predict(self, X):
        scores = self.model.predict(X, batch_size=1)
        y_pred = []
        for score in scores:
            pred, _ = tf.contrib.crf.viterbi_decode(score, self.transition_matrix)
            y_pred.append(pred)

        return np.asarray(y_pred)

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_length = tf.constant(shape=(self.config.batch_size,), value=self.config.num_steps, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        self.transition_matrix = K.eval(transition_params)

        return loss

    def _build_model(self):
        # build character based word embedding
        char_input = Input(shape=(self.config.num_steps, self.config.max_word_len), dtype='int32')
        x1 = TimeDistributed(Embedding(input_dim=self.config.char_vocab_size,
                                       output_dim=self.config.char_embedding_size)
                             )(char_input)
        x1 = Conv2D(self.config.nb_filters, (1, self.config.nb_kernels), activation='tanh')(x1)
        x1 = MaxPool2D((1, self.config.max_word_len - self.config.nb_kernels + 1))(x1)
        x1 = Reshape((self.config.num_steps, self.config.nb_filters))(x1)

        # build word embedding
        word_input = Input(shape=(self.config.num_steps,), dtype='int32')
        x2 = Embedding(input_dim=self.embeddings.shape[0],
                       output_dim=self.embeddings.shape[1],
                       input_length=self.config.num_steps,
                       weights=[self.embeddings])(word_input)

        # combine characters and word
        x = Concatenate(axis=-1)([x1, x2])
        x = Dropout(self.config.dropout)(x)

        x = Bidirectional(LSTM(units=self.config.hidden_size,
                               return_sequences=True,
                               dropout=self.config.dropout,
                               recurrent_dropout=self.config.dropout))(x)
        x = TimeDistributed(Dense(self.config.hidden_size, activation='tanh'))(x)
        x = TimeDistributed(Dense(self.ntags))(x)
        # pred = TimeDistributed(Dense(self.ntags, activation='softmax'))(x)

        model = Model(inputs=[word_input, char_input], outputs=[x])
        model.compile(loss=self.loss,
                      optimizer=RMSprop(lr=self.config.learning_rate,
                                        clipvalue=self.config.max_grad_norm),
                      metrics=['acc'])
        model.summary()
        self.model = model
