"""
A Keras implementation of LSTM-CRF for named-entity recognition.

References
--
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
"Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
https://arxiv.org/abs/1603.01360
"""
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

from anago.models.base_model import BaseModel


class BiLSTMCrf(BaseModel):

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_length = tf.constant(shape=(self.config.batch_size,), value=self.config.num_steps, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length)
        self.transition_matrix = transition_params
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def _build_model(self):
        # build character based word embedding
        char_input = Input(shape=(self.config.num_steps, self.max_word_len), dtype='int32')
        x1 = Embedding(input_dim=self.char_vocab_size,
                       output_dim=self.char_embedding_size)(char_input)
        x1 = TimeDistributed(Bidirectional(LSTM(self.char_lstm_dim,
                                                dropout=self.config.dropout,
                                                recurrent_dropout=self.config.dropout)))(x1)

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
        # pred = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)

        self.model = Model(inputs=[word_input, char_input], outputs=[x])
        self.model.compile(loss=self.loss,
                           optimizer=RMSprop(lr=self.config.learning_rate,
                                             clipvalue=self.config.max_grad_norm),
                           metrics=['acc'])
        self.model.summary()
