"""
A Keras implementation of LSTM-CRF for named-entity recognition.

References
--
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
"Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
https://arxiv.org/abs/1603.01360
"""
import itertools

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Reshape, Flatten
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model, save_model
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

from anago.data import loader


class NeuralEntityModel(object):

    def __init__(self, maxlen, max_features, word_embedding_dim, lstm_dim, num_classes, index2chunk, n_char, max_length):
        self.char_embedding_size = 25
        self.char_vocab_size = n_char
        self.char_lstm_dim = self.char_embedding_size
        self.indices_tag = index2chunk
        self.lstm_dim = lstm_dim
        self.max_sent_len = maxlen
        self.max_word_len = max_length
        self.num_classes = num_classes
        self.word_embedding_size = word_embedding_dim
        self.word_vocab_size = max_features
        self.transition_matrix = None

        self.model = None

    def train(self, X_word_train, X_char_train, y_train, batch_size, epochs):
        self._build_model()
        self.model.fit([X_word_train, X_char_train], y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, X):
        self._check_model()
        pred = self.model.predict(X, batch_size=1)
        return pred

    def evaluate(self, X_word_test, X_char_test, y_test):
        self._check_model()
        score = self.model.evaluate([X_word_test, X_char_test], y_test, batch_size=1)
        return score

    def report(self, X_word_test, X_char_test, y_test):
        y_true = [y.argmax() for y in itertools.chain(*y_test)]
        transition_matrix = self.transition_matrix.eval(session=K.get_session())
        y_pred = self.predict([X_word_test, X_char_test])
        y_pred, _ = zip(*[tf.contrib.crf.viterbi_decode(y_, transition_matrix) for y_ in y_pred])
        y_pred = list(itertools.chain(*y_pred))

        tagset = set(self.indices_tag) - {'O', '<PAD>'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(self.indices_tag)}

        print(classification_report(
            y_true,
            y_pred,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        ))

    def save(self, filepath):
        self._check_model()
        save_model(self.model, filepath)

    def load(self, filepath):
        self.model = load_model(filepath=filepath)

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_length = tf.constant(shape=(batch_size,), value=self.max_sent_len, dtype=tf.int32)
        if self.transition_matrix:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length, self.transition_matrix)
        else:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length)
        self.transition_matrix = transition_params
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def _build_model(self):
        dropout = 0.5
        char_input = Input(shape=(self.max_sent_len, self.max_word_len), dtype='int32', name='char_input')
        char_emb = Embedding(input_dim=self.char_vocab_size, output_dim=self.char_embedding_size, name='char_emb')(char_input)

        print(char_emb)
        #char_emb = Reshape((self.max_sent_len*self.max_word_len, self.char_embedding_size))(char_emb)
        #print(char_emb)
        #a = TimeDistributed(LSTM(output_dim=self.char_lstm_dim, return_sequences=True, return_state=True))(char_emb)
        #b = TimeDistributed(LSTM(output_dim=self.char_lstm_dim, return_sequences=True, return_state=True, go_backwards=True))(char_emb)
        #print(a)
        #chars_emb = Concatenate(axis=-1)([a, b])

        #print(chars_emb)

        chars_emb = TimeDistributed(Bidirectional(LSTM(self.char_lstm_dim,
                                       dropout=dropout, recurrent_dropout=dropout), name='input_bilstm'))(char_emb)

        word_input = Input(shape=(self.max_sent_len,), dtype='int32', name='word_input')
        word_emb = Embedding(input_dim=self.word_vocab_size, output_dim=self.word_embedding_size,
                             input_length=self.max_sent_len, name='word_emb')(word_input)

        concat_layer = Concatenate(axis=-1)
        word_embeddings = concat_layer([word_emb, chars_emb])
        word_embeddings = Dropout(dropout)(word_embeddings)

        bilstm = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(word_embeddings)
        bilstm = Dropout(dropout)(bilstm)
        dense1 = TimeDistributed(Dense(self.lstm_dim, activation='tanh'))(bilstm)
        dense = TimeDistributed(Dense(self.num_classes))(dense1)
        # dense = TimeDistributed(Dense(self.num_classes, activation='softmax'))(bilstm)

        self.model = Model(inputs=[word_input, char_input], outputs=[dense])
        self.model.compile(loss=self.loss,
                           optimizer=RMSprop(lr=0.01, clipvalue=5.0),
                           metrics=['acc'])
        self.model.summary()

    def _check_model(self):
        if self.model is None:
            raise ValueError('model does NOT exist.')


if __name__ == '__main__':
    max_sent_len = 50  # cut texts after this number of words (among top max_features most common words)
    max_word_len = 20
    word_embedding_size = 100
    lstm_dim = 100
    batch_size = 64
    char_dim = 25

    print('Loading data...')
    X_word_train, y_train = loader.load_file(loader.TRAIN_DATA)
    X_word_test, y_test = loader.load_file(loader.TEST_DATA)

    indices_word, word_indices = loader.word_mapping(X_word_train, str.lower)
    indices_char, char_indices = loader.char_mapping(X_word_train, str.lower)
    indices_tag, tag_indices = loader.tag_mapping(y_train)

    word_vocab_size = len(indices_word)
    char_vocab_size = len(indices_char)
    num_tags = len(indices_tag)

    X_word_train, X_char_train = loader.prepare_sentence(X_word_train, word_indices, char_indices, lower=True)
    X_word_test, X_char_test = loader.prepare_sentence(X_word_test, word_indices, char_indices, lower=True)
    y_train, y_test = loader.convert_tag_str(y_train, tag_indices), loader.convert_tag_str(y_test, tag_indices)

    X_word_train = sequence.pad_sequences(X_word_train, maxlen=max_sent_len, padding='post')
    X_word_test = sequence.pad_sequences(X_word_test, maxlen=max_sent_len, padding='post')
    X_char_train = loader.pad_chars(X_char_train, max_word_len, max_sent_len)
    X_char_test = loader.pad_chars(X_char_test, max_word_len, max_sent_len)

    y_train = sequence.pad_sequences(y_train, maxlen=max_sent_len, padding='post')
    #y_train = np.asarray(y_train, dtype=np.int32)
    y_train = np.asarray([to_categorical(y, num_classes=num_tags) for y in y_train], dtype=np.int32)
    y_test = sequence.pad_sequences(y_test, maxlen=max_sent_len, padding='post')
    #y_test = np.asarray(y_test, dtype=np.int32)
    y_test = np.asarray([to_categorical(y, num_classes=num_tags) for y in y_test], dtype=np.int32)
    print(X_word_train.shape)
    print(X_char_train.shape)
    print(y_train.shape)
    r = (len(X_word_train) // batch_size) * batch_size
    #r = batch_size
    X_word_train = X_word_train[:r]
    X_char_train = X_char_train[:r]
    y_train = y_train[:r]
    #X_word_test = X_word_test[:r]
    #X_char_test = X_char_test[:r]
    #y_test = y_test[:r]

    model = NeuralEntityModel(max_sent_len, word_vocab_size, word_embedding_size, lstm_dim, num_tags, indices_tag,
                              char_vocab_size, max_word_len)
    model.train(X_word_train, X_char_train, y_train, batch_size, epochs=10)
    model.report(X_word_test, X_char_test, y_test)
