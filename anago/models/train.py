"""
A Keras implementation of bidirectional LSTM for named entity recognition.
"""
import itertools

import numpy as np
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model, save_model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from anago.data import conll2003


class NeuralEntityModel(object):

    def __init__(self, maxlen, max_features, word_embedding_dim, lstm_dim, num_classes, index2chunk):
        self.maxlen = maxlen
        self.max_features = max_features
        self.word_embedding_dim = word_embedding_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.index2chunk = index2chunk
        self.model = None

    def train(self, X_train, y_train, batch_size, epochs):
        self._build_model()
        self.model.fit([X_train], y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, X):
        self._check_model()
        pred = self.model.predict(X, batch_size=1)
        return pred

    def evaluate(self, X_test, y_test):
        self._check_model()
        score = self.model.evaluate(X_test, y_test, batch_size=1)
        return score

    def report(self, X_test, y_test):
        y_true = [y.argmax() for y in itertools.chain(*y_test)]
        y_pred = self.predict(X_test)
        y_pred = [y.argmax() for y in itertools.chain(*y_pred)]

        tagset = set(self.index2chunk) - {'O', '<PAD>'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(self.index2chunk)}

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

    def _build_model(self):
        word_input = Input(shape=(self.maxlen,), dtype='int32', name='word_input')
        from anago.models.keras_gensim_embeddings import word2vec_embedding_layer
        layer = word2vec_embedding_layer()
        word_emb = layer(word_input)
        #word_emb = Embedding(self.max_features, self.word_embedding_dim, input_length=self.maxlen, name='word_emb')(word_input)
        bilstm = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(word_emb)
        bilstm_d = Dropout(0.5)(bilstm)
        dense = TimeDistributed(Dense(self.num_classes, activation='softmax'))(bilstm_d)
        self.model = Model(inputs=[word_input], outputs=[dense])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(0.01),
                           metrics=['acc'])
        self.model.summary()

    def _check_model(self):
        if self.model is None:
            raise ValueError('model does NOT exist.')


if __name__ == '__main__':
    maxlen = 50  # cut texts after this number of words (among top max_features most common words)
    word_embedding_dim = 100
    lstm_dim = 100
    batch_size = 64

    print('Loading data...')
    X_train, y_train, X_test, y_test, index2word, index2chunk = conll2003.load_data(word_preprocess=lambda w: w.lower())

    max_features = len(index2word)
    nb_chunk_tags = len(index2chunk)

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
    y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
    y_train = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_train])
    y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post')
    y_test = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_test])

    model = NeuralEntityModel(maxlen, max_features, word_embedding_dim, lstm_dim, nb_chunk_tags, index2chunk)
    model.train(X_train, y_train, batch_size, epochs=3)
    model.report(X_test, y_test)
