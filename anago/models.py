"""
Model definition.
"""
import json

import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation
from keras.layers.merge import Concatenate
from keras.models import Model

from anago.layers import ChainCRF


class BaseModel(object):

    def __init__(self):
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            params = {name: val for name, val in vars(self).items() if name not in {'_loss', 'model'}}
            json.dump(params, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, weights_file, params_file):
        cls.load_params(params_file)
        cls.load_weights(weights_file)

    @classmethod
    def load_weights(cls, file_path):
        cls.model.load_weights(filepath=file_path)

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)
            self = cls(**params)
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


class BiLSTMCRF(BaseModel):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                 word_lstm_units=100, dropout=0.5, char_feature=True, use_crf=True,
                 word_vocab_size=10000, char_vocab_size=100, embeddings=None, ntags=None):
        super(BiLSTMCRF).__init__()
        self._char_emb_size = char_emb_size
        self._word_emb_size = word_emb_size
        self._char_lstm_units = char_lstm_units
        self._word_lstm_units = word_lstm_units
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._dropout = dropout
        self._char_feature = char_feature
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._ntags = ntags
        self._loss = None

    def build_model(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_emb_size,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings])(word_ids)

        # build character based word embedding
        if self._char_feature:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32')
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_emb_size,
                                        mask_zero=True
                                        )(char_ids)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self._char_emb_size)))(char_embeddings)

            fwd_state = LSTM(self._char_lstm_units, return_state=True)(char_embeddings)[-2]
            bwd_state = LSTM(self._char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
            # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self._char_lstm_units]))(char_embeddings)

            # combine characters and word
            x = Concatenate(axis=-1)([word_embeddings, char_embeddings])
        else:
            x = word_embeddings

        x = Dropout(self._dropout)(x)
        x = Bidirectional(LSTM(units=self._word_lstm_units, return_sequences=True))(x)
        x = Dropout(self._dropout)(x)
        x = Dense(self._word_lstm_units, activation='tanh')(x)
        x = Dense(self._ntags)(x)

        if self._use_crf:
            crf = ChainCRF()
            self._loss = crf.loss
            pred = crf(x)
        else:
            pred = Activation('softmax')(x)

        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        if self._char_feature:
            self.model = Model(inputs=[word_ids, char_ids, sequence_lengths], outputs=[pred])
        else:
            self.model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])

    def get_loss(self):
        return self._loss
