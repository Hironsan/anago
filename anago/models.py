"""
Model definition.
"""
import json

import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model

from anago.layers import CRF


class BaseModel(object):

    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in {'_loss', 'model', '_embeddings'}}
            json.dump(params, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.build()
        self.load_weights(weights_file)

        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)

        return params

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

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._loss = None

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        lengths = Input(batch_shape=(None, None), dtype='int32')
        inputs = [word_ids]
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings])(word_ids)

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True
                                        )(char_ids)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self._char_embedding_dim)))(char_embeddings)

            fwd_state = LSTM(self._char_lstm_size, return_state=True)(char_embeddings)[-2]
            bwd_state = LSTM(self._char_lstm_size, return_state=True, go_backwards=True)(char_embeddings)[-2]
            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
            # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self._char_lstm_size]))(char_embeddings)

            # combine characters and word
            word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])
        inputs.append(lengths)

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dropout(self._dropout)(z)
        z = Dense(self._fc_dim, activation='tanh')(z)
        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            self._loss = crf.loss_function
            pred = crf(z)
        else:
            self._loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        self.model = Model(inputs=inputs, outputs=pred)

    def get_loss(self):
        return self._loss
