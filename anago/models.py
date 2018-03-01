import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split

from anago.layers import ChainCRF
from anago.reader import batch_iter
from anago.callbacks import get_callbacks


class BaseModel(object):

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def score(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)


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
                 word_vocab_size=10000, char_vocab_size=100, embeddings=None, ntags=None,
                 batch_size=32, optimizer='adam', max_epoch=15, early_stopping=False):
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
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._max_epoch = max_epoch
        self._early_stopping = early_stopping

    def _build_model(self):
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
            self.crf = ChainCRF()
            pred = self.crf(x)
        else:
            pred = Activation('softmax')(x)

        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        if self._char_feature:
            self.model = Model(inputs=[word_ids, char_ids, sequence_lengths], outputs=[pred])
        else:
            self.model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])

    def fit(self, X, y):
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(x_train,
                                                y_train,
                                                self._batch_size,
                                                preprocessor=self.preprocessor)
        valid_steps, valid_batches = batch_iter(x_valid,
                                                y_valid,
                                                self._batch_size,
                                                preprocessor=self.preprocessor)

        self._build_model()

        if self._use_crf:
            self.model.compile(loss=self.crf.loss,
                               optimizer=self._optimizer)
        else:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self._optimizer)

        # Prepare callbacks
        """
        callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                  tensorboard=self.tensorboard,
                                  eary_stopping=self._early_stopping,
                                  valid=(valid_steps, valid_batches, self.preprocessor))
        """
        callbacks = []

        # Train the model
        self.model.fit_generator(generator=train_batches,
                                 steps_per_epoch=train_steps,
                                 epochs=self._max_epoch,
                                 callbacks=callbacks)
