import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model


class BaseModel(object):

    def __init__(self, config, embeddings, ntags):
        self.config = config
        self.embeddings = embeddings
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)


class SeqLabeling(BaseModel):
    """A Keras implementation of BiLSTM-CRF for named-entity recognition.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, embeddings=None, ntags=None):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        if embeddings is None:
            word_embeddings = Embedding(input_dim=config.vocab_size,
                                        output_dim=config.word_dim,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=embeddings.shape[0],
                                        output_dim=embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[embeddings])(word_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_dim,
                                    mask_zero=True
                                    )(char_ids)
        s = K.shape(char_embeddings)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], config.char_dim)))(char_embeddings)

        fwd_state = LSTM(config.char_lstm_size, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(config.char_lstm_size, return_state=True, go_backwards=True)(char_embeddings)[-2]
        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * config.char_lstm_size]))(char_embeddings)

        # combine characters and word
        x = Concatenate(axis=-1)([word_embeddings, char_embeddings])

        x = Bidirectional(LSTM(units=config.lstm_size, return_sequences=True))(x)
        x = Dropout(config.dropout)(x)
        pred = Dense(ntags)(x)

        self.sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        self.model = Model(inputs=[word_ids, char_ids, self.sequence_lengths], outputs=[pred])
        self.transition_params = K.softmax(K.random_uniform_variable(low=0, high=1, shape=(ntags, ntags)))
        self.config = config

    def predict(self, X, sequence_lengths):
        logits = self.model.predict_on_batch(X)
        if self.config.crf:
            viterbi_sequences = []
            transition_params = K.eval(self.transition_params)
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                #viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, K.eval(self.transition_params))
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                viterbi_sequences += [viterbi_sequence]
            return viterbi_sequences
        else:
            raise NotImplementedError('not implemented')

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = K.cast(y_t, tf.int32)
        sequence_lengths = K.reshape(self.sequence_lengths, (-1,))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_t, sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss
