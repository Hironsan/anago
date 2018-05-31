"""
Wrapper class.
"""
from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from anago.trainer import Trainer

from seqeval.metrics import f1_score


class Sequence(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True,
                 initial_vocab=None,
                 optimizer='adam'):

        self.model = None
        self.p = None

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.embeddings = embeddings
        self.use_char = use_char
        self.use_crf = use_crf
        self.initial_vocab = initial_vocab
        self.optimizer = optimizer

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            epochs=1, batch_size=32, verbose=1, callbacks=None):
        """Fit the model according to the given training data.

        Args:
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

            y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns:
            self : object.
        """
        # Build preprocessors.
        p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        p.fit(x_train, y_train)

        # Build a model.
        model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                          word_vocab_size=p.word_vocab_size,
                          num_labels=p.label_size,
                          word_embedding_dim=self.word_embedding_dim,
                          char_embedding_dim=self.char_embedding_dim,
                          word_lstm_size=self.word_lstm_size,
                          char_lstm_size=self.char_lstm_size,
                          fc_dim=self.fc_dim,
                          dropout=self.dropout,
                          embeddings=self.embeddings,
                          use_char=self.use_char,
                          use_crf=self.use_crf)
        model.build()
        model.compile(loss=model.get_loss(), optimizer=self.optimizer)

        # Train the model.
        trainer = Trainer(model, preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks)

        self.p = p
        self.model = model

        return self

    def score(self, x_test, y_test):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Args:
            X : array-like, shape = (n_samples, n_features)
            Test samples.

            y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns:
            score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            y_pred = self.model.predict(x_test)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def analyze(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze(words)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def save(self, weights_file, params_file, preprocessor_file):
        self.p.save(preprocessor_file)
        self.model.save(weights_file, params_file)

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()

        # Load preprocessor
        self.p = IndexTransformer.load(preprocessor_file)

        # Load the model.
        self.model = BiLSTMCRF.load(weights_file, params_file)

        return self
