"""
Wrapper class.
"""
from seqeval.metrics import f1_score

from anago.models import BiLSTMCRF, save_model, load_model
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from anago.trainer import Trainer
from anago.utils import filter_embeddings


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
        self.tagger = None

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
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Fit the model for a fixed number of epochs.

        Args:
            x_train: list of training data.
            y_train: list of training target (label) data.
            x_valid: list of validation data.
            y_valid: list of validation target (label) data.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch). `shuffle` will default to True.
        """
        p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        p.fit(x_train, y_train)
        embeddings = filter_embeddings(self.embeddings, p._word_vocab.vocab, self.word_embedding_dim)

        model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                          word_vocab_size=p.word_vocab_size,
                          num_labels=p.label_size,
                          word_embedding_dim=self.word_embedding_dim,
                          char_embedding_dim=self.char_embedding_dim,
                          word_lstm_size=self.word_lstm_size,
                          char_lstm_size=self.char_lstm_size,
                          fc_dim=self.fc_dim,
                          dropout=self.dropout,
                          embeddings=embeddings,
                          use_char=self.use_char,
                          use_crf=self.use_crf)
        model, loss = model.build()
        model.compile(loss=loss, optimizer=self.optimizer)

        trainer = Trainer(model, preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks,
                      shuffle=shuffle)

        self.p = p
        self.model = model

    def predict(self, x_test):
        """Returns the prediction of the model on the given test data.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

        Returns:
            y_pred : array-like, shape = (n_smaples, sent_length)
            Prediction labels for x.
        """
        if self.model:
            lengths = map(len, x_test)
            x_test = self.p.transform(x_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            return y_pred 
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def score(self, x_test, y_test):
        """Returns the f1-micro score on the given test data and labels.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.

        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def analyze(self, text, tokenizer=str.split):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.

        Returns:
            res: dict.
        """
        if not self.tagger:
            self.tagger = Tagger(self.model,
                                 preprocessor=self.p,
                                 tokenizer=tokenizer)

        return self.tagger.analyze(text)

    def save(self, weights_file, params_file, preprocessor_file):
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()
        self.p = IndexTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, params_file)

        return self
