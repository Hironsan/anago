"""Training-related module.
"""
from anago.callbacks import F1score
from anago.utils import NERSequence


class Trainer(object):
    """A trainer that train the model.

    Attributes:
        _model: Model.
        _preprocessor: Transformer. Preprocessing data for feature extraction.
    """

    def __init__(self, model, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

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

        train_seq = NERSequence(x_train, y_train, batch_size, self._preprocessor.transform)

        if x_valid and y_valid:
            valid_seq = NERSequence(x_valid, y_valid, batch_size, self._preprocessor.transform)
            f1 = F1score(valid_seq, preprocessor=self._preprocessor)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose,
                                  shuffle=shuffle)
