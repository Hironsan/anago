"""Training-related module.
"""
from anago.utils import batch_iter
from anago.callbacks import F1score


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

        # Prepare training and validation data(steps, generator)
        train_steps, train_generator = batch_iter(x_train, y_train,
                                                  batch_size,
                                                  shuffle=shuffle,
                                                  preprocessor=self._preprocessor)

        if x_valid and y_valid:
            valid_steps, valid_generator = batch_iter(x_valid, y_valid,
                                                      batch_size,
                                                      shuffle=False,
                                                      preprocessor=self._preprocessor)
            f1 = F1score(valid_steps, valid_generator,
                         preprocessor=self._preprocessor)
            callbacks = [f1] + callbacks if callbacks else [f1]

        # Train the model
        self._model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose)
