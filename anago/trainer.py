"""
Model Trainer.
"""
import numpy as np

from anago.utils import batch_iter
from anago.callbacks import F1score


class Trainer(object):

    def __init__(self, model, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None):

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(x_train, y_train,
                                                batch_size,
                                                preprocessor=self._preprocessor)

        if isinstance(x_valid, np.ndarray) and isinstance(y_valid, np.ndarray):
            valid_steps, valid_batches = batch_iter(x_valid, y_valid,
                                                    batch_size,
                                                    preprocessor=self._preprocessor)
            f1 = F1score(valid_steps, valid_batches,
                         preprocessor=self._preprocessor)
            callbacks = callbacks + [f1] if callbacks else [f1]

        # Train the model
        self._model.fit_generator(generator=train_batches,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose)
