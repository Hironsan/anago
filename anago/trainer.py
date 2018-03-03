"""
Model Trainer.
"""

from anago.utils import batch_iter
from anago.callbacks import get_callbacks


class Trainer(object):

    def __init__(self, model, loss='categorical_crossentropy', optimizer='adam',
                 max_epoch=15, batch_size=32, checkpoint_path=None,
                 log_dir=None, preprocessor=None, early_stopping=False):
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._max_epoch = max_epoch
        self._batch_size = batch_size
        self._checkpoint_path = checkpoint_path
        self._log_dir = log_dir
        self._early_stopping = early_stopping
        self._preprocessor = preprocessor

    def train(self, x_train, y_train, x_valid=None, y_valid=None):

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(x_train, y_train,
                                                self._batch_size,
                                                preprocessor=self._preprocessor)
        valid_steps, valid_batches = batch_iter(x_valid, y_valid,
                                                self._batch_size,
                                                preprocessor=self._preprocessor)

        self._model.compile(loss=self._loss, optimizer=self._optimizer)

        # Prepare callbacks
        callbacks = get_callbacks(log_dir=self._log_dir,
                                  checkpoint_dir=self._checkpoint_path,
                                  eary_stopping=self._early_stopping,
                                  valid=(valid_steps, valid_batches, self._preprocessor))

        # Train the model
        self._model.fit_generator(generator=train_batches,
                                  steps_per_epoch=train_steps,
                                  epochs=self._max_epoch,
                                  callbacks=callbacks)
