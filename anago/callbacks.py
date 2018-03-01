import os

import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from seqeval.metrics import f1_score


def get_callbacks(log_dir=None, valid=(), tensorboard=True, eary_stopping=True):
    """Get callbacks.

    Args:
        log_dir (str): the destination to save logs(for TensorBoard).
        valid (tuple): data for validation.
        tensorboard (bool): Whether to use tensorboard.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if log_dir and tensorboard:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)
        callbacks.append(TensorBoard(log_dir))

    if valid:
        callbacks.append(F1score(*valid))

    if log_dir:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.2f}']) + '.h5'
        save_callback = ModelCheckpoint(os.path.join(log_dir, file_name),
                                        monitor='f1',
                                        save_weights_only=True)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=3, mode='max'))

    return callbacks


class F1score(Callback):

    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            sequence_lengths = data[-1]  # shape of (batch_size, 1)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]


        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        logs['f1'] = score
