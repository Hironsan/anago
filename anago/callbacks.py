"""
Custom callbacks.
"""
from keras.callbacks import Callback
from seqeval.metrics import f1_score


class F1score(Callback):

    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(self.valid_steps):
            x_true, y_true = next(self.valid_batches)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true)
            y_pred = self.p.inverse_transform(y_pred)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        logs['f1'] = score