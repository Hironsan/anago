
import numpy as np
from keras.callbacks import Callback
from anago.reader import batch_iter
from anago.metrics import F1score


class Evaluator(object):
    def __init__(self,
                 model,
                 preprocessor=None):

        self.model = model
        self.preprocessor = preprocessor

    def eval(self, x_test, y_test):

        # Prepare test data(steps, generator)
        train_steps, train_batches = batch_iter(x_test,
                                                y_test,
                                                batch_size=x_test.shape[0],  # Todo: if batch_size=1, eval does not work.
                                                shuffle=False,
                                                preprocessor=self.preprocessor)

        # Build the evaluator and evaluate the model
        # f1score = F1score(train_steps, train_batches, self.preprocessor)
        # f1score.model = self.model
        # f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.

        # return raw probabilities
        gcp = GetClassProbabilities(train_steps, train_batches, self.preprocessor)
        gcp.model = self.model
        pred, actual = gcp.on_epoch_end(epoch=-1)
        return pred, actual


class GetClassProbabilities(Callback):
    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(GetClassProbabilities, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor

    def on_epoch_end(self, epoch):
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            
            y_true = label
            y_true = np.argmax(y_true, -1)
            seq_len = data[-1]
            seq_len = np.reshape(seq_len, (-1,))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, seq_len)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, seq_len)]
        return y_pred, y_true


'''
class F1score(Callback):

    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            sequence_lengths = data[-1] # shape of (batch_size, 1)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))
            #y_pred = np.asarray(self.model_.predict(data, sequence_lengths))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]

            a, b, c = self.count_correct_and_pred(y_true, y_pred, sequence_lengths)
            correct_preds += a
            total_preds += b
            total_correct += c

        f1 = self._calc_f1(correct_preds, total_correct, total_preds)
        print(' - f1: {:04.2f}'.format(f1 * 100))
        logs['f1'] = f1

    def _calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return f1

    def count_correct_and_pred(self, y_true, y_pred, sequence_lengths):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]

            lab_chunks = set(get_entities(lab))
            lab_pred_chunks = set(get_entities(lab_pred))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
        return correct_preds, total_correct, total_preds
'''
