import os

from anago.data.metrics import F1score
from anago.data.preprocess import WordPreprocessor
from anago.data.reader import batch_iter
from anago.models import SeqLabeling


class Evaluator(object):

    def __init__(self, config, weights):
        self.config = config
        self.weights = weights

    def eval(self, x_test, y_test):
        p = WordPreprocessor.load(os.path.join(self.config.save_path, 'preprocessor.pkl'))
        train_steps, train_batches = batch_iter(
            list(zip(x_test, y_test)), self.config.batch_size, preprocessor=p)

        self.config.char_vocab_size = len(p.vocab_char)
        self.config.vocab_size = len(p.vocab_word)

        model = SeqLabeling(self.config, ntags=len(p.vocab_tag))
        model.load(filepath=os.path.join(self.config.save_path, self.weights))
        f1score = F1score(train_steps, train_batches, p, model)
        f1score.on_epoch_end(epoch=-1)  # epoch is some value
