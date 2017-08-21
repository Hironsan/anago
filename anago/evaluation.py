from anago.data.reader import batch_iter, load_word_embeddings
from anago.models.models import SeqLabeling


class Evaluator(object):

    def __init__(self, config):
        self.config = config

    def eval(self, x_test, y_test):
        p = None  # load preprocessor
        train_steps, train_batches = batch_iter(
            list(zip(x_test, y_test)), self.config.batch_size, preprocessor=p)

        embeddings = load_word_embeddings(p.vocab_word, self.config.glove_path, self.config.word_dim)
        self.config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(self.config, embeddings, len(p.vocab_tag))
        model.load()  # load weights
        model.evaluate_generator(train_batches, train_steps)
