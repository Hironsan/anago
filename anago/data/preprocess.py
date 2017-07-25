import re
import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from anago.data.reader import UNK, PAD


def pad_words(xs, config):
    return sequence.pad_sequences(xs, maxlen=config.num_steps, padding='post')


def to_onehot(ys, config, ntags):
    ys = sequence.pad_sequences(ys, maxlen=config.num_steps, padding='post')
    return np.asarray([to_categorical(y, num_classes=ntags) for y in ys])


def get_processing_word(vocab=None, lowercase=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        word = digit_to_zero(word)

        # 2. get id of word
        if vocab is not None:
            word = vocab.get(word, vocab.get(UNK, vocab[PAD]))

        return word

    return f


def digit_to_zero(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
