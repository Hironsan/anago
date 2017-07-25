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


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, use_char=False):
    """
    Args:
        vocab_words: dict[word] = idx
        vocab_chars: dict[char] = idx
        lowercase: if True, word is converted to lowercase
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. preprocess word
        if lowercase:
            word = word.lower()
        word = digit_to_zero(word)

        # 1. get chars of words
        if vocab_chars is not None and use_char:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars.get(char, vocab_chars[UNK])]

        # 2. get id of word
        if vocab_words is not None:
            word = vocab_words.get(word, vocab_words.get(UNK, vocab_words[PAD]))

        # 3. return tuple char ids, word id
        if vocab_chars is not None and use_char:
            return char_ids, word
        else:
            return word

    return f


def digit_to_zero(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
