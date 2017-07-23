import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


def pad_words(xs, config):
    return sequence.pad_sequences(xs, maxlen=config.num_steps, padding='post')


def to_onehot(ys, config):
    ys = sequence.pad_sequences(ys, maxlen=config.num_steps, padding='post')
    return np.asarray([to_categorical(y, num_classes=config.num_classes) for y in ys])
