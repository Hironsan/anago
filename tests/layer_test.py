import unittest

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

from anago.layers import ChainCRF


class LayerTest(unittest.TestCase):

    def test_chain_crf(self):
        vocab_size = 20
        n_classes = 11
        model = Sequential()
        model.add(Embedding(vocab_size, n_classes))
        layer = ChainCRF()
        model.add(layer)
        model.compile(loss=layer.loss, optimizer='sgd')

        # Train first mini batch
        batch_size, maxlen = 2, 2
        x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
        y = np.random.randint(n_classes, size=(batch_size, maxlen))
        y = np.eye(n_classes)[y]
        model.train_on_batch(x, y)

        print(x)
        print(y)
