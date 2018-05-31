import os
import shutil
import unittest

from anago.models import BiLSTMCRF


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.save_root = os.path.join(os.path.dirname(__file__), 'models')
        cls.weights_file = os.path.join(cls.save_root, 'weights.h5')
        cls.params_file = os.path.join(cls.save_root, 'params.json')
        if not os.path.exists(cls.save_root):
            os.mkdir(cls.save_root)
        if os.path.exists(cls.weights_file):
            os.remove(cls.weights_file)
        if os.path.exists(cls.weights_file):
            os.remove(cls.params_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_root)

    def test_build_model(self):
        char_vocab_size = 100
        word_vocab_size = 10000
        num_labels = 10

        # Normal.
        model = BiLSTMCRF(char_vocab_size=char_vocab_size,
                          word_vocab_size=word_vocab_size,
                          num_labels=num_labels)
        model.build()

        # No CRF.
        model = BiLSTMCRF(char_vocab_size=char_vocab_size,
                          word_vocab_size=word_vocab_size,
                          num_labels=num_labels,
                          use_crf=False)
        model.build()

        # No character feature.
        model = BiLSTMCRF(char_vocab_size=char_vocab_size,
                          word_vocab_size=word_vocab_size,
                          num_labels=num_labels,
                          use_char=False)
        model.build()

    def test_save_and_load(self):
        char_vocab_size = 100
        word_vocab_size = 10000
        num_labels = 10

        model = BiLSTMCRF(char_vocab_size=char_vocab_size,
                          word_vocab_size=word_vocab_size,
                          num_labels=num_labels)
        model.build()

        self.assertFalse(os.path.exists(self.weights_file))
        self.assertFalse(os.path.exists(self.params_file))

        model.save(self.weights_file, self.params_file)

        self.assertTrue(os.path.exists(self.weights_file))
        self.assertTrue(os.path.exists(self.params_file))

        model = BiLSTMCRF.load(self.weights_file, self.params_file)
        self.assertEqual(model._char_vocab_size, char_vocab_size)
        self.assertEqual(model._word_vocab_size, word_vocab_size)
        self.assertEqual(model._num_labels, num_labels)
