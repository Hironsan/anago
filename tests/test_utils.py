import os
import unittest

from anago.utils import load_data_and_labels, batch_iter, Vocabulary

from anago.preprocess import DynamicPreprocessor


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_extract(self):
        sents, labels = load_data_and_labels(self.filename)
        self.assertTrue(len(sents) == len(labels))

    def test_batch_iter(self):
        sents, labels = load_data_and_labels(self.filename)
        batch_size = 32
        p = DynamicPreprocessor()
        steps, batches = batch_iter(list(zip(sents, labels)), batch_size, preprocessor=p)
        self.assertEqual(len([_ for _ in batches]), steps)  # Todo: infinite loop


class TestVocabulary(unittest.TestCase):

    def test_add_documents(self):
        # word vocabulary.
        docs = [['a'], ['a', 'b'], ['a', 'b', 'c']]
        token2id = {'<pad>': 0, 'a': 1, 'b': 2, 'c': 3, '<unk>': 4}
        vocab = Vocabulary()
        vocab.add_documents(docs)
        vocab.build()
        self.assertEqual(vocab._token2id, token2id)

        token2id = {'<pad>': 0, 'a': 1, 'b': 2, 'c': 3}
        vocab = Vocabulary(unk_token=False)
        vocab.add_documents(docs)
        vocab.build()
        self.assertEqual(vocab._token2id, token2id)

        token2id = {'<pad>': 0, '<s>': 1, 'a': 2, 'b': 3, 'c': 4}
        vocab = Vocabulary(unk_token=False, specials=('<pad>', '<s>'))
        vocab.add_documents(docs)
        vocab.build()
        self.assertEqual(vocab._token2id, token2id)

        token2id = {'a': 0, 'b': 1, 'c': 2}
        vocab = Vocabulary(unk_token=False, specials=())
        vocab.add_documents(docs)
        vocab.build()
        self.assertEqual(vocab._token2id, token2id)

        # char vocabulary.
        docs = ['hoge', 'fuga', 'bar']
        vocab = Vocabulary()
        vocab.add_documents(docs)
        vocab.build()
        num_chars = len(set(''.join(docs))) + 2
        self.assertEqual(len(vocab._token2id), num_chars)

    def test_doc2id(self):
        # word ids.
        docs = [['a'], ['a', 'b'], ['a', 'b', 'c']]
        vocab = Vocabulary()
        vocab.add_documents(docs)
        vocab.build()
        another_doc = ['a', 'b', 'c', 'd']
        doc_ids = vocab.doc2id(another_doc)
        self.assertEqual(doc_ids, [1, 2, 3, 4])

        # char_ids.
        docs = ['hoge', 'fuga', 'bar']
        vocab = Vocabulary()
        vocab.add_documents(docs)
        vocab.build()
        doc_ids = vocab.doc2id(docs[0])
        correct = [vocab.token_to_id(c) for c in docs[0]]
        self.assertEqual(doc_ids, correct)

    def test_id2doc(self):
        # word ids.
        docs = [['B-PSN'], ['B-ORG', 'I-ORG'], ['B-LOC', 'I-LOC', 'O']]
        vocab = Vocabulary(unk_token=False, lower=False)
        vocab.add_documents(docs)
        vocab.build()
        true_doc = ['O', 'B-LOC', 'O', 'O']
        doc_ids = vocab.doc2id(true_doc)
        pred_doc = vocab.id2doc(doc_ids)
        self.assertEqual(pred_doc, true_doc)
