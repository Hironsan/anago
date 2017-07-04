import os
import unittest

from anago.data.loader import *


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.preprocess = lambda x: x.lower()

    def test_load_file(self):
        words, tags = load_file(TRAIN_DATA)
        self.assertEqual(len(words), len(tags))

    def test_word_mapping(self):
        words, tags = load_file(TRAIN_DATA)
        indices_word, word_indices = word_mapping(words, self.preprocess)
        self.assertEqual(len(indices_word), len(word_indices))

    def test_char_mapping(self):
        words, tags = load_file(TRAIN_DATA)
        indices_char, char_indices = char_mapping(words, self.preprocess)
        self.assertEqual(len(indices_char), len(char_indices))

    def test_tag_mapping(self):
        words, tags = load_file(TRAIN_DATA)
        indices_tag, tag_indices = tag_mapping(tags)
        self.assertEqual(len(indices_tag), len(tag_indices))

    def test_convert_word_str(self):
        words, tags = load_file(TRAIN_DATA)
        indices_word, word_indices = word_mapping(words, self.preprocess)
        words = convert_words_str(words[0], word_indices, lower=True)
        print(words)

    def test_convert_char_str(self):
        words, tags = load_file(TRAIN_DATA)
        indices_char, char_indices = char_mapping(words, self.preprocess)
        chars = convert_char_str(words[0], char_indices, lower=True)
        print(chars)

    def test_convert_tag_str(self):
        words, tags = load_file(TRAIN_DATA)
        indices_tag, tag_indices = tag_mapping(tags)
        print(tags[0])
        tags = convert_tag_str(tags, tag_indices)
        print(tags[0])

    def test_pad_word_chars(self):
        words, tags = load_file(TRAIN_DATA)
        indices_char, char_indices = char_mapping(words, self.preprocess)
        chars = convert_char_str(words[0], char_indices, lower=True)
        print(chars)
        a, b, c = pad_word_chars(chars)
        print(a)

    def test_prepare_sentence(self):
        words, tags = load_file(TRAIN_DATA)
        indices_word, word_indices = word_mapping(words, self.preprocess)
        indices_char, char_indices = char_mapping(words, self.preprocess)
        words, chars = prepare_sentence(words, word_indices, char_indices, lower=True)
        print(words[0])