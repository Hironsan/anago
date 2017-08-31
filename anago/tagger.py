import os
from collections import defaultdict

import numpy as np

from anago.models import SeqLabeling
from anago.data.metrics import get_entities


class Tagger(object):

    def __init__(self,
                 config,
                 weights,
                 save_path='',
                 preprocessor=None,
                 tokenizer=str.split):

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        # Build the model
        self.model = SeqLabeling(config, ntags=len(self.preprocessor.vocab_tag))
        self.model.load(filepath=os.path.join(save_path, weights))

    def predict(self, words):
        sequence_lengths = [len(words)]
        X = self.preprocessor.transform([words])
        pred = self.model.predict(X, sequence_lengths)
        pred = np.argmax(pred, -1)
        pred = self.preprocessor.inverse_transform(pred[0])

        return pred

    def tag(self, sent):
        """Tags a sentence named entities.

        Args:
            sent: a sentence

        Return:
            labels_pred: list of (word, tag) for a sentence

        Example:
            >>> sent = 'President Obama is speaking at the White House.'
            >>> print(self.tag(sent))
            [('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
             ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
             ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        pred = [t.split('-')[-1] for t in pred]  # remove prefix: e.g. B-Person -> Person

        return list(zip(words, pred))

    def get_entities(self, sent):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        entities = self._get_chunks(words, pred)

        return entities

    def _get_chunks(self, words, tags):
        """
        Args:
            words: sequence of word
            tags: sequence of labels

        Returns:
            dict of entities for a sequence

        Example:
            words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
            tags = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        chunks = get_entities(tags)
        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(words[chunk_start: chunk_end]))  # todo delimiter changeable

        return res
