from collections import defaultdict

import numpy as np

from anago.metrics import get_entities


class Tagger(object):

    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, words):
        length = np.array([len(words)])
        X = self.preprocessor.transform([words])
        pred = self.model.predict(X, length)

        return pred

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response(self, words, tags, prob):
        res = {
            'words': words,
            'entities': [

            ]
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, words):
        assert isinstance(words, list)

        pred = self.predict(words)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(words, tags, prob)

        return res

    def tag(self, words):
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
        assert isinstance(words, list)

        pred = self.predict(words)
        pred = [t.split('-')[-1] for t in pred]  # remove prefix: e.g. B-Person -> Person

        return list(zip(words, pred))

    def get_entities(self, words):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        assert isinstance(words, list)

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
