"""
Model API.
"""

import numpy as np

from seqeval.metrics.sequence_labeling import get_entities


class Tagger(object):

    def __init__(self, model, preprocessor=None, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def predict(self, X):
        """Predict using the model.

        Args:
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
           The input data.

       Returns:
           y : array-like, shape (n_samples,) or (n_samples, n_classes)
           The predicted classes.
       """
        length = np.array([len(X)])
        X = self.preprocessor.transform([X])
        y = self.model.predict(X, length)

        return y

    def predict_proba(self, X):
        """Probability estimates.

        Args:
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns:
            y_prob : array-like, shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        pass

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
