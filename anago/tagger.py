"""
Model API.
"""
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


class Tagger(object):

    def __init__(self, model, preprocessor, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def predict(self, sent):
        """Predict using the model.

        Args:
            sent : string, the input data.

       Returns:
           y : array-like, shape (n_samples,) or (n_samples, n_classes)
           The predicted classes.
       """
        words = self.tokenizer(sent)
        X = self.preprocessor.transform([words])
        y = self.model.predict(X)

        return y

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred)

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response(self, sent, tags, prob):
        words = self.tokenizer(sent)
        res = {
            'words': words,
            'entities': [

            ]
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, sent):
        assert isinstance(sent, str)

        pred = self.predict(sent)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(sent, tags, prob)

        return res

    def label(self, sent):
        pred = self.predict(sent)
        tags = self._get_tags(pred)

        return tags[0]
