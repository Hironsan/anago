"""
Model API.
"""
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


class Tagger(object):
    """A model API that tags input sentence.

    Attributes:
        model: Model.
        preprocessor: Transformer. Preprocessing data for feature extraction.
        tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.
    """

    def __init__(self, model, preprocessor, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def predict_proba(self, text):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Args:
            text : string, the input text.

        Returns:
            y : array-like, shape = [num_words, num_classes]
            Returns the probability of the word for each class in the model,
        """
        assert isinstance(text, str)

        words = self.tokenizer(text)
        X = self.preprocessor.transform([words])
        y = self.model.predict(X)
        y = y[0]  # reduce batch dimension.

        return y

    def _get_prob(self, pred):
        prob = np.max(pred, -1)

        return prob

    def _get_tags(self, pred):
        tags = self.preprocessor.inverse_transform([pred])
        tags = tags[0]  # reduce batch dimension

        return tags

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

    def analyze(self, text):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.

        Returns:
            res: dict.

        Examples:
            >>> text = 'President Obama is speaking at the White House.'
            >>> model.analyze(text)
            {
                "words": [
                    "President",
                    "Obama",
                    "is",
                    "speaking",
                    "at",
                    "the",
                    "White",
                    "House."
                ],
                "entities": [
                    {
                        "beginOffset": 1,
                        "endOffset": 2,
                        "score": 1,
                        "text": "Obama",
                        "type": "PER"
                    },
                    {
                        "beginOffset": 6,
                        "endOffset": 8,
                        "score": 1,
                        "text": "White House.",
                        "type": "ORG"
                    }
                ]
            }
        """
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(text, tags, prob)

        return res

    def predict(self, text):
        """Predict using the model.

        Args:
            text: string, the input text.

        Returns:
            tags: list, shape = (num_words,)
            Returns predicted values.
        """
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)

        return tags
