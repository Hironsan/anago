from collections import defaultdict


class Tagger(object):

    def __init__(self, tokenizer=str.split):
        self._tokenizer = tokenizer
        self._model = None  # needs to load model

    def tag(self, sent):
        """Tags a sentence named entities.

        Args:
            sent: a sentence
        Return:
            labels_pred: list of (word, tag) for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = [('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
                      ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
                      ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
        """
        assert isinstance(sent, str)

        words = self._tokenizer(sent)
        words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
        # pred = self._model.predict(words)
        pred = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
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

        words = self._tokenizer(sent)
        words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
        # pred = self._model.predict(words)
        pred = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
        entities = self.get_chunks(words, pred)

        return entities

    def get_chunks(self, words, tags):
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
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(tags):
            # End of a chunk 1
            if tok == 'O' and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != 'O':
                tok_chunk_class, tok_chunk_type = tok.split('-')
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(tags))
            chunks.append(chunk)

        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(words[chunk_start: chunk_end]))  # todo delimiter changeable

        return res
