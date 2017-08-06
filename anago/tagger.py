from anago.data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word
from anago.models.tf_model import NERModel
from anago.config_tf import Config
from anago.config_jp import Config as ConfigJP


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
        # output = self._model.predict(words)
        # transform model output

        return [('', '')]

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
        # output = self._model.predict(words)
        # transform model output

        return {'': ['']}


def get_tagger():
    # create instance of config
    config = Config()

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=config.chars)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags), nchars=len(vocab_chars))
    model.build()

    return model, vocab_tags, processing_word


def get_jp_tagger():
    # create instance of config
    config = ConfigJP()

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=config.chars)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags), nchars=len(vocab_chars))
    model.build()

    return model, vocab_tags, processing_word
