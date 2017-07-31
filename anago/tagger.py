from anago.data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word, CoNLLDataset
from anago.models.tf_model import NERModel
from anago.config_tf import Config
from anago.config_jp import Config as ConfigJP


def get_tagger():
    # create instance of config
    config = Config()

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=config.chars)
    processing_tag  = get_processing_word(vocab_tags, lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars))
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
    processing_tag  = get_processing_word(vocab_tags, lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars))
    model.build()

    return model, vocab_tags, processing_word
