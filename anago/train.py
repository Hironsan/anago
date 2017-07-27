import argparse

from anago.config import Config
from anago.data import reader, metrics, preprocess
from anago.models.bilstm import BiLSTM
from anago.models.bilstm_cnn import BiLSTMCNN

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Where the training/test data is stored.')
parser.add_argument('--save_path', help='Where the trained model is stored.')
parser.add_argument('--log_dir', help='Where log data is stored.')
parser.add_argument('--glove_path', default=None, help='Where GloVe embedding is stored.')
args = parser.parse_args()


def main():
    if not args.data_path:
        raise ValueError('Must set --data_path to conll data directory')

    config = Config()
    config.log_dir = args.log_dir

    vocab_words, vocab_chars, vocab_tags = reader.load_vocab(args.data_path, args.glove_path,
                                                             preprocess.get_processing_word(lowercase=True))

    embeddings = reader.get_glove_vectors(vocab_words, args.glove_path, dim=300)

    # get processing functions
    processing_word = preprocess.get_processing_word(vocab_words, vocab_chars,
                                                     lowercase=True, use_char=config.use_char)
    processing_tag = preprocess.get_processing_word(vocab_tags, lowercase=False)

    raw_data = reader.conll_raw_data(args.data_path, processing_word, processing_tag)
    train_data, valid_data, test_data = raw_data

    if config.use_char:
        sents_char = [list(zip(*d))[0] for d in train_data['X']]
        sents_word = [list(zip(*d))[1] for d in train_data['X']]
        sents_char = preprocess.pad_chars(sents_char, config)
        sents_word = preprocess.pad_words(sents_word, config)
        r = len(sents_char) // config.batch_size * config.batch_size
        sents_char = sents_char[:r]
        sents_word = sents_word[:r]
        X_train = [sents_word, sents_char]
        sents_char = [list(zip(*d))[0] for d in test_data['X']]
        sents_word = [list(zip(*d))[1] for d in test_data['X']]
        sents_word = preprocess.pad_words(sents_word, config)
        sents_char = preprocess.pad_chars(sents_char, config)
        X_test = [sents_word, sents_char]
        config.char_vocab_size = len(vocab_chars)
        config.char_embedding_size = 25
        from models.bilstm_cnn_crf import BiLSTMCNNCrf
        model = BiLSTMCNN(config, embeddings, ntags=len(vocab_tags))
        #model = BiLSTMCNNCrf(config, embeddings, ntags=len(vocab_tags))
        y_train = preprocess.to_onehot(train_data['y'], config, ntags=len(vocab_tags))
        y_train = y_train[:r]
    else:
        X_train = preprocess.pad_words(train_data['X'], config)
        X_test = preprocess.pad_words(test_data['X'], config)
        model = BiLSTM(config, embeddings, ntags=len(vocab_tags))
        y_train = preprocess.to_onehot(train_data['y'], config, ntags=len(vocab_tags))

    y_test = preprocess.to_onehot(test_data['y'], config, ntags=len(vocab_tags))

    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.report(y_test, y_pred, vocab_tags)
    print(metrics.run_evaluate(y_test, y_pred, vocab_tags))

    if args.save_path:
        print('Saving model to {}.'.format(args.save_path))
        model.save(args.save_path)


if __name__ == '__main__':
    main()
