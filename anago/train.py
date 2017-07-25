import argparse

from anago.config import Config
from anago.data import reader, metrics, preprocess
from anago.models.bilstm import BiLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Where the training/test data is stored.')
parser.add_argument('--save_path', help='Where the trained model is stored.')
parser.add_argument('--glove_path', default=None, help='Where GloVe embedding is stored.')
args = parser.parse_args()


def main():
    if not args.data_path:
        raise ValueError('Must set --data_path to conll data directory')

    config = Config()

    vocab_words, vocab_tags = reader.load_vocab(args.data_path, args.glove_path,
                                                preprocess.get_processing_word(lowercase=True))

    embeddings = reader.get_glove_vectors(vocab_words, args.glove_path, dim=300)

    # get processing functions
    processing_word = preprocess.get_processing_word(vocab_words, lowercase=True)
    processing_tag = preprocess.get_processing_word(vocab_tags, lowercase=False)

    raw_data = reader.conll_raw_data(args.data_path, processing_word, processing_tag)
    train_data, valid_data, test_data = raw_data

    X_train = preprocess.pad_words(train_data['X'], config)
    X_test = preprocess.pad_words(test_data['X'], config)
    y_train = preprocess.to_onehot(train_data['y'], config, ntags=len(vocab_tags))
    y_test = preprocess.to_onehot(test_data['y'], config, ntags=len(vocab_tags))

    model = BiLSTM(config, embeddings, ntags=len(vocab_tags))
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.report(y_test, y_pred, vocab_tags)

    if args.save_path:
        print('Saving model to {}.'.format(args.save_path))
        model.save(args.save_path)


if __name__ == '__main__':
    main()
