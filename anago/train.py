import argparse

from anago.config import Config
from anago.data import reader, metrics, preprocess
from anago.models.bilstm import BiLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Where the training/test data is stored.')
parser.add_argument('--save_path', help='Model output directory.')
args = parser.parse_args()


def main():
    if not args.data_path:
        raise ValueError('Must set --data_path to conll data directory')

    import os
    vocab_path = os.path.join(os.path.dirname(__file__), 'models/map.json')
    raw_data = reader.conll_raw_data(args.data_path, vocab_path=vocab_path)
    train_data, valid_data, test_data, word_to_id, entity_to_id = raw_data

    config = Config(word_to_id, entity_to_id)
    config.embedding_path = os.path.join(os.path.dirname(__file__), 'models/embeddings.npz')
    config.word_to_id = word_to_id

    X_train = preprocess.pad_words(train_data['X'], config)
    X_test = preprocess.pad_words(test_data['X'], config)
    y_train = preprocess.to_onehot(train_data['y'], config)
    y_test = preprocess.to_onehot(test_data['y'], config)

    model = BiLSTM(config)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.report(y_test, y_pred, entity_to_id)

    if args.save_path:
        print('Saving model to {}.'.format(args.save_path))
        model.save(args.save_path)


if __name__ == '__main__':
    main()
