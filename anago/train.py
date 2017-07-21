import argparse

import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from anago.config import Config
from anago.data import reader, metrics
from anago.models.bilstm import BiLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Where the training/test data is stored.')
parser.add_argument('--save_path', help='Model output directory.')
args = parser.parse_args()


def main():
    if not args.data_path:
        raise ValueError('Must set --data_path to conll data directory')

    raw_data = reader.conll_raw_data(args.data_path)
    train_data, valid_data, test_data, word_to_id, entity_to_id = raw_data

    config = Config()

    ### -----------------rewrite---------------------------
    maxlen = 50  # cut texts after this number of words (among top max_features most common words)
    word_embedding_dim = 100
    lstm_dim = 100
    batch_size = 64

    max_features = len(word_to_id)
    nb_chunk_tags = len(entity_to_id)

    X_train = sequence.pad_sequences(train_data['X'], maxlen=maxlen, padding='post')
    X_test = sequence.pad_sequences(test_data['X'], maxlen=maxlen, padding='post')
    y_train = sequence.pad_sequences(train_data['y'], maxlen=maxlen, padding='post')
    y_train = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_train])
    y_test = sequence.pad_sequences(test_data['y'], maxlen=maxlen, padding='post')
    y_test = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_test])

    model = BiLSTM(maxlen, max_features, word_embedding_dim, lstm_dim, nb_chunk_tags, batch_size, epoch_size=10)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    # metrics.report(y_test, y_pred, index2chunk)
    ### -----------------rewrite---------------------------

    if args.save_path:
        print('Saving model to {}.'.format(args.save_path))
        # save the model


if __name__ == '__main__':
    main()