import argparse

from anago.config import Config
from anago.data import reader

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

    if args.save_path:
        print('Saving model to {}.'.format(args.save_path))
        # save the model


if __name__ == '__main__':
    main()