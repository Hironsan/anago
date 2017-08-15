import argparse

from anago.data import reader
from anago.data.preprocess import load_word_embeddings
from anago.models.bilstm_crf import LstmCrfModel


parser = argparse.ArgumentParser()
# data settings
parser.add_argument('--data_path', required=True, help='Where the training/test data is stored.')
parser.add_argument('--save_path', required=True, help='Where the trained model is stored.')
parser.add_argument('--log_dir', help='Where log data is stored.')
parser.add_argument('--glove_path', default=None, help='Where GloVe embedding is stored.')
# model settings
parser.add_argument('--dropout', default=0.5, type=float, help='The probability of keeping weights in the dropout layer')
parser.add_argument('--char_dim', default=100, type=int, help='Character embedding dimension')
parser.add_argument('--word_dim', default=300, type=int, help='Word embedding dimension')
parser.add_argument('--lstm_size', default=300, type=int, help='The number of hidden units in lstm')
parser.add_argument('--char_lstm_size', default=100, type=int, help='The number of hidden units in char lstm')
parser.add_argument('--use_char', default=True, help='Use character feature', action='store_false')
parser.add_argument('--crf', default=True, help='Use CRF', action='store_false')
parser.add_argument('--train_embeddings', default=True, help='Fine-tune word embeddings', action='store_false')
# learning settings
parser.add_argument('--batch_size', default=20, type=int, help='The batch size')
parser.add_argument('--clip_value', default=0.0, type=float, help='The clip value')
parser.add_argument('--learning_rate', default=0.001, type=float, help='The initial value of the learning rate')
parser.add_argument('--lr_decay', default=0.9, type=float, help='The decay of the learning rate for each epoch')
parser.add_argument('--lr_method', default='adam', help='The learning method')
parser.add_argument('--max_epoch', default=15, type=int, help='The number of epochs')
parser.add_argument('--reload', default=False, help='Reload model', action='store_true')
parser.add_argument('--nepoch_no_imprv', default=3, type=int, help='For early stopping')
config = parser.parse_args()


def main():
    datasets = reader.read_datasets(config.data_path, config.glove_path)
    vocab_word = datasets.train._preprocessor.vocab_word
    vocab_char = datasets.train._preprocessor.vocab_char
    vocab_tag  = datasets.train._preprocessor.vocab_tag
    embeddings = load_word_embeddings(vocab_word, config.glove_path, config.word_dim)
    print('{} words, {} chars'.format(len(vocab_word), len(vocab_char)))

    model = LstmCrfModel(config, embeddings, vocab_char, vocab_tag)
    model.train(datasets.train, datasets.valid)
    model.evaluate(datasets.test)

if __name__ == '__main__':
    main()
