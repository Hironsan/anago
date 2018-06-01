import argparse
import os

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

from anago.utils import load_data_and_labels, filter_embeddings
from anago.trainer import Trainer
from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer


def main(args):
    print('Loading datasets...')
    X, y = load_data_and_labels(args.data_path)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    embeddings = KeyedVectors.load(args.embedding_path).wv

    print('Transforming datasets...')
    p = IndexTransformer()
    p.fit(X, y)
    embeddings = filter_embeddings(embeddings, p._word_vocab, embeddings.vector_size)

    print('Building a model...')
    model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      embeddings=embeddings,
                      char_embedding_dim=50)
    model.build()

    print('Training the model...')
    trainer = Trainer(model, preprocessor=p)
    trainer.train(x_train, y_train, x_valid, y_valid)

    print('Saving the model...')
    model.save(args.weights_file, args.params_file)
    p.save(args.preprocessor_file)


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../tests/data')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--data_path', default=os.path.join(DATA_DIR, 'datasets_word.tsv'))
    parser.add_argument('--embedding_path',
                        default=os.path.join(DATA_DIR, 'jawiki-embeddings/wiki.ja.word2vec.model'))
    parser.add_argument('--log_dir', default=os.path.join(os.path.dirname(__file__), 'logs'))
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))
    args = parser.parse_args()
    main(args)
