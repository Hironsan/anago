"""
Tagging example.
"""
import argparse
import os
from pprint import pprint

from anago.tagger import Tagger
from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer


def main(args):
    print('Loading objects...')
    model = BiLSTMCRF.load(args.weights_file, args.params_file)
    it = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=it)

    print('Tagging a sentence...')
    res = tagger.analyze(args.sent)
    pprint(res)


if __name__ == '__main__':
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    parser = argparse.ArgumentParser(description='Tagging a sentence.')
    parser.add_argument('--sent', default='President Obama is speaking at the White House.')
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))
    args = parser.parse_args()
    main(args)
