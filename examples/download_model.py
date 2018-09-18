import os

import anago
from anago.utils import download, load_data_and_labels


if __name__ == '__main__':
    dir_path = 'test_dir'
    url = 'https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/conll2003_en.zip'
    DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')

    test_path = os.path.join(DATA_ROOT, 'test.txt')
    x_test, y_test = load_data_and_labels(test_path)

    weights, params, preprocessor = download(url)

    model = anago.Sequence.load(weights, params, preprocessor)
    score = model.score(x_test, y_test)
    print('F1(micro): {}'.format(score))
