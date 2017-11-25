import os

import anago
from anago.reader import load_data_and_labels, load_glove

DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
EMBEDDING_PATH = 'glove.6B.100d.txt'

train_path = os.path.join(DATA_ROOT, 'train.txt')
valid_path = os.path.join(DATA_ROOT, 'valid.txt')

print('Loading data...')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
print(len(x_train), 'train sequences')
print(len(x_valid), 'valid sequences')

embeddings = load_glove(EMBEDDING_PATH)

# Use pre-trained word embeddings
model = anago.Sequence(max_epoch=1, embeddings=embeddings)
model.train(x_train, y_train, x_valid, y_valid)
