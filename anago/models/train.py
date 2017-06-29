import numpy as np
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix

from anago.data import conll2003


maxlen = 50  # cut texts after this number of words (among top max_features most common words)
word_embedding_dim = 100
lstm_dim = 100
batch_size = 64

print('Loading data...')
X_words_train, y_train, X_words_test, y_test, index2word, index2chunk = conll2003.load_data(word_preprocess=lambda w: w.lower())

max_features = len(index2word)
nb_chunk_tags = len(index2chunk)

X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')
X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')
y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
y_train = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_train])
y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post')
y_test = np.array([to_categorical(y, num_classes=nb_chunk_tags) for y in y_test])

print('Unique words       :', max_features)
print('Unique chunk tags  :', nb_chunk_tags)
print('X_words_train shape:', X_words_train.shape)
print('X_words_test shape :', X_words_test.shape)
print('y_train shape      :', y_train.shape)
print('y_test shape       :', y_test.shape)

print('Build model...')
word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
word_emb = Embedding(max_features, word_embedding_dim, input_length=maxlen, name='word_emb')(word_input)
bilstm = Bidirectional(LSTM(lstm_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(word_emb)
bilstm_d = Dropout(0.2)(bilstm)
dense = TimeDistributed(Dense(nb_chunk_tags, activation='softmax'))(bilstm_d)

model = Model(inputs=[word_input], outputs=[dense])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(0.01),
              metrics=['acc'])

model.summary()

print('Train...')
model.fit([X_words_train], y_train, batch_size=batch_size, epochs=3)
score = model.evaluate(X_words_test, y_test, batch_size=batch_size)
print(score)
