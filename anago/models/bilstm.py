"""
A Keras implementation of bidirectional LSTM for named entity recognition.
"""
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

from anago.models.base_model import BaseModel


class BiLSTM(BaseModel):

    def __init__(self, maxlen, max_features, word_embedding_dim, lstm_dim, num_classes, batch_size, epoch_size):
        self.maxlen = maxlen
        self.max_features = max_features
        self.word_embedding_dim = word_embedding_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.model = None

    def _build_model(self):
        word_input = Input(shape=(self.maxlen,), dtype='int32', name='word_input')
        word_emb = Embedding(self.max_features, self.word_embedding_dim, input_length=self.maxlen, name='word_emb')(word_input)
        bilstm = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(word_emb)
        bilstm_d = Dropout(0.5)(bilstm)
        dense = TimeDistributed(Dense(self.num_classes, activation='softmax'))(bilstm_d)
        self.model = Model(inputs=[word_input], outputs=[dense])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(0.01),
                           metrics=['acc'])
        self.model.summary()
