"""
A Keras implementation of bidirectional LSTM for named entity recognition.
"""
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

from anago.models.base_model import BaseModel


class BiLSTM(BaseModel):

    def _build_model(self):
        word_input = Input(shape=(self.config.num_steps,), dtype='int32')
        x = Embedding(input_dim=self.embeddings.shape[0],
                      output_dim=self.embeddings.shape[1],
                      input_length=self.config.num_steps,
                      weights=[self.embeddings],
                      )(word_input)
        x = Bidirectional(LSTM(units=self.config.hidden_size,
                               return_sequences=True,
                               dropout=self.config.dropout,
                               recurrent_dropout=self.config.dropout))(x)
        x = Dropout(self.config.dropout)(x)
        preds = TimeDistributed(Dense(units=self.ntags, activation='softmax'))(x)
        model = Model(inputs=word_input, outputs=preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(self.config.learning_rate),
                      metrics=['acc'])
        model.summary()
        self.model = model
