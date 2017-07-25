"""
A Keras implementation of bidirectional LSTM for named entity recognition.
"""
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Reshape, Conv2D, MaxPool2D
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

from anago.models.base_model import BaseModel


def CNN(seq_length, length, input_size, feature_maps, kernels, x):
    concat_input = []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        #conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        #maxp = MaxPool2D((1, reduced_l), data_format="channels_last")(conv)
        conv = Conv2D(feature_map, (1, kernel), activation='tanh')(x)
        maxp = MaxPool2D((1, reduced_l))(conv)
        concat_input.append(maxp)

    #x = Concatenate()(concat_input)
    x = concat_input[0]
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x

class BiLSTMCNN(BaseModel):

    def _build_model(self):
        # build character based word embedding
        feature_maps = [30]
        kernels = [3]
        char_input = Input(shape=(self.config.num_steps, self.config.max_word_len), dtype='int32')
        x1 = TimeDistributed(Embedding(input_dim=self.config.char_vocab_size,
                                       output_dim=self.config.char_embedding_size)
                             )(char_input)
        x1 = CNN(self.config.num_steps, self.config.max_word_len, self.config.char_embedding_size,
                  feature_maps, kernels, x1)

        # build word embedding
        word_input = Input(shape=(self.config.num_steps,), dtype='int32')
        x2 = Embedding(input_dim=self.embeddings.shape[0],
                       output_dim=self.embeddings.shape[1],
                       input_length=self.config.num_steps,
                       weights=[self.embeddings],
                       )(word_input)

        # combine characters and word
        x = Concatenate(axis=-1)([x1, x2])
        x = Dropout(self.config.dropout)(x)

        x = Bidirectional(LSTM(units=self.config.hidden_size,
                               return_sequences=True,
                               dropout=self.config.dropout,
                               recurrent_dropout=self.config.dropout))(x)
        x = Dropout(self.config.dropout)(x)
        preds = TimeDistributed(Dense(units=self.ntags, activation='softmax'))(x)
        model = Model(inputs=[word_input, char_input], outputs=preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=self.config.learning_rate,
                                        #decay=self.config.lr_decay,
                                        #clipvalue=self.config.max_grad_norm
                                        ),
                      metrics=['acc'])
        model.summary()
        self.model = model
