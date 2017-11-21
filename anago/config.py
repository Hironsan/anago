import json


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self, char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                 word_lstm_units=100, dropout=0.5, char_feature=True, crf=True):

        # Number of unique words in the vocab (plus 2, for <UNK>, <PAD>).
        self.vocab_size = None
        self.char_vocab_size = None

        # LSTM input and output dimensionality, respectively.
        self.char_embedding_size = char_emb_size
        self.num_char_lstm_units = char_lstm_units
        self.word_embedding_size = word_emb_size
        self.num_word_lstm_units = word_lstm_units

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.dropout = dropout

        # If True, use character feature.
        self.char_feature = char_feature

        # If True, use crf.
        self.crf = crf

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(vars(self), f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, file):
        with open(file) as f:
            variables = json.load(f)
            self = cls()
            for key, val in variables.items():
                setattr(self, key, val)
        return self


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self, batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=15, early_stopping=True, patience=3,
                 train_embeddings=True, max_checkpoints_to_keep=5):

        # Batch size
        self.batch_size = batch_size

        # Optimizer for training the model.
        self.optimizer = optimizer

        # Learning rate for the initial phase of training.
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        # If not None, clip gradients to this value.
        self.clip_gradients = clip_gradients

        # The number of max epoch size
        self.max_epoch = max_epoch

        # Parameters for early stopping
        self.early_stopping = early_stopping
        self.patience = patience

        # Fine-tune word embeddings
        self.train_embeddings = train_embeddings

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
