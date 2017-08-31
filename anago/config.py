import os


class Config(object):

    def __init__(self):
        # data settings
        self.data_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
        self.save_path = os.path.join(os.path.dirname(__file__), '../models/')
        self.log_dir = os.path.join(os.path.dirname(__file__), '../logs/')
        self.glove_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.100d.txt')

        # model settings
        self.dropout = 0.5              # The probability of keeping weights in the dropout layer
        self.char_embedding_size = 25   # Character embedding dimension
        self.num_char_lstm_units = 25   # The number of hidden units in char lstm
        self.word_embedding_size = 100  # Word embedding dimension
        self.num_word_lstm_units = 100  # The number of hidden units in lstm
        self.vocab_size = None
        self.char_vocab_size = None
        self.use_char = True         # Use character feature
        self.crf = True              # Use CRF

        # training settings
        self.batch_size = 20          # The batch size
        self.clip_gradients = 5.0     # The clip value
        self.learning_rate = 0.001    # The initial value of the learning rate
        self.lr_decay = 0.9           # The decay of the learning rate for each epoch
        self.optimizer = 'adam'       # The learning method
        self.max_epoch = 15           # The number of epochs
        self.nepoch_no_imprv = 3      # For early stopping
        self.train_embeddings = True  # Fine-tune word embeddings
        self.early_stopping = True


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        # Number of unique words in the vocab (plus 2, for <UNK>, <PAD>).
        self.vocab_size = None
        self.char_vocab_size = None

        # Batch size.
        self.batch_size = 32

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.char_embedding_size = 25
        self.num_char_lstm_units = 25
        self.word_embedding_size = 100
        self.num_word_lstm_units = 100

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.dropout = 0.5

        # If True, use character feature.
        self.char_feature = True

        # If True, use crf.
        self.crf = True


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""

        # Batch size
        self.batch_size = 20

        # Optimizer for training the model.
        self.optimizer = 'adam'

        # Learning rate for the initial phase of training.
        self.learning_rate = 0.001
        self.lr_decay = 0.9

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # The number of max epoch size
        self.max_epoch = 15

        # Parameters for early stopping
        self.early_stopping = True
        self.patience = 3

        # Fine-tune word embeddings
        self.train_embeddings = True

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
