import os


class Config(object):
    # data settings
    data_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
    save_path = os.path.join(os.path.dirname(__file__), '../models/')
    log_dir = os.path.join(os.path.dirname(__file__), '../logs/')
    glove_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.100d.txt')

    # model settings
    dropout = 0.5           # The probability of keeping weights in the dropout layer
    char_dim = 25          # Character embedding dimension
    word_dim = 100          # Word embedding dimension
    lstm_size = 100         # The number of hidden units in lstm
    char_lstm_size = 25    # The number of hidden units in char lstm
    use_char = True         # Use character feature
    crf = True              # Use CRF

    # training settings
    batch_size = 20          # The batch size
    clip_value = 0.0         # The clip value
    learning_rate = 0.001    # The initial value of the learning rate
    lr_decay = 0.9           # The decay of the learning rate for each epoch
    lr_method = 'adam'       # The learning method
    max_epoch = 15           # The number of epochs
    reload=False             # Reload model
    nepoch_no_imprv = 3      # For early stopping
    train_embeddings = True  # Fine-tune word embeddings
    early_stopping = True


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"

        # Number of unique words in the vocab (plus 1, for <UNK>).
        # The default value is larger than the expected actual vocab size to allow
        # for differences between tokenizer versions used in preprocessing. There is
        # no harm in using a value greater than the actual vocab size, but using a
        # value less than the actual vocab size will result in an error.
        self.vocab_size = 12000

        # Batch size.
        self.batch_size = 32

        # File containing an Inception v3 checkpoint to initialize the variables
        # of the Inception model. Must be provided when starting training for the
        # first time.
        self.inception_checkpoint_file = None

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.num_lstm_units = 512

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 586363

        # Optimizer for training the model.
        self.optimizer = "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_inception_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
