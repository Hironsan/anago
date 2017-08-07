"""Sequential labeling model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):

    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50  # cut texts after this number of words (among top max_features most common words)
    hidden_size = 100
    epoch_size = 5
    dropout = 0.5
    lr_decay = 0.9
    batch_size = 32
    vocab_size = 10000
    max_word_len = 20

    use_char = True
    nb_filters = 30
    nb_kernels = 3


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