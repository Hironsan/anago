import tensorflow as tf

from anago.data_utils import pad_sequences


class LstmCrfModel(object):

    def __init__(self, config, embeddings, vocab_char, vocab_tag):
        """
        Args:
            config: class with hyper parameters
            embeddings: np array with embeddings
            nchars: (int) size of chars vocabulary
        """
        self.config     = config
        self.embeddings = embeddings
        self.nchars     = len(vocab_char)
        self.ntags      = len(vocab_tag)

    def build_inputs(self):
        """Builds the ops for reading input data.

        Outputs:
            self.word_ids
            self.char_ids
            self.sequence_lengths
            self.word_lengths
            self.labels
            self.dropout
        """
        self.word_ids = tf.placeholder(tf.int32, (None, None), name='word_ids')
        self.char_ids = tf.placeholder(tf.int32, (None, None, None), name='char_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, (None), name='sequence_lengths')
        self.word_lengths = tf.placeholder(tf.int32, (None, None), name='word_lengths')
        self.labels = tf.placeholder(tf.int32, (None, None), name='labels')
        self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def build_word_embeddings(self):
        """Builds the word embeddings.

        Inputs:
            self.word_ids
            self.char_ids
            self.word_lengths
        Output:
            self.word_embeddings
        """
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable(self.embeddings,
                                           name='_word_embeddings',
                                           dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name='word_embeddings')

        with tf.variable_scope('chars'):
            if self.config.use_char:
                _char_embeddings = tf.get_variable(name='_char_embeddings',
                                                   dtype=tf.float32,
                                                   shape=[self.nchars, self.config.char_dim])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids,
                                                         name='char_embeddings')
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.char_dim])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])

                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size, state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw,
                                                                                      char_embeddings,
                                                                                      sequence_length=word_lengths,
                                                                                      dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.config.char_lstm_size])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def build_model(self):
        """Builds the model.

        Inputs:
            self.word_embeddings
            self.sequence_lengths
        Output:
            self.logits
        """
        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope('proj'):
            W = tf.get_variable('W', shape=[2 * self.config.lstm_size, self.ntags], dtype=tf.float32)

            b = tf.get_variable('b', shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.lstm_size])
            pred = tf.matmul(output, W) + b
            logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

        self.logits = logits

    def build_loss(self):
        """Builds the loss

        Inputs:
            self.logits
            self.labels
            self.sequence_lengths
        Output:
            self.loss
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            batch_loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            batch_loss = tf.reduce_mean(losses)

        tf.losses.add_loss(batch_loss)
        self.loss = batch_loss

    def build(self):
        self.build_inputs()
        self.build_word_embeddings()
        self.build_model()
        self.build_loss()

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of words.
                A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.use_char:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_char:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr:
            feed[self.lr] = lr

        if dropout:
            feed[self.dropout] = dropout

        return feed, sequence_lengths
