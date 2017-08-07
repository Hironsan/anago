from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from anago import config
from anago.models.bilstm_crf import LstmCrfModel

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


class Trainer(object):

    def __init__(self, *args):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.save_path = save_path
        self.max_epoch = max_epoch
        self.lr_decay = lr_decay
        self.nepoch_no_imprv = npoch_no_imprv
        self.logger = logger

    def run_epoch(self, sess, train, dev, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) number of the epoch
        """
        nbatches = (len(train.sents) + self.batch_size - 1) // self.batch_size
        prog = Progbar(target=nbatches)
        for i in range(nbatches):
            words, labels = train.next_batch(self.batch_size)

            fd, _ = self.get_feed_dict(words, labels, self.learning_rate, self.dropout)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100*acc, 100*f1))
        return acc, f1

    def train(self, train, dev):
        """
        Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
        """
        model = LstmCrfModel(model_config, mode="train", train_inception=FLAGS.train_inception)
        model.build()

        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)
            if self.reload:
                self.logger.info("Reloading the latest trained model...")
                saver.restore(sess, self.save_path)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.max_epoch):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.max_epoch))

                acc, f1 = self.run_epoch(sess, train, dev, epoch)

                # decay learning rate
                self.learning_rate *= self.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    saver.save(sess, self.save_path)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

    def train(self, dataset):
        model_config = config.ModelConfig()
        model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
        training_config = config.TrainingConfig()

        # Create training directory.
        train_dir = FLAGS.train_dir
        if not tf.gfile.IsDirectory(train_dir):
            tf.logging.info("Creating training directory: %s", train_dir)
            tf.gfile.MakeDirs(train_dir)

        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():
            # Build the model.
            model = LstmCrfModel(model_config, mode="train", train_inception=FLAGS.train_inception)
            model.build()

            # Set up the learning rate.
            learning_rate = tf.constant(training_config.initial_learning_rate)
            if training_config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (training_config.num_examples_per_epoch / model_config.batch_size)
                decay_steps = int(num_batches_per_epoch * training_config.num_epochs_per_decay)

                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=training_config.learning_rate_decay_factor,
                        staircase=True)

                learning_rate_decay_fn = _learning_rate_decay_fn

            # Set up the training ops.
            train_op = tf.contrib.layers.optimize_loss(
                loss=model.total_loss,
                global_step=model.global_step,
                learning_rate=learning_rate,
                optimizer=training_config.optimizer,
                clip_gradients=training_config.clip_gradients,
                learning_rate_decay_fn=learning_rate_decay_fn)

            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            train_dir,
            log_every_n_steps=FLAGS.log_every_n_steps,
            graph=g,
            global_step=model.global_step,
            number_of_steps=FLAGS.number_of_steps,
            init_fn=model.init_fn,
            saver=saver)


if __name__ == "__main__":
    tf.app.run()