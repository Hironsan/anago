import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec


class ChainCRF(Layer):

    def __init__(self, **kwargs):
        super(ChainCRF, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], input_shape[2])

    def _fetch_mask(self):
        mask = None
        if self.inbound_nodes:
            mask = self.inbound_nodes[0].input_masks[0]
        return mask

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_steps is None or n_steps >= 2
        self.transition_params = self.add_weight((n_classes, n_classes),
                                                 initializer='uniform',
                                                 name='transition')
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, n_steps, n_classes))]
        self.built = True

    def viterbi_decode(self, x, mask):
        viterbi_sequences = []
        transition_params = K.eval(self.transition_params)
        #logits = tf.map_fn(lambda x: x[0][:x[1]], [x, mask])
        #print(logits)
        logits = x
        sequences = tf.map_fn(lambda p: tf.contrib.crf.viterbi_decode(p, transition_params)[0], logits)
        print(sequences)
        for logit, sequence_length in zip(x, mask):
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences

    def call(self, x, mask=[2,2]):
        y_pred = self.viterbi_decode(x, mask)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        return K.in_train_phase(x, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        mask = self._fetch_mask()
        #sequence_lengths = K.reshape(mask, (-1,))
        sequence_lengths = mask
        y_t = K.argmax(y_true, -1)
        y_t = K.cast(y_t, tf.int32)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_t, sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_config(self):
        config = {
            'transition_params': initializers.serialize(self.transition_params),
        }
        base_config = super(ChainCRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    """Returns the custom objects, needed for loading a persisted model."""
    instanceHolder = {'instance': None}

    class ClassWrapper(ChainCRF):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    return {'ChainCRF': ClassWrapper, 'loss': loss}


if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Embedding
    import numpy as np

    vocab_size = 20
    n_classes = 11
    model = Sequential()
    model.add(Embedding(vocab_size, n_classes))
    layer = ChainCRF()
    model.add(layer)
    model.compile(loss=layer.loss, optimizer='sgd')

    # Train first mini batch
    batch_size, maxlen = 2, 2
    x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)

    print(x)
    print(y)