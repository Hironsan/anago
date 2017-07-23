class Config(object):

    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50  # cut texts after this number of words (among top max_features most common words)
    hidden_size = 100
    epoch_size = 5
    dropout = 0.5
    lr_decay = 0.5
    batch_size = 64
    vocab_size = 10000
    word_emb_size = 100

    def __init__(self, word_to_id, entity_to_id):
        self.vocab_size = len(word_to_id)
        self.num_classes = len(entity_to_id)