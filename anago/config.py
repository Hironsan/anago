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
    user_char = False
