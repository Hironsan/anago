class Config(object):

    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 50  # cut texts after this number of words (among top max_features most common words)
    hidden_size = 100
    epoch_size = 10
    dropout = 0.5
    lr_decay = 0.9
    batch_size = 32
    vocab_size = 10000
    max_word_len = 20

    use_char = True
    nb_filters = 30
    nb_kernels = 3
