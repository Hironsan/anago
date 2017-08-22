<div align="center">
  <img src="https://github.com/Hironsan/anago/blob/develop/docs/images/anago.png?raw=true" width="350">
</div>

-----------------

# anaGo
anaGo is a state-of-the-art library for sequence labeling using Keras.

<img src="https://github.com/Hironsan/anago/blob/docs/docs/images/example.ja.png?raw=true">

This library uses bidirectional LSTM + CRF model described in
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
by Lample, Guillaume, et al., NAACL 2016.


```python
>>> trainer = anago.Trainer(config)
>>> trainer.train(x_train, y_train, x_valid, y_valid)
Epoch 1/15
...
702/703 [=======================>......] - ETA: 1s - loss: 1.6950
703/703 [==============================] - 1021s - loss: 1.6619 - f1: 88.40
>>> sent = 'President Obama is speaking at the White House.'
>>> tagger = anago.Tagger(config)
>>> tagger.get_entities(sent)
{'Person': ['Obama'], 'LOCATION': ['White House']}
>>> tagger.tag(sent)
[('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
 ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
 ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
```

## Feature Support
anaGo provide following features:
* downloading learned model for many tasks. (e.g. NER, POS Tagging, etc...)
* learning your own task without any knowledge.
* defining your own model.


## Install
To install anaGo, simply run:

```commandline
$ pip install anago
```

or install from the repository:

```commandline
$ git clone https://github.com/Hironsan/anago.git
$ cd anago
$ pip install -r requirements.txt
```

## Get Started
### Import
```python
import anago
from anago.config import Config
from anago.data.reader import load_data_and_labels
```

### Loading data
```python
config = Config()
train_path = os.path.join(config.data_path, 'train.txt')
valid_path = os.path.join(config.data_path, 'valid.txt')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
```

### Training
```python
trainer = anago.Trainer(config)
trainer.train(x_train, y_train, x_valid, y_valid)
```

### Evaluation
```python
saved_model = 'path_to_model'
evaluator = anago.Evaluator(config, saved_model)
evaluator.eval(x_test, y_test)
```

### Tagging
```python
saved_model = 'path_to_model'
tagger = anago.Tagger(config, saved_model)
tagger.get_entities(sent)
```

## Documentation
Coming Soon...

## How to Contribute
Coming Soon...