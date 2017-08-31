# anaGo
***anaGo*** is a state-of-the-art library for sequence labeling using Keras. 

anaGo can performs named-entity recognition (NER), part-of-speech tagging (POS tagging), semantic role labeling (SRL) and so on. 

<img src="https://github.com/Hironsan/anago/blob/docs/docs/images/example.ja.png?raw=true">


## Feature Support
anaGo provide following features:
* learning your own task without any knowledge.
* defining your own model.
* downloading learned model for many tasks. (e.g. NER, POS Tagging, etc...)


## Install
To install anaGo, simply run:

```
$ pip install anago
```

or install from the repository:

```
$ git clone https://github.com/Hironsan/anago.git
$ cd anago
$ pip install -r requirements.txt
```

## Get Started
### Import
First, import the necessary modules:
```python
import os
import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.data.preprocess import prepare_preprocessor
from anago.config import ModelConfig, TrainingConfig
```
They include loading modules, a preprocessor and configs.


And set parameters to use later:
```python
DATA_ROOT = 'data/conll2003/en/ner'
SAVE_ROOT = './models'  # trained model
LOG_ROOT = './logs'     # checkpoint, tensorboard
embedding_path = './data/glove.6B/glove.6B.100d.txt'
model_config = ModelConfig()
training_config = TrainingConfig()
```

### Loading data

After importing the modules, read data for training, validation and test:
```python
train_path = os.path.join(DATA_ROOT, 'train.txt')
valid_path = os.path.join(DATA_ROOT, 'valid.txt')
test_path = os.path.join(DATA_ROOT, 'test.txt')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
x_test, y_test = load_data_and_labels(test_path)
```

After reading the data, prepare preprocessor and pre-trained word embeddings:
```python
p = prepare_preprocessor(x_train, y_train)
p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))

embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
model_config.char_vocab_size = len(p.vocab_char)
```

Now we are ready for training :)


### Training a model
Let's train a model. For training a model, we can use ***Trainer***. 
Trainer manages everything about training.
Prepare an instance of Trainer class and give train data and valid data to train method:
```
trainer = anago.Trainer(model_config, training_config, checkpoint_path=LOG_ROOT, save_path=SAVE_ROOT,
                        preprocessor=p, embeddings=embeddings)
trainer.train(x_train, y_train, x_valid, y_valid)
```

If training is progressing normally, progress bar will be displayed as follows:

```commandline
...
Epoch 3/15
702/703 [============================>.] - ETA: 0s - loss: 60.0129 - f1: 89.70
703/703 [==============================] - 319s - loss: 59.9278   
Epoch 4/15
702/703 [============================>.] - ETA: 0s - loss: 59.9268 - f1: 90.03
703/703 [==============================] - 324s - loss: 59.8417   
Epoch 5/15
702/703 [============================>.] - ETA: 0s - loss: 58.9831 - f1: 90.67
703/703 [==============================] - 297s - loss: 58.8993   
...
```


### Evaluation for a model
To evaluate the trained model, we can use ***Evaluator***.
Evaluator performs evaluation.
Prepare an instance of Evaluator class and give test data to eval method:

```
p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
model_config.vocab_size = len(p.vocab_word)
model_config.char_vocab_size = len(p.vocab_char)

weights = os.path.join(SAVE_ROOT, 'model_weights.h5')

evaluator = anago.Evaluator(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)
evaluator.eval(x_test, y_test)
```

After evaluation, F1 value is output:
```commandline
- f1: 90.67
```

### Tagging a sentence
To tag any text, we can use ***Tagger***.
Prepare an instance of Tagger class and give text to tag method:
```
p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
model_config.vocab_size = len(p.vocab_word)
model_config.char_vocab_size = len(p.vocab_char)

weights = os.path.join(SAVE_ROOT, 'model_weights.h5')

tagger = anago.Tagger(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)
```

Let's try tagging a sentence, "President Obama is speaking at the White House."
We can do it as follows:
```python
>>> sent = 'President Obama is speaking at the White House.'
>>> print(tagger.tag(sent))
[('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
 ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
 ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
>>> print(tagger.get_entities(sent))
{'Person': ['Obama'], 'LOCATION': ['White House']}
```


## Reference
This library uses bidirectional LSTM + CRF model based on
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
by Lample, Guillaume, et al., NAACL 2016.