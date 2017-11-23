# anaGo
***anaGo*** is a state-of-the-art library for sequence labeling using Keras. 

anaGo can performs named-entity recognition (NER), part-of-speech tagging (POS tagging), semantic role labeling (SRL) and so on for **many languages**. 
For example, **English Named-Entity Recognition** is shown in the following picture:
<img src="https://github.com/Hironsan/anago/blob/docs/docs/images/example.en2.png?raw=true">

**Japanese Named-Entity Recognition** is shown in the following picture:
<img src="https://github.com/Hironsan/anago/blob/docs/docs/images/example.ja2.png?raw=true">

Similarly, **you can solve your task for your language.**
You have only to prepare input and output data. :)

## Feature Support
anaGo provide following features:
* learning your own task without any knowledge.
* defining your own model.
* ~~(Not yet supported)downloading learned model for many tasks. (e.g. NER, POS Tagging, etc...)~~


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

## Data and Word Vectors
The data must be in the following format(tsv).
We provide an example in train.txt:

```
EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
.	O

Peter	B-PER
Blackburn	I-PER
```

You also need to download [GloVe vectors](https://nlp.stanford.edu/projects/glove/) and store it in *data/glove.6B* directory.

## Get Started
### Import
First, import the necessary modules:
```python
import anago
from anago.reader import load_data_and_labels
```

### Loading data
After importing the modules, load training, validation and test dataset:
```python
x_train, y_train = load_data_and_labels('train.txt')
x_valid, y_valid = load_data_and_labels('valid.txt')
x_test, y_test = load_data_and_labels('test.txt')
```

Now we are ready for training :)


### Training a model
Let's train a model. For training a model, we can use train method:
```python
model = anago.Sequence()
model.train(x_train, y_train, x_valid, y_valid)
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


### Evaluating a model
To evaluate the trained model, we can use eval method:

```python
model.eval(x_test, y_test)
```

After evaluation, F1 value is output:
```commandline
- f1: 90.67
```

### Tagging a sentence
Let's try tagging a sentence, "President Obama is speaking at the White House."
We can do it as follows:
```python
>>> words = 'President Obama is speaking at the White House.'.split()
>>> model.analyze(words)
{
  'words': [
             'President',
             'Obama',
             'is',
             'speaking',
             'at',
             'the',
             'White',
             'House.'
            ],
  'entities': [
    {
      'beginOffset': 1,
      'endOffset': 2,
      'score': 1.0,
      'text': 'Obama',
      'type': 'PER'
    },
    {
      'beginOffset': 6,
      'endOffset': 8,
      'score': 1.0,
      'text': 'White House.',
      'type': 'ORG'
    }
  ]
}
```


## Reference
This library uses bidirectional LSTM + CRF model based on
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
by Lample, Guillaume, et al., NAACL 2016.