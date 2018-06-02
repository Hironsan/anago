# anaGo

**anaGo** is a Python library for sequence labeling, implemented in Keras.

anaGo can solve sequence labeling tasks such as named entity recognition (NER), Part-of-Speech tagging (POS tagging), semantic role labeling (SRL) and so on. Unlike traditional sequence labeling solver, we don't need to define any language dependent features. Thus, we can easily expand for any languages.

As an example of anaGo, the following images show named entity recognition in English and Japanese:

![English NER](https://github.com/Hironsan/anago/blob/docs/docs/images/example.en2.png?raw=true)

![Japanese NER](https://github.com/Hironsan/anago/blob/docs/docs/images/example.ja2.png?raw=true)

Behold, the power of anaGo:

```python
>>> import anago
>>> from anago.utils import load_data_and_labels
>>> x_train, y_train = load_data_and_labels('train.txt')
>>> x_test, y_test = load_data_and_labels('test.txt')
>>> model = anago.Sequence()
>>> model.fit(x_train, y_train)
Epoch 1/15
541/541 [==============================] - 176s 324ms/step - loss: 12.8500 - f1: 82.25
...
>>> model.score(x_test, y_test)
90.67  # f1 score
>>> text = 'President Obama is speaking at the White House.'
>>> model.analyze(text)
{
    "words": [
        "President",
        "Obama",
        "is",
        "speaking",
        "at",
        "the",
        "White",
        "House."
    ],
    "entities": [
        {
            "beginOffset": 1,
            "endOffset": 2,
            "score": 1,
            "text": "Obama",
            "type": "PER"
        },
        {
            "beginOffset": 6,
            "endOffset": 8,
            "score": 1,
            "text": "White House.",
            "type": "LOC"
        }
    ]
}
```

## Feature Support

anaGo supports following features:

* Model Training
* Model Evaluation
* Tagging Text
* No Feature Definition
* Custom Model Support
* Downloading pre-trained model

anaGo officially supports Python 3.4–3.6.

## Installation

To install anaGo, simply use `pip`:

```bash
$ pip install anago
```

or install from the repository:

```bash
$ git clone https://github.com/Hironsan/anago.git
$ cd anago
$ python setup.py install
```

<!--
## Data and Word Vectors

Training data takes a tsv format.
The following text is an example of training data:

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

anaGo supports pre-trained word embeddings like [GloVe vectors](https://nlp.stanford.edu/projects/glove/).

### Downloading pre-trained models

To download a pre-trained model, call `download` function:

```python
from anago.utils import download

dir_path = 'models'
url = 'https://storage.googleapis.com/chakki/datasets/public/models.zip'
download(url, dir_path)
model = anago.Sequence.load(dir_path)
```
-->

## Reference

This library uses bidirectional LSTM + CRF model based on
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
by Lample, Guillaume, et al., NAACL 2016.