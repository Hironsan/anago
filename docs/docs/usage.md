# Usage

## Data Loading

```python
>>> from anago.utils import load_data_and_labels

>>> x_train, y_train = load_data_and_labels('train.txt')
>>> x_test, y_test = load_data_and_labels('test.txt')
>>> x_train[0]
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
>>> y_train[0]
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

Todo: write data format.

## Training

```python
>>> import anago

>>> model = anago.Sequence()
>>> model.fit(x_train, y_train, epochs=15)
Epoch 1/15
541/541 [==============================] - 166s 307ms/step - loss: 12.9774
...
```

## Evaluation

```python
>>> model.score(x_test, y_test)
80.20  # f1-micro score
# For more performance, you have to use pre-trained word embeddings.
# For now, anaGo's best score is 90.70 f1-micro score.
```

## Tagging

```python
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

To download a pre-trained model, call `download` function:

```python
>>> from anago.utils import download

>>> url = 'https://storage.googleapis.com/chakki/datasets/public/ner/model_en.zip'
>>> download(url)
'Downloading...'
'Complete!'
>>> model = anago.Sequence.load('weights.h5', 'params.json', 'preprocessor.pickle')
>>> model.score(x_test, y_test)
90.61
```