# Pre-trained Models

anaGo privides some pre-trained models.

| Name | Language  | Type |
|---|---|---|
| model_en  | English  | Named entity recognition  |
| model_ja  | Japanese | Named entity recognition  |
| model_ar  | Arabic   | Named entity recognition  |
|   |   |   |
|   |   |   |

## Downloading models

To download a pre-trained model, call `download` function:

```python
from anago.utils import download

url = 'https://storage.googleapis.com/chakki/datasets/public/ner/model_en.zip'
download(url)
```

## Using models with anaGo

To load a model, use `Sequence.load()`  with the model's path to the data directory:

```python
import anago

model = anago.Sequence.load('weights.h5', 'params.json', 'preprocessor.pickle')
model.analyze('President Obama is speaking at the White House.')
```