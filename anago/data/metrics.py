import itertools
from sklearn.metrics import classification_report


def report(y_true, y_pred, index2chunk):
    y_true = [y.argmax() for y in itertools.chain(*y_true)]
    y_pred = [y.argmax() for y in itertools.chain(*y_pred)]

    tagset = set(index2chunk) - {'O', '<PAD>'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(index2chunk)}

    print(classification_report(
        y_true,
        y_pred,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    ))