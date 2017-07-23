import itertools
from sklearn.metrics import classification_report


def report(y_true, y_pred, entity_to_id):
    y_true = [y.argmax() for y in itertools.chain(*y_true)]
    y_pred = [y.argmax() for y in itertools.chain(*y_pred)]

    tagset = set(entity_to_id) - {'O', '<PAD>'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

    print(classification_report(
        y_true,
        y_pred,
        labels=[entity_to_id[cls] for cls in tagset],
        target_names=tagset,
    ))