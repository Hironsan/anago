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


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


NONE = 'O'
def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


import numpy as np
def run_evaluate(y_true, y_pred, tags):
    """
    Evaluates performance on test set
    Args:
        sess: tensorflow session
        test: dataset that yields tuple of sentences, tags
        tags: {tag: index} dictionary
    Returns:
        accuracy
        f1 score
    """
    labels = [y.argmax(axis=1) for y in y_true]
    labels_pred = [y.argmax(axis=1) for y in y_pred]
    sequence_lengths = [y.argmin() for y in labels]
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]
        accs += [a == b for (a, b) in zip(lab, lab_pred)]
        lab_chunks = set(get_chunks(lab, tags))
        lab_pred_chunks = set(get_chunks(lab_pred, tags))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    return acc, f1