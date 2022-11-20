import os
import numpy as np


def query_least_confidence(nlp, included_components, examples,
                           exclude, n_instances, spans_key):
    """Least confidence sampling strategy for multilabeled data for spaCy's
    SpanCategorizer. Based on mean score of all labels for each span.
    """
    def _get_least_confident():
        for _idx in indexes_by_score:
            include = include_arr[_idx]
            if include:
                return _idx

    disabled_comps = set(nlp.pipe_names) - set(included_components)
    texts = [
            example.text
            for i, example in enumerate(examples)
            if i not in exclude
        ]
    n_process = os.cpu_count()
    predictions = nlp.pipe(texts, disable=disabled_comps, n_process=n_process)

    ex_len = len(examples)
    include_arr = np.repeat(True, ex_len)
    for ex in exclude:
        include_arr[ex] = False

    scores = np.zeros(ex_len)
    for i, pred in enumerate(predictions):
        spans = pred.spans[spans_key]
        if spans:
            scores[i] = spans.attrs["scores"].mean()
    indexes_by_score = np.argsort(scores)  # ascending order

    n_queried = 0
    while n_queried < n_instances:
        idx = _get_least_confident()
        include_arr[idx] = False
        example = examples[idx]
        n_queried += 1
        yield idx, example
