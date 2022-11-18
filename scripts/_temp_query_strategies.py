import numpy as np
from random import randint


def query_random(examples, exclude, n_instances):
    """Random sampling strategy."""
    n_queried = 0
    max_idx = len(examples) - 1
    _inner_exclude = set(exclude)
    while n_queried < n_instances:
        idx = randint(0, max_idx)
        if idx not in _inner_exclude:
            _inner_exclude.add(idx)
            n_queried += 1
            yield idx, examples[idx]


def query_least_confidence(nlp, included_components, examples,
                           exclude, n_instances, spans_key):
    """Least confidence sampling strategy for multilabeled data for spaCy's
    SpanCategorizer. Based on mean score of all labels for each span.
    """
    def _get_least_confident():
        """Calculate mean score of all labels for single span."""
        scores = np.repeat(2.0, ex_len)  # 2 means unscored, 0 < score < 1
        # idx < 0 means unscored
        scores_idxs = np.linspace(-1, 0, ex_len, endpoint=False)
        for i in range(ex_len):
            if i in exclude:
                continue
            example = examples[i]
            with nlp.select_pipes(enable=included_components):
                pred = nlp(example.text)
            spans = pred.spans[spans_key]
            if spans:
                scores[i] = spans.attrs["scores"].mean()
                scores_idxs[i] = i
        non_zero_score_argmin = np.argmin(scores[np.nonzero(scores)])
        return int(scores_idxs[non_zero_score_argmin])

    n_queried = 0
    ex_len = len(examples)
    while n_queried < n_instances:
        idx = _get_least_confident()
        exclude.add(idx)
        example = examples[idx]
        n_queried += 1
        yield idx, example
