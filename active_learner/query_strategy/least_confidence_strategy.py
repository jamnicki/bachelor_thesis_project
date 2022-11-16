import numpy as np


def query_least_confidence(nlp, examples, exclude, n_instances, spans_key):
    """Least confidence sampling strategy for multilabeled data for
    SpanCategorizer. Based on mean score of all labels for each span.
    """
    def _get_least_confident():
        """Calculate mean score of all labels for single span."""
        example_conf_scores = np.zeros(ex_len)
        for i in range(ex_len):
            if i in _inner_exclude:
                continue
            example = _inner_examples[i]
            pred = nlp(example.text)
            spans = pred.spans[spans_key]
            if spans:
                example_conf_scores[i] = spans.attrs["scores"].mean()
            else:
                example_conf_scores[i] = 0.0
            _inner_exclude.add(i)

        return example_conf_scores.argmin()

    _inner_examples = list(examples)
    _inner_exclude = set(exclude)
    n_queried = 0
    ex_len = len(_inner_examples)
    while n_queried < n_instances:
        idx = _get_least_confident()
        example = _inner_examples[idx]
        n_queried += 1
        yield idx, example
