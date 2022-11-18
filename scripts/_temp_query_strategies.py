from random import randint


def query_random(examples, exclude, n_instances):
    """Random sampling strategy."""
    n_queried = 0
    max_idx = len(examples)
    _inner_exclude = set(exclude)
    while n_queried < n_instances:
        idx = randint(0, max_idx)
        if idx not in _inner_exclude:
            _inner_exclude.add(idx)
            n_queried += 1
            yield idx, examples[idx]


def query_least_confidence(nlp, examples, exclude, n_instances, spans_key):
    """Least confidence sampling strategy for multilabeled data for
    SpanCategorizer. Based on mean score of all labels for each span.
    """
    def _get_least_confident():
        """Calculate mean score of all labels for single span."""
        # TODO: Refactor for better performance
        scores = []
        score_indexes = []
        for i in range(ex_len):
            if i in _inner_exclude:
                continue
            example = _inner_examples[i]
            pred = nlp(example.text)
            spans = pred.spans[spans_key]
            if spans:
                example_mean_score = spans.attrs["scores"].mean()
                score_indexes.append(i)
                scores.append(example_mean_score)
        min_score_idx = scores.index(min(scores))
        return score_indexes[min_score_idx]

    _inner_examples = list(examples)
    _inner_exclude = set(exclude)
    n_queried = 0
    ex_len = len(_inner_examples)
    while n_queried < n_instances:
        idx = _get_least_confident()
        _inner_exclude.add(idx)
        example = _inner_examples[idx]
        n_queried += 1
        yield idx, example
