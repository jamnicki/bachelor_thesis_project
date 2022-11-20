import os
import numpy as np


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
        texts = [
            example.text
            for i, example in enumerate(examples)
            if i not in exclude
        ]
        cpus = os.cpu_count()
        predictions = nlp.pipe(texts, disable=disabled_comps, n_process=cpus)
        for i, pred in enumerate(predictions):
            spans = pred.spans[spans_key]
            if spans:
                scores[i] = spans.attrs["scores"].mean()
                scores_idxs[i] = i
        non_zero_score_argmin = np.argmin(scores[np.nonzero(scores)])
        return int(scores_idxs[non_zero_score_argmin])

    disabled_comps = set(nlp.pipe_names) - set(included_components)
    n_queried = 0
    ex_len = len(examples)
    while n_queried < n_instances:
        idx = _get_least_confident()
        exclude.add(idx)
        example = examples[idx]
        n_queried += 1
        yield idx, example
