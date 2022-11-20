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
