from random import randint


def query_random(records, exclude, n_instances):
    """Random sampling strategy."""
    n_queried = 0
    max_idx = len(records) - 1
    while n_queried < n_instances:
        idx = randint(0, max_idx)
        if idx not in exclude:
            exclude.add(idx)
            n_queried += 1
            yield idx, records[idx]
