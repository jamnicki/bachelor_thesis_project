import numpy as np


def main():
    def _get_least_confident():
        for _idx in indexes_by_score:
            include = include_arr[_idx]
            if include:
                return _idx

    dummy_data = [
        {"spans": False},
        {"spans": False},
        {"spans": {"scores": np.array([0.01, 0.55, 0.0])}},
        {"spans": False},
        {"spans": False},
        {"spans": False},
        {"spans": {"scores": np.array([0.33])}},
        {"spans": False},
        {"spans": {"scores": np.array([0.06, 0.91])}},
        {"spans": False},
        {"spans": {"scores": np.array([0.78, 0.04, 0.11, 0.22])}}
    ]

    exclude = set()
    ex_len = len(dummy_data)
    include_arr = np.repeat(True, ex_len)
    for ex in exclude:
        include_arr[ex] = False

    scores = np.zeros(ex_len)
    for i, data in enumerate(dummy_data):
        if data["spans"]:
            scores[i] = data["spans"]["scores"].mean()
    indexes_by_score = np.argsort(scores)  # ascending order

    n_instances = 11
    n_queried = 0
    while n_queried < n_instances:
        idx = _get_least_confident()
        include_arr[idx] = False
        example = dummy_data[idx]
        n_queried += 1


if __name__ == "__main__":
    main()
