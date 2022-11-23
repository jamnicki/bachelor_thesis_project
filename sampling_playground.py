import logging
import numpy as np
from active_learner.models.sampler import LeastConfidenceSampler

logging.basicConfig(level=logging.DEBUG)


def main():
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

    exclude = set([1, 4])
    scores = [
        data["spans"]["scores"].mean() if data["spans"] else 0.0
        for data in dummy_data
    ]

    sampler = LeastConfidenceSampler()
    for idx in sampler(scores, exclude, num=8):
        print(f"{idx=}, {dummy_data[idx]=}, {scores[idx]=}")


if __name__ == "__main__":
    main()
