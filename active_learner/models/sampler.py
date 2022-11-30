import numpy as np

from abc import ABC
from typing import Iterable, Union, Generator, NewType

Idx = NewType("Idx", int)
Score = NewType("Score", Union[float, int])


class Sampler(ABC):
    """Abstract base class for all samplers."""


class LeastConfidenceSampler(Sampler):
    """Least confidence sampling strategy based on given scores."""

    def __init__(self) -> None:
        super().__init__()
        self.indexes_by_score: Iterable[Idx] = []
        self.include_arr: np.ndarray = np.array([])

    def __call__(
        self,
        scores: Iterable[Score],
        exclude: Iterable[Idx],
        num: int,
        strict: bool = True
    ) -> Generator[Union[Idx, None], None, None]:

        scores_len = len(scores)
        if strict and scores_len - len(exclude) < num:
            raise ValueError("Number of samples cannot be larger than lenght "
                             "of the included scores!")
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        self.include_arr = np.repeat(True, scores_len)
        for ex in exclude:
            self.include_arr[ex] = False
        # ascending order
        self.indexes_by_score = np.argsort(scores, kind="stable")
        n_queried = 0
        while n_queried < num:
            idx = self._get_least_confident()
            self.include_arr[idx] = False
            n_queried += 1
            yield idx

    def _get_least_confident(self) -> Union[Idx, None]:
        """Get the index of the least confident score if not excluded."""
        for _idx in self.indexes_by_score:
            include = self.include_arr[_idx]
            if include:
                return _idx
