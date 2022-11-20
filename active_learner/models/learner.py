from abc import ABC, abstractmethod


class Learner(ABC):
    """Base class for all learners."""

    @abstractmethod
    def __init__(self, model, **kwargs):
        pass
