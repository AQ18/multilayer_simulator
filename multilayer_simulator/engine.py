from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from structure import Structure


class Engine(ABC):
    """
    Interface for a class representing an optical physics engine.
    """

    @abstractmethod
    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float64],
        angles: NDArray[np.float64],
        **kwargs
    ):
        pass
