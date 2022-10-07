from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from multilayer_simulator.structure import Structure


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
        """
        Simulate the propagation of light through the structure.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float64]
        :param angles: _description_
        :type angles: NDArray[np.float64]
        """
        pass
