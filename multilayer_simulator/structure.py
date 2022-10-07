

from abc import ABC, abstractmethod
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray
from attrs import mutable, frozen, field

from multilayer_simulator.material import Material

class Structure(ABC):
    """
    Interface for a class representing a structure to be optically modeled in 1D.
    """
    
    @abstractmethod
    def index(self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1) -> NDArray[np.float_]:
        pass
    
    @property
    @abstractmethod
    def thickness(self) -> NDArray[np.float_]:
        pass

@frozen
class Layer(Structure):
    """
    Represent a single layer.
    """
    index: Callable
    thickness: float
    
    @classmethod
    def from_material(cls, material: Material, thickness: float):
        pass
        


class Multilayer(Structure):
    """
    Represent a multilayer.
    """
    pass
