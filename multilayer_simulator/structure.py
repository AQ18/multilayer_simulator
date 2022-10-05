

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from attrs import mutable, field

from material import Material

class Structure(ABC):
    """
    Interface for a class representing a structure to be optically modeled in 1D.
    """
    
    @abstractmethod
    def index(self, frequencies: NDArray[np.float64], component: Literal[1, 2, 3] = 1):
        pass
    
    @abstractmethod
    def thickness(self):
        pass

@mutable
class Layer(Structure):
    """
    Represent a single layer.
    """
    
    
    @classmethod
    def from_material(cls, material: Material, thickness: float):
        
        


class Multilayer(Structure):
    """
    Represent a multilayer.
    """
    pass
