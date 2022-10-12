

from abc import ABC, abstractmethod
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray
from attrs import mutable, frozen, field, setters

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

@mutable # Has to be mutable to allow binding of index function without hacks
class Layer(Structure):
    """
    Represent a single layer.
    """
    _index: Callable[[NDArray[np.float_], Literal[1, 2, 3], NDArray[np.float_]], NDArray[np.float_]] # TODO: Type this as callback protocol instead
    thickness: NDArray[np.float_] = field(converter=np.atleast_1d, on_setattr=setters.convert)
    
    @thickness.default
    def _thickness_default(self=None): # set as function so can be used in factory class methods
        return 0
    
    # def __attrs_post_init__(self):
    #     self._index = self._index.__get__(self) # bind index function as instance method - or use property approach?
    
    @classmethod
    def from_material(cls, material: Material, thickness: float = _thickness_default()):
        index = material.index
        return cls(index, thickness)

    def index(self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1) -> NDArray[np.float_]:
        return self._index(frequencies, component)
        
    # @property
    # def thickness(self) -> NDArray[np.float_]:
    #     return np.atleast_1d(self._thickness)

class Multilayer(Structure):
    """
    Represent a multilayer.
    """
    layers: list[Layer]

    def index(self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1) -> NDArray[np.float_]:
        index_array = np.array([layer.index(frequencies, component) for layer in self.layers])
        return index_array
    
    @property
    def thickness(self) -> NDArray[np.float_]:
        thickness_array = np.array([layer.thickness for layer in self.layers]).squeeze() # squeeze from nx1 to n array
        return thickness_array
        