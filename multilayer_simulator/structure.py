

from abc import ABC, abstractmethod


class Structure(ABC):
    """
    Interface for a class representing a structure to be optically modeled in 1D.
    """
    
    @abstractmethod
    def n(self, frequencies: NDArray[np.float64]):
        pass
    
    @abstractmethod
    def d(self)
        pass
    
class Multilayer(Structure):
    """
    Represent a multilayer.
    """
    pass

class Layer(Structure):
    """
    Represent a single layer.
    """
    pass
