from abc import ABC, abstractmethod
from typing import Callable, Iterable, Literal
import numpy as np
from numpy.typing import NDArray
from attrs import mutable, frozen, field, setters
import copy

from multilayer_simulator.material import Material, ConstantIndex


class Structure(ABC):
    """
    Interface for a class representing a structure to be optically modeled in 1D.
    """

    @abstractmethod
    def index(
        self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1
    ) -> NDArray[np.float_]:
        pass

    @property
    @abstractmethod
    def thickness(self) -> NDArray[np.float_]:
        pass


@mutable  # Has to be mutable to allow binding of index function without hacks
class Layer(Structure):
    """
    Represent a single layer.
    """

    _index: Callable[
        [NDArray[np.float_], Literal[1, 2, 3], NDArray[np.float_]], NDArray[np.float_]
    ]  # TODO: Type this as callback protocol instead
    thickness: NDArray[np.float_] = field(
        converter=np.atleast_1d, on_setattr=setters.convert
    )

    @thickness.default
    def _thickness_default(
        self=None,
    ):  # set as function so can be used in factory class methods
        return 0

    @classmethod
    def from_material(
        cls,
        material: Material = ConstantIndex(1),
        thickness: float = _thickness_default(),
    ) -> 'Layer':
        """
        Create Layer from Material and thickness.

        :param material: Material that the Layer is composed of, defaults to ConstantIndex(1) (i.e. vacuum)
        :type material: Material, optional
        :param thickness: Thickness of the Layer, defaults to _thickness_default()
        :type thickness: float, optional
        :return: Instance of Layer
        :rtype: Layer
        """
        index = material.index
        return cls(index, thickness)

    def index(
        self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1
    ) -> NDArray[np.float_]:
        return self._index(frequencies, component)


@mutable
class Multilayer(Structure):
    """
    Represent a multilayer.
    """

    layers: list[Layer]

    def index(
        self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1
    ) -> NDArray[np.float_]:
        index_array = np.array(
            [layer.index(frequencies, component) for layer in self.layers]
        )
        return index_array

    @property
    def thickness(self) -> NDArray[np.float_]:
        thickness_array = np.array(
            [layer.thickness for layer in self.layers]
        ).squeeze()  # squeeze from nx1 to n array
        return thickness_array
