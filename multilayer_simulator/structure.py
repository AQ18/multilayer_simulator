from abc import ABC, abstractmethod
from typing import Callable, Iterable, Literal, Optional, Union
import numpy as np
import dis
from numpy.typing import NDArray
from attrs import mutable, frozen, field, validators, setters
import copy

from multilayer_simulator.helpers.mixins import NamedMixin
from multilayer_simulator.material import Material, ConstantIndex


class Structure(ABC, NamedMixin):
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
    ] = field(
        default=ConstantIndex(1).index, eq=dis.dis, repr=False
    )  # TODO: Type this as callback protocol instead
    thickness: NDArray[np.float_] = field(
        converter=np.atleast_1d,
        validator=validators.ge(0),
        on_setattr=setters.pipe(setters.convert, setters.validate),
    )
    material: Optional[Material] = field(default=None, eq=False)

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
        name: Optional[Union[str, Callable]] = None,
    ) -> "Layer":
        """
        Create Layer from Material and thickness.

        :param material: Material that the Layer is composed of, defaults to ConstantIndex(1) (i.e. vacuum).
        :type material: Material, optional
        :param thickness: Thickness of the Layer, defaults to _thickness_default().
        :type thickness: float, optional
        :param name: Name of the Layer, defaults to None.
            If None, use material.name.
            If callable, use name(material.name). Must accept a single argument.
            Otherwise (e.g. if string), use name itself.
        :type name: Optional[Union[str, Callable]], optional
        :return: Instance of Layer.
        :rtype: Layer
        """
        index = material.index
        if name is None:
            name = material.name
        elif callable(name):
            name = name(material.name)
        else:
            name = name
        return cls(index, thickness, material=material, name=name)

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
    unit_cell: Optional[list[Layer]] = field(default=None, eq=False, repr=False)

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

    @property
    def stack_layers(self):
        """
        Just a convenient shorthand for layers[1:-1], i.e. excluding
        the incident and exit layers.

        :return: _description_
        :rtype: _type_
        """
        return self.layers[1:-1]

    @classmethod
    def from_given_unit_cell(
        cls,
        unit_cell: Iterable[Layer],
        incident_layer: Layer = Layer.from_material(),
        exit_layer: Layer = Layer.from_material(),
        num_periods: int = 1,
        copy_layers: bool = True,
    ) -> "Multilayer":
        """
        Return a periodic Multilayer containing some number of repetitions of a unit cell.
        The Multilayer is not guaranteed to be periodic after instantiation - internal
        Layer objects can be deleted or modified.
        If copy_layers is True, then every Layer is deepcopied.
        Otherwise, the Layers will be the same objects across unit cells,
        and also across instances if they are used in multiple constructors.
        This can save memory and simplify modifications to the unit cell, but may lead to
        unexpected behaviour.

        :param unit_cell: The layers that will be repeated to make up the multilayer
        :type unit_cell: Iterable[Layer]
        :param incident_layer: Layer defining the incident medium, defaults to Layer.from_material()
        :type incident_layer: Layer, optional
        :param exit_layer: Layer defining the exit medium, defaults to Layer.from_material()
        :type exit_layer: Layer, optional
        :param num_periods: Number of repetitions of the unit cell, defaults to 1
        :type num_periods: int, optional
        :param copy_layers: Whether to deepcopy every Layer, defaults to True
        :type copy_layers: bool, optional
        :return: Instance of Multilayer
        :rtype: Multilayer
        """
        layers = []
        if copy_layers:
            layers.append(copy.deepcopy(incident_layer))
            for _ in range(num_periods):
                for layer in unit_cell:
                    layers.append(copy.deepcopy(layer))
            layers.append(copy.deepcopy(exit_layer))
        else:
            layers.append(incident_layer)
            for _ in range(num_periods):
                for layer in unit_cell:
                    layers.append(layer)
            layers.append(exit_layer)
        return cls(layers, unit_cell=unit_cell)

    def from_own_unit_cell(
        self,
        unit_cell: Optional[Iterable[Layer]] = None,
        incident_layer: Layer = Layer.from_material(), # TODO default should be to use existing incident and exit layers
        exit_layer: Layer = Layer.from_material(),
        num_periods: int = 1,
        copy_layers: bool = True,
    ) -> "Multilayer":
        """
        Return a new periodic Multilayer containing some number of repetitions of a unit cell.
        If unit_cell is None (default), will use the Multilayer's own unit_cell property if it
        is not None (default), else stack_layers (the entire stack excluding incident/exit layers).
        See the classmethod from_given_unit_cell() for more details.

        :param unit_cell: The layers that will be repeated to make up the multilayer, defaults to None
        :type unit_cell: Optional[Iterable[Layer]], optional
        :param incident_layer: Layer defining the incident medium, defaults to Layer.from_material()
        :type incident_layer: Layer, optional
        :param exit_layer: Layer defining the exit medium, defaults to Layer.from_material()
        :type exit_layer: Layer, optional
        :param num_periods: Number of repetitions of the unit cell, defaults to 1
        :type num_periods: int, optional
        :param copy_layers: Whether to deepcopy every Layer, defaults to True
        :type copy_layers: bool, optional
        :return: Instance of Multilayer
        :rtype: Multilayer
        """
        if unit_cell is None:
            unit_cell = self.stack_layers if self.unit_cell is None else self.unit_cell
        return self.from_given_unit_cell(
            unit_cell=unit_cell,
            incident_layer=incident_layer,
            exit_layer=exit_layer,
            num_periods=num_periods,
            copy_layers=copy_layers,
        )
