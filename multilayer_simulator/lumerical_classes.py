from typing import ClassVar, Dict, Iterable, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from attrs import frozen, field, Factory
import lumapi

from multilayer_simulator.engine import Engine
from helpers.helpers import filter_mapping
from multilayer_simulator.structure import Structure


@frozen
class STACKRT(Engine):
    session: lumapi.FDTD

    def __call__(
        self,
        index: NDArray[np.float_],
        thickness: NDArray[np.float_],
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
    ) -> Dict[str, NDArray[np.float_]]:
        """
        Thin wrapper around lumapi.FDTD.stackrt.

        :param index: _description_
        :type index: NDArray[np.float_]
        :param thickness: _description_
        :type thickness: NDArray[np.float_]
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :return: _description_
        :rtype: Dict[str, NDArray[np.float_]]
        """
        n = index
        d = thickness
        f = frequencies
        theta = angles

        result = self.session.stackrt(n, d, f, theta)
        return result

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float64],
        angles: NDArray[np.float64],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the reflection and transmission coefficients.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float64]
        :param angles: _description_
        :type angles: NDArray[np.float64]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        data = self(index, thickness, frequencies, angles)
        return data


@frozen
class STACKFIELD(Engine):
    _allowed_kwargs: ClassVar[Iterable[str]] = ["resolution", "min", "max"]
    session: lumapi.FDTD

    def __call__(
        self,
        index: NDArray[np.float_],
        thickness: NDArray[np.float_],
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
        resolution: Optional[int] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> Dict[str, NDArray[np.float_]]:
        """
        Thin wrapper around lumapi.fdtd.stackfield which handles the optional rel, min, and max arguments.

        :param index: _description_
        :type index: NDArray[np.float_]
        :param thickness: _description_
        :type thickness: NDArray[np.float_]
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :param resolution: _description_, defaults to None
        :type resolution: Optional[int], optional
        :param min: _description_, defaults to None
        :type min: Optional[float], optional
        :param max: _description_, defaults to None
        :type max: Optional[float], optional
        :return: _description_
        :rtype: Dict[str, NDArray[np.float_]]
        """
        n = index
        d = thickness
        f = frequencies
        theta = angles
        args = []
        if resolution is not None:  # Why this bizarre nested structure?
            args.append(resolution)  # Because stackfield only takes positional args
            if (
                min is not None
            ):  # TODO Reimplement this such that max can be set independently of res and min
                args.append(min)
                if max is not None:
                    args.append(max)

        result = self.session.stackfield(n, d, f, theta, *args)
        return result

    @classmethod
    def filter_kwargs(cls, **kwargs):
        filtered_kwargs = filter_mapping(kwargs, ["resolution", "min", "max"])
        return filtered_kwargs

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float64],
        angles: NDArray[np.float64],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the field profile between min and max with the given resolution.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float64]
        :param angles: _description_
        :type angles: NDArray[np.float64]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        filtered_kwargs = self.filter_kwargs(**kwargs)

        data = self(index, thickness, frequencies, angles, **filtered_kwargs)
        return data


@frozen
class LumericalSTACK(Engine):
    session: lumapi.FDTD
    stackrt: STACKRT = Factory(lambda self: STACKRT(self.session), takes_self=True)
    stackfield: STACKFIELD = Factory(
        lambda self: STACKFIELD(self.session), takes_self=True
    )

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the output of STACKRT.simulate() and STACKFIELD.simulate() in a tuple.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        filtered_kwargs_for_stackfield = self.stackfield.filter_kwargs(**kwargs)
        rt_data = self.stackrt(index, thickness, frequencies, angles)
        field_data = self.stackfield(
            index, thickness, frequencies, angles, **filtered_kwargs_for_stackfield
        )

        return rt_data, field_data
