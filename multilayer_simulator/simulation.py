import numpy as np
from numpy.typing import ArrayLike
from attrs import mutable, field, setters

from multilayer_simulator.engine import Engine
from multilayer_simulator.helpers.formatters import DataFormatter
from multilayer_simulator.helpers.mixins import SpectrumMixinV0_2
from multilayer_simulator.structure import Structure


def simulate(structure, engine, frequencies, angles, **kwargs):
    data = engine.simulate(structure, frequencies, angles, **kwargs)
    return data


@mutable
class Simulation(SpectrumMixinV0_2):
    """
    Manages a simulation of light propagating through a given Structure at a given frequency/wavelength and angle.
    The simulation is powered by a given Engine.

    Outputs data in some TBD format than can be post-processed (e.g. compared with other simulations) and visualised.
    """

    structure: Structure = field(default=None)
    engine: Engine = field(default=None)
    angles: ArrayLike = field(
        factory=lambda: [0],
        kw_only=True,
        converter=np.atleast_1d,
        on_setattr=setters.convert,
    )
    formatter: DataFormatter = field(default=None)
    data = field(default=None, init=False)

    def simulate(
        self,
        structure=None,
        engine=None,
        frequencies=None,
        angles=None,
        formatter=None,
        keep_data=True,
        **kwargs
    ):
        structure = self.structure if structure is None else structure
        engine = self.engine if engine is None else engine
        frequencies = self.frequencies if frequencies is None else frequencies
        angles = self.angles if angles is None else angles
        data = simulate(structure, engine, frequencies, angles, **kwargs)

        formatter = self.formatter if formatter is None else formatter
        if formatter is not None:
            data = formatter.format(data)  # maybe implement filtered kwargs here

        if keep_data is True:
            self.data = data

        return data
