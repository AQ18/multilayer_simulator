import numpy as np
from numpy.typing import ArrayLike
from attrs import mutable, field, setters

from engine import Engine
from helpers.mixins import SpectrumMixinV0_2
from structure import Structure


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
        factory=lambda: [0], kw_only=True, converter=np.atleast_1d, on_setattr=setters.convert
    )
    data = field(init=False)

    def simulate(
        self,
        structure=None,
        engine=None,
        frequencies=None,
        angles=None,
        save_data=True,
        **kwargs
    ):
        if structure is None:
            structure = self.structure
        if engine is None:
            engine = self.engine
        if frequencies is None:
            frequencies = self.frequencies
        if angles is None:
            angles = self.angles
        data = simulate(structure, engine, frequencies, angles, **kwargs)

        if save_data is True:
            self.data = data

        return data
