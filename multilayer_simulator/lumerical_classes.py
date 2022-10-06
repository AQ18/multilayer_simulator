from typing import ClassVar, Dict
import numpy as np
from numpy.typing import NDArray
from attrs import frozen
import lumapi

from engine import Engine
from structure import Structure


@frozen
class LumericalSTACK(Engine):
    _properties_mapping_for_stackfield: ClassVar[Dict[str, str]] = {
        'field_resolution': 'res',
        'field_min': 'min',
        'field_max': 'max'
    }
    
    session: lumapi.FDTD
    field_resolution: int = 1000
    field_min: float = 0
    field_max: float = field()
    
    @field_max.default
    
    def stackrt(self, index: NDArray[np.float_], thickness: NDArray[np.float_], frequencies: NDArray[np.float_], angles: NDArray[np.float_]) -> Dict[str, NDArray[np.float_]]:
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
        
        result = self.session.stackrt(n=index, d=thickness, f=frequencies, t=angles)
        return result
    
    @classmethod
    def _parse_kwargs_for_stackfield(cls, **kwargs):
        allowed_
    
    def stackfield(self, index: NDArray[np.float_], thickness: NDArray[np.float_], frequencies: NDArray[np.float_], angles: NDArray[np.float_], **kwargs
                   ) -> Dict[str, NDArray[np.float_]]:
    
    def simulate(self, structure: Structure, frequencies: NDArray[np.float_], angles: NDArray[np.float_], **kwargs):
        