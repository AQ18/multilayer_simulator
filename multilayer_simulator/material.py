from typing import Literal, Optional
from attrs import mutable
import numpy as np
from numpy.typing import NDArray


class Material:
    """
    Represent a material.
    """

    def index(
        self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1
    ) -> NDArray[np.float_]:
        """
        Return the complex refractive index at frequency f, which may be a float or an array of floats.

        For anisotropic materials the component is 1, 2, or 3. (Not currently supported.)
        """
        pass


@mutable
class ConstantIndex(Material):
    _index: float = 1

    def index(
        self,
        frequencies: Optional[NDArray[np.float_]] = None,
        component: Literal[1, 2, 3] = 1,
    ) -> NDArray[np.float_]:
        frequencies = np.atleast_1d(frequencies)
        return np.full(shape=frequencies.shape, fill_value=self._index)
