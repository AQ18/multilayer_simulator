from typing import Literal
from attrs import mutable


@mutable
class DataFormatter:  # consider making this abstract
    """
    Format the output from a given engine in some consistent way.
    """

    OutputFormats = Literal[
        None, "xarray_dataset", "xarray_dataarray"
    ]  # allowed formats when passed as a parameter

    output_format: OutputFormats = None

    def format(self, data, **kwargs):
        raise NotImplementedError
