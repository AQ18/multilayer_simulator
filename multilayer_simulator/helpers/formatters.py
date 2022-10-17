

class DataFormatter:
    """
    Format the output from a given engine in some consistent way.
    """
    OutputFormats = Literal[None, 'xarray_dataset', 'xarray_dataarray'] # allowed formats when passed as a parameter
    pass