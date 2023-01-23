from typing import Hashable, Iterable, Union
import xarray as xr
import numpy as np


def vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def add_vector_norms_to_xarray_dataset(dataset: xr.Dataset, dim):
    """
    Modify a dataset in-place by adding the norm of each dataset with respect to dim as a new data variable.

    :param dataset: _description_
    :type dataset: xr.Dataset
    :param dim: _description_
    :type dim: _type_
    """
    for variable in dataset:
        dataset[f"|{variable}|^2"] = vector_norm(dataset[variable], dim)


def add_absorption_to_xarray_dataset(dataset: xr.Dataset, reflectance_key: Union[Iterable, Hashable], transmittance_key: Union[Iterable, Hashable], absorptance_key: Union[Iterable, Hashable]):
    """
    Modify a dataset by calculating the absorptance from the DataArrays indicated by the reflectance and transmittance keys, and
    adding it to the dataset under the absorptance key. Returns the modified dataset as a copy.
    Fails silently if either the reflectance or transmittance keys are missing.
    TODO: Raise warning instead.

    :param dataset: An xarray dataset
    :type dataset: xr.Dataset
    :param reflectance_key: Key or sequence of keys of reflectance variable.
    :type reflectance_key: Union[Iterable, Hashable]
    :param transmittance_key: Key or sequence of keys of transmittance variable.
    :type transmittance_key: Union[Iterable, Hashable]
    :param absorptance_key: Key or sequence of keys of absorptance variable.
    :type absorptance_key: Union[Iterable, Hashable]
    """
    r_keys = np.atleast_1d(reflectance_key)
    t_keys = np.atleast_1d(transmittance_key)
    a_keys = np.atleast_1d(absorptance_key)
    
    variables = {a: lambda x: 1 - x[r] - x[t] for r, t, a in zip(r_keys, t_keys, a_keys)}
    
    modified = dataset.assign(variables=variables)
    
    return modified