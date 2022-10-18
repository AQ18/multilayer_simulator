from typing import Hashable
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


def add_absorption_to_xarray_dataset(dataset: xr.Dataset, reflectance_key: Hashable, transmittance_key: Hashable, absorptance_key: Hashable):
    """
    Modify a dataset in-place by calculating the absorptance from the DataArrays indicated by the reflectance and transmittance keys, and
    adding it to the dataset under the absorptance key.
    Fails silently if either the reflectance or transmittance keys are missing.
    TODO: Raise warning instead.

    :param dataset: _description_
    :type dataset: xr.Dataset
    :param reflectance_key: _description_
    :type reflectance_key: Hashable
    :param transmittance_key: _description_
    :type transmittance_key: Hashable
    :param absorptance_key: _description_
    :type absorptance_key: Hashable
    """
    try:
        reflectance = dataset[reflectance_key]
        transmittance = dataset[transmittance_key]
    except KeyError:
        pass
    else:
        dataset[absorptance_key] = 1 - reflectance - transmittance
    