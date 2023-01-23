from typing import Hashable, Iterable, Union
import xarray as xr
import numpy as np

DatasetKeys = Union[Iterable[Hashable], Hashable]


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


def add_absorption_to_xarray_dataset(
    dataset: xr.Dataset,
    reflectance_key: DatasetKeys,
    transmittance_key: DatasetKeys,
    absorptance_key: DatasetKeys,
) -> xr.Dataset:
    """
    Modify a dataset by calculating the absorptance from the DataArrays indicated by the reflectance and transmittance keys, and
    adding it to the dataset under the absorptance key. Returns the modified dataset as a copy.
    TODO: Raise warning if either the reflectance or transmittance keys are missing.

    :param dataset: An xarray dataset
    :type dataset: xr.Dataset
    :param reflectance_key: Key or sequence of keys of reflectance variable.
    :type reflectance_key: DatasetKeys
    :param transmittance_key: Key or sequence of keys of transmittance variable.
    :type transmittance_key: DatasetKeys
    :param absorptance_key: Key or sequence of keys of absorptance variable.
    :type absorptance_key: DatasetKeys
    """
    r_keys = np.atleast_1d(reflectance_key)
    t_keys = np.atleast_1d(transmittance_key)
    a_keys = np.atleast_1d(absorptance_key)

    variables = {
        a: lambda x: 1 - x[r] - x[t] for r, t, a in zip(r_keys, t_keys, a_keys)
    }

    modified = dataset.assign(variables=variables)

    return modified


def add_unpolarised_to_xarray_dataset(
    dataset: xr.Dataset,
    s_key: DatasetKeys,
    p_key: DatasetKeys,
    unpolarised_key: DatasetKeys,
) -> xr.Dataset:
    """
    Modify a dataset by calculating the unpolarised variable from the DataArrays indicated by the s- and p-polarisation keys,
    and adding it to the dataset under the unpolarised key. Return the modified dataset as a copy.
    TODO: Raise warning if either the reflectance or transmittance keys are missing.

    :param dataset: An xarray dataset
    :type dataset: xr.Dataset
    :param s_key: Key or sequence of keys of s_polarised variable.
    :type s_key: DatasetKeys
    :param p_key: Key or sequence of keys of p_polarised variable.
    :type p_key: DatasetKeys
    :param unpolarised_key: Key or sequence of keys of unpolarised variable.
    :type unpolarised_key: DatasetKeys
    :return: Modified dataset
    :rtype: xr.Dataset
    """
    s_keys = np.atleast_1d(s_key)
    p_keys = np.atleast_1d(p_key)
    u_keys = np.atleast_1d(unpolarised_key)

    variables = {
        u: lambda x: (x[s] + s[p]) / 2 for s, p, u in zip(s_keys, p_keys, u_keys)
    }

    modified = dataset.assign(variables=variables)

    return modified
