

from typing import Iterable, Mapping, Any


def filter_mapping(mapping=Mapping[Any, Any], filter: Iterable[Any]) -> Dict[Any, Any]:
    """
    Return a dictionary with only those key-value pairs from mapping where the key is in filter.
    Faster if filter is shorter than mapping.

    :param filter: _description_
    :type filter: Iterable[Any]
    :param mapping: _description_, defaults to Mapping[Any, Any]
    :type mapping: _type_, optional
    :return: _description_
    :rtype: Dict[Any, Any]
    """
    filtered_mapping = {k: mapping[k] for k in filter if k in mapping.keys()}
    return filtered_mapping