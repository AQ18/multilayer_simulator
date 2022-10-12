from typing import Iterable, Mapping, Any


def filter_mapping(mapping: Mapping[Any, Any], filter: Iterable[Any]) -> dict[Any, Any]:
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


def relabel_mapping(
    mapping: Mapping[Any, Any], key_map: Mapping[Any, Any]
) -> Mapping[Any, Any]:
    """
    Return mapping with the keys relabeled according to key_map.

    :param mapping: _description_
    :type mapping: Mapping[Any, Any]
    :param key_map: Must contain all the keys in mapping.
    :type key_map: Mapping[Any, Any]
    :return: Returns the same type as mapping.
    :rtype: Mapping[Any, Any]
    """
    relabeled_generator = ((key_map[key], value) for key, value in mapping.items())
    relabeled_mapping = type(mapping)(relabeled_generator)
    return relabeled_mapping
