import pytest
from pytest_cases import fixture, parametrize, parametrize_with_cases

# import random
from multilayer_simulator.structure import Layer, Multilayer
from multilayer_simulator.material import ConstantIndex


@fixture
def material_1():
    return ConstantIndex(1)


@fixture
def material_2():
    return ConstantIndex(2)


@fixture
def thickness_1():
    return 1


@fixture
def thickness_2():
    return 2


@fixture
def layer_1(material_1, thickness_1):
    return Layer.from_material(material_1, thickness_1)


@fixture
def layer_2(material_2, thickness_2):
    return Layer.from_material(material_2, thickness_2)


class TestMultilayer:
    def test_index_and_thickness():
        pass
