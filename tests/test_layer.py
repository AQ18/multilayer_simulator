import pytest
from pytest_cases import fixture, parametrize, parametrize_with_cases
import random
import dis
import copy
import attrs
from multilayer_simulator.structure import Layer
from multilayer_simulator.material import ConstantIndex


@fixture
def default_index():
    return ConstantIndex(1).index


@fixture
def default_thickness():
    return 0


@fixture
def constant_index_material():
    return ConstantIndex()


class IndexCases:
    def case_default(self, default_index):
        """
        Assumed default index for Layers.
        'None' must be interpreted as 'default' in test function.
        """
        return None, default_index


class ThicknessCases:
    def case_default(self):
        """
        Assumed default thickness for Layers.
        'None' must be interpreted as 'default' in test function.
        """
        return None, 0

    def random_positive(self):
        number = random.random()
        return number, number

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def random_negative(self):
        number = -random.random()
        return number, number


class NameCases:
    def case_default(self):
        return None

    def case_string(self):
        return "Bob"

    def case_integer(self):
        return 2


class FuncCases:
    @parametrize("out", ["None", "x", "str(x) + ' Layer'"])
    def name_callables_passing(self, out: str):
        return lambda x: eval(out)

    @pytest.mark.xfail(strict=True)
    @parametrize("func", [(lambda x, y: None), (lambda: None)])
    def name_callables_failing(self, func):
        return func


class TestLayer:
    @pytest.mark.repeat(1)
    def test_default_init(self, default_index, default_thickness):
        """Default values are as stipulated."""
        layer = Layer()
        assert dis.dis(layer.index) == dis.dis(default_index)
        assert layer.thickness == default_thickness

    @parametrize_with_cases(
        "thickness, expected_thickness", cases=ThicknessCases, prefix="random"
    )
    def test_init(self, thickness, expected_thickness):
        """Positive thicknesses are allowed and negative ones should raise."""
        layer = Layer(thickness=thickness)
        # Check the bytecode is the same
        assert layer.thickness == expected_thickness

    @parametrize_with_cases(
        "new_thickness, expected_thickness", cases=ThicknessCases, prefix="random"
    )
    def test_thickness_modification(self, new_thickness, expected_thickness):
        """Thickness can be modified but negative values raise."""
        layer = Layer()
        layer.thickness = new_thickness
        assert layer.thickness == expected_thickness

    def test_default_from_material(self):
        """.from_material() alternate constructor has same defaults as init()."""
        layer1 = Layer()
        layer2 = Layer.from_material()
        assert layer1 == layer2

    @parametrize_with_cases("name", cases=NameCases)
    def test_from_material_default_name(self, constant_index_material, name):
        """.from_material() should copy material name by default."""
        constant_index_material.name = name
        layer = Layer.from_material(constant_index_material)
        assert layer.name == name

    @parametrize_with_cases("name", cases=NameCases)
    @parametrize_with_cases("func", cases=FuncCases, prefix="name")
    def test_from_material_callable_name(self, constant_index_material, name, func):
        """Passing callable as name to .from_material() should call with material name."""
        constant_index_material.name = name
        layer = Layer.from_material(constant_index_material, name=func)
        assert layer.name == func(name)

    @parametrize_with_cases("name", cases=NameCases)
    def test_from_material_callable_name(self, constant_index_material, name):
        """Passing callable as name to .from_material() should call with material name."""
        layer = Layer.from_material(constant_index_material, name=name)
        assert layer.name == name

    @parametrize("frequency, new_index", [(1, 2)])
    def test_material_modification_index(
        self, constant_index_material, frequency, new_index
    ):
        """Modifying material index property should modify layer index property."""
        layer = Layer.from_material(constant_index_material)
        constant_index_material._index = new_index
        assert layer.index(frequency) == new_index

    @parametrize("frequency, new_index", [(1, 2)])
    def test_copied_material_modification_index(
        self, constant_index_material, frequency, new_index
    ):
        """Copied layer.material property should give a handle on layer.index()."""
        layer = copy.deepcopy(Layer.from_material(constant_index_material))
        assert layer.material == constant_index_material
        old_index = constant_index_material._index
        constant_index_material._index = new_index
        assert layer.index(frequency) == old_index == 1
        # This hard-coded number probably doesn't belong in this test
        layer.material._index = new_index
        assert layer.index(frequency) == new_index

    @parametrize_with_cases("new_name", cases=NameCases)
    def test_material_modification_name(
        self, constant_index_material, new_name
    ):
        """Modifying material name property will not modify layer name property."""
        layer = Layer.from_material(constant_index_material)
        constant_index_material.name = new_name
        # Check that the layer name is still the default value
        assert layer.name == attrs.fields(type(constant_index_material)).name.default