import pytest
from pytest_cases import fixture, parametrize, parametrize_with_cases
import numpy as np
import copy

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
def vacuum_layer():
    return Layer()


@fixture
def layer_1(material_1, thickness_1):
    return Layer.from_material(material_1, thickness_1)


@fixture
def layer_2(material_2, thickness_2):
    return Layer.from_material(material_2, thickness_2)


@fixture
def layers(layer_1, layer_2):
    return [layer_1, layer_2]


class TestMultilayer:

    # @pytest.mark.repeat(1)
    @parametrize("frequencies", [np.linspace(100, 200)])
    def test_index_and_thickness(self, layers, frequencies):
        layer_1, layer_2 = layers
        multilayer = Multilayer(layers)
        assert np.array_equal(
            multilayer.index(frequencies)[0], layer_1.index(frequencies)
        )
        assert np.array_equal(
            multilayer.index(frequencies)[1], layer_2.index(frequencies)
        )
        assert multilayer.thickness[0] == layer_1.thickness
        assert multilayer.thickness[1] == layer_2.thickness

    @parametrize(num_periods=[3, 10, 100])
    def test_from_given_unit_cell_no_copy(
        self, vacuum_layer, layers, material_1, num_periods
    ):
        multilayer = Multilayer.from_given_unit_cell(
            layers,
            incident_layer=vacuum_layer,
            exit_layer=vacuum_layer,
            num_periods=num_periods,
            copy_layers=False,
        )

        # Check the layers have been constructed correctly
        assert len(multilayer.layers) == len(layers) * num_periods + 2
        assert multilayer.layers[0] == multilayer.layers[-1]
        assert multilayer.layers[1] == multilayer.layers[1 + len(layers)]
        assert multilayer.unit_cell == layers

        # Check the layer objects are reused
        modified_layer = multilayer.layers[1]
        old_layer = copy.deepcopy(modified_layer)
        assert old_layer == modified_layer
        modified_layer.thickness += 1
        assert not old_layer == modified_layer
        assert modified_layer.thickness == multilayer.layers[1 + len(layers)].thickness
        material_1._index += 1
        assert modified_layer.index == multilayer.layers[1 + len(layers)].index
        assert modified_layer.index(1) == old_layer.index(1) + 1

    @parametrize(num_periods=[3, 10, 100])
    def test_from_given_unit_cell_copy(
        self, vacuum_layer, layers, material_1, num_periods
    ):
        multilayer = Multilayer.from_given_unit_cell(
            layers,
            incident_layer=vacuum_layer,
            exit_layer=vacuum_layer,
            num_periods=num_periods,
            copy_layers=True,
        )

        # Check the layers have been constructed correctly
        assert len(multilayer.layers) == len(layers) * num_periods + 2
        assert multilayer.layers[0] == multilayer.layers[-1]
        assert multilayer.layers[1] == multilayer.layers[1 + len(layers)]
        assert multilayer.unit_cell == layers

        # Check the layer objects are independent copies
        modified_layer = multilayer.layers[1]
        old_layer = layers[0]
        assert old_layer == modified_layer
        modified_layer.thickness += 1
        assert not old_layer == modified_layer
        assert (
            modified_layer.thickness == multilayer.layers[1 + len(layers)].thickness + 1
        )
        material_1._index += 1  # This should affect old_layer but not multilayer.layers
        assert modified_layer.index(1) == old_layer.index(1) - 1
        modified_layer.material._index += 1
        assert modified_layer.index(1) == old_layer.index(1)

    @parametrize(num_periods=[3, 10, 100])
    def test_stack_layers(self, vacuum_layer, layers, num_periods):
        multilayer = Multilayer.from_given_unit_cell(
            layers,
            incident_layer=vacuum_layer,
            exit_layer=vacuum_layer,
            num_periods=num_periods,
            copy_layers=True,
        )
        assert multilayer.stack_layers == layers * num_periods

    @parametrize(num_periods=[2])
    def test_from_own_unit_cell_default(self, vacuum_layer, layers, num_periods):
        multilayer = Multilayer([vacuum_layer] + layers + [vacuum_layer])
        assert multilayer.unit_cell is None
        derived_multilayer = multilayer.from_own_unit_cell()
        assert derived_multilayer == multilayer  # unit_cell is not checked for equality
        assert derived_multilayer.unit_cell == multilayer.stack_layers == layers
        another_multilayer = derived_multilayer.from_own_unit_cell(
            num_periods=num_periods
        )
        assert another_multilayer.unit_cell == derived_multilayer.unit_cell
        assert (
            another_multilayer.stack_layers
            == another_multilayer.unit_cell * num_periods
        )
