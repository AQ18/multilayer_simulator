from typing import ClassVar, Dict, Iterable, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from attrs import mutable, frozen, field, Factory
import lumapi

from multilayer_simulator.engine import Engine
from multilayer_simulator.helpers.helpers import filter_mapping, relabel_mapping
from multilayer_simulator.structure import Structure
from multilayer_simulator.material import Material


@frozen
class STACKRT(Engine):
    session: lumapi.FDTD

    def __call__(
        self,
        index: NDArray[np.float_],
        thickness: NDArray[np.float_],
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
    ) -> Dict[str, NDArray[np.float_]]:
        """
        Thin wrapper around lumapi.FDTD.stackrt.

        :param index: _description_
        :type index: NDArray[np.float_]
        :param thickness: _description_
        :type thickness: NDArray[np.float_]
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :return: _description_
        :rtype: Dict[str, NDArray[np.float_]]
        """
        n = index
        d = thickness
        f = frequencies
        theta = angles

        result = self.session.stackrt(n, d, f, theta)
        return result

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float64],
        angles: NDArray[np.float64],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the reflection and transmission coefficients.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float64]
        :param angles: _description_
        :type angles: NDArray[np.float64]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        data = self(index, thickness, frequencies, angles)
        return data


@frozen
class STACKFIELD(Engine):
    _allowed_kwargs: ClassVar[Iterable[str]] = ["resolution", "min", "max"]
    session: lumapi.FDTD

    def __call__(
        self,
        index: NDArray[np.float_],
        thickness: NDArray[np.float_],
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
        resolution: Optional[int] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> Dict[str, NDArray[np.float_]]:
        """
        Thin wrapper around lumapi.fdtd.stackfield which handles the optional rel, min, and max arguments.

        :param index: _description_
        :type index: NDArray[np.float_]
        :param thickness: _description_
        :type thickness: NDArray[np.float_]
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :param resolution: _description_, defaults to None
        :type resolution: Optional[int], optional
        :param min: _description_, defaults to None
        :type min: Optional[float], optional
        :param max: _description_, defaults to None
        :type max: Optional[float], optional
        :return: _description_
        :rtype: Dict[str, NDArray[np.float_]]
        """
        n = index
        d = thickness
        f = frequencies
        theta = angles
        args = []
        if resolution is not None:  # Why this bizarre nested structure?
            args.append(resolution)  # Because stackfield only takes positional args
            if (
                min is not None
            ):  # TODO Reimplement this such that max can be set independently of res and min
                args.append(min)
                if max is not None:
                    args.append(max)

        result = self.session.stackfield(n, d, f, theta, *args)
        return result

    @classmethod
    def filter_kwargs(cls, **kwargs):
        filtered_kwargs = filter_mapping(kwargs, ["resolution", "min", "max"])
        return filtered_kwargs

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float64],
        angles: NDArray[np.float64],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the field profile between min and max with the given resolution.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float64]
        :param angles: _description_
        :type angles: NDArray[np.float64]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        filtered_kwargs = self.filter_kwargs(**kwargs)

        data = self(index, thickness, frequencies, angles, **filtered_kwargs)
        return data


@frozen
class LumericalSTACK(Engine):
    session: lumapi.FDTD
    stackrt: STACKRT = Factory(lambda self: STACKRT(self.session), takes_self=True)
    stackfield: STACKFIELD = Factory(
        lambda self: STACKFIELD(self.session), takes_self=True
    )

    def simulate(
        self,
        structure: Structure,
        frequencies: NDArray[np.float_],
        angles: NDArray[np.float_],
        **kwargs
    ):
        """
        Simulate the propagation of light through the structure in one dimension using Lumerical's STACK solver.
        Returns the output of STACKRT.simulate() and STACKFIELD.simulate() in a tuple.

        :param structure: _description_
        :type structure: Structure
        :param frequencies: _description_
        :type frequencies: NDArray[np.float_]
        :param angles: _description_
        :type angles: NDArray[np.float_]
        :return: _description_
        :rtype: _type_
        """
        index = structure.index(frequencies)
        thickness = structure.thickness
        filtered_kwargs_for_stackfield = self.stackfield.filter_kwargs(**kwargs)
        rt_data = self.stackrt(index, thickness, frequencies, angles)
        field_data = self.stackfield(
            index, thickness, frequencies, angles, **filtered_kwargs_for_stackfield
        )

        return rt_data, field_data


class LumericalMaterial(Material):
    """
    Represent and control a material in the Lumerical materials database.
    """

    _properties_mapping = {
        "name": "name",
        "mesh_order": "mesh order",
        "color": "color",
        "anisotropy": "anisotropy",
        "type": "type",
    }

    @classmethod
    def _make_property_struct(
        cls, key_map: Mapping[str, str] = _properties_mapping, **kwargs
    ):
        # struct = {key_map[key]: value for key, value in kwargs.items()}
        struct = relabel_mapping(kwargs, key_map)
        return struct

    def __init__(
        self,
        session,
        material_type: str,
        name=None,
        properties_mapping=_properties_mapping,
        **kwargs
    ):
        """
        A class representing a new material added to the Lumerical materials database for a session.
        Probably can be trivially extended to include existing materials too.
        Not tested for any engine but FDTD because I don't use them.

        Property getters use private member variables rather than API calls for a very minor performance advantage. Check that there is agreement using check_properties() method.

        session: an instance of a Lumerical product session
        material_type: str returned by print(session.addmaterial())

        Examples
        --------

        fdtd = lumapi.FDTD()
        oscillator = Material(fdtd, 'Lorentz')

        """

        self.session = session
        self._name = session.addmaterial(material_type)
        if name:
            self.name = name
        self._properties_mapping = dict(properties_mapping)
        self.set_property(**kwargs)
        self.sync_backwards()

    # def _map_kwargs_to_properties(self):
    #     mapping = {'name': 'name',
    #                'mesh_order': 'mesh order',
    #                'color': 'color',
    #                'anisotropy': 'anisotropy',
    #                'type': 'type'
    #               }
    #     return mapping

    def get_property(self, prop=None):
        """
        General getter method for material properties. Is it confusing that it returns different types depending on input? Probably, but that does reflect how the API call works.
        TODO: Maybe I should figure out how to get it to return a dict every time. For now, if prop is a list (even with a single entry), returns a dict.

        prop: None or str or List[str]
        """
        if prop == "all":
            return self.get_property(
                self.get_property()
            )  # being too clever for my own good
        elif prop:
            return self.session.getmaterial(self.name, prop)
        else:
            return self.session.getmaterial(self.name).split("\n")

    def set_property(self, prop=None, value=None, **kwargs):
        """
        General setter method for material properties. Only sets the properties on the Lumerical material, not the Material instance - call sync_backwards() to update instance variables.

        WARNING: if you use this to set the name, it will not automatically update self._name, which will break all future API calls until self._name is corrected.

        prop: None or str or Dict[str, value]
        value: whatever appropriate type the value for prop should be
        """
        if prop and value:
            self.session.setmaterial(self.name, prop, value)
        elif prop:
            self.session.setmaterial(self.name, prop)
        elif kwargs:
            struct = self._make_property_struct(
                mapping=self._properties_mapping, **kwargs
            )
            self.session.setmaterial(self.name, struct)
        else:
            return self.session.setmaterial(self.name).split("\n")

        return

    def sync_backwards(self):
        properties = self.get_property("all")
        self._name = properties["name"]
        self._mesh_order = properties["mesh order"]
        self._color = properties["color"]
        self._anisotropy = properties["anisotropy"]
        self._type = properties["type"]
        return properties

    def delete(self):
        self.session.deletematerial(self.name)

    def index(
        self, frequencies: NDArray[np.float_], component: Literal[1, 2, 3] = 1
    ) -> NDArray[np.float_]:
        """
        Return the complex refractive index at frequency f, which may be a float or an array of floats.

        For anisotropic materials the component is 1, 2, or 3. (Not currently supported.)
        """
        return self.session.getindex(self.name, frequencies, component).reshape(
            -1
        )  # squeeze from nx1 to n array

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        """
        This setter is different to the others because once the new name is set, the old name cannot be used in session.getmaterial() to get the new name.
        """
        self.set_property("name", new_name)
        self._name = self.session.getmaterial(new_name, "name")

    @property
    def mesh_order(self):
        return self._mesh_order

    @mesh_order.setter
    def mesh_order(self, new_mesh_order):
        self.set_property("mesh order", new_mesh_order)
        self._mesh_order = self.get_property("mesh order")

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color):
        self.set_property("color", new_color)
        self._color = self.get_property("color")

    @property
    def anisotropy(self):
        return self._anisotropy

    @anisotropy.setter
    def anisotropy(self, new_anisotropy):
        self.set_property("anisotropy", new_anisotropy)
        self._anisotropy = self.get_property("anisotropy")

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        self.set_property("type", new_type)
        self._type = self.get_property("type")

    def check_properties(self):
        raise NotImplementedError()


class LumericalOscillator(LumericalMaterial):
    """
    Represent and control a Lorentz Oscillator type material in the Lumerical materials database.
    """

    _properties_mapping = LumericalMaterial._properties_mapping | {
        "refractive_index": "Refractive Index",
        "permittivity": "Permittivity",
        "lorentz_permittivity": "Lorentz Permittivity",
        "lorentz_resonance": "Lorentz Resonance",
        "lorentz_linewidth": "Lorentz Linewidth",
    }

    @classmethod
    def _make_property_struct(cls, mapping=_properties_mapping, **kwargs):
        return super()._make_property_struct(mapping, **kwargs)

    def __init__(
        self, session, name=None, properties_mapping=_properties_mapping, **kwargs
    ):
        super().__init__(
            session=session,
            material_type="Lorentz",
            name=name,
            properties_mapping=properties_mapping,
            **kwargs
        )

    def sync_backwards(self):
        properties = super().sync_backwards()
        self._refractive_index = properties["Refractive Index"]
        self._permittivity = properties["Permittivity"]
        self._lorentz_permittivity = properties["Lorentz Permittivity"]
        self._lorentz_resonance = properties["Lorentz Resonance"]
        self._lorentz_linewidth = properties["Lorentz Linewidth"]

    @property
    def refractive_index(self):
        """I think this returns the refractive index at infinity?"""
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, new_ri):
        self.set_property("Refractive Index", new_ri)
        self._refractive_index = self.get_property("Refractive Index")

    @property
    def permittivity(self):
        return self._permittivity

    @permittivity.setter
    def permittivity(self, new_permittivity):
        self.set_property("Permittivity", new_permittivity)
        self._permittivity = self.get_property("Permittivity")

    @property
    def lorentz_permittivty(self):
        return self._lorentz_permittivty

    @lorentz_permittivty.setter
    def lorentz_permittivty(self, new_lorentz_permittivty):
        self.set_property("Lorentz Permittivity", new_lorentz_permittivty)
        self._lorentz_permittivty = self.get_property("Lorentz Permittivity")

    @property
    def lorentz_resonance(self):
        return self._lorentz_resonance

    @lorentz_resonance.setter
    def lorentz_resonance(self, new_lorentz_resonance):
        self.set_property("Lorentz Resonance", new_lorentz_resonance)
        self._lorentz_resonance = self.get_property("Lorentz Resonance")

    @property
    def lorentz_linewidth(self):
        return self._lorentz_linewidth

    @lorentz_linewidth.setter
    def lorentz_linewidth(self, new_lorentz_linewidth):
        self.set_property("Lorentz Linewidth", new_lorentz_linewidth)
        self._lorentz_linewidth = self.get_property("Lorentz Linewidth")
