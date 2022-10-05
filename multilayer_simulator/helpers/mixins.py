from functools import cached_property
import numpy as np
from numpy.typing import ArrayLike
from attrs import mutable, field, converters, setters


c = 2.99792458e8

def convert_wavelength_and_frequency(value, c=c):
    return c/value

def set_wavelengths(instance, attrib, new_value):
    instance.wavelengths = convert_wavelength_and_frequency(new_value, instance.c)
    return new_value

def set_frequencies(instance, attrib, new_value):
    instance.frequencies = convert_wavelength_and_frequency(new_value, instance.c)
    return new_value

def unset_wavelengths(instance, attrib, new_value):
    try:
        del instance.wavelengths
    except: AttributeError
    
    return new_value

@mutable
class SpectrumMixinV0_1:
    """
    Implements a 'frequencies' and 'wavelengths' interface.
    
    This version attempts to give frequencies and wavelengths equal footing, using setters to update one property when the other is set.
    FIXME: However, it doesn't work because an infinite loop is created.
    """
    
    c: float = c
    frequencies: ArrayLike = field(default=None, kw_only=True, converter=converters.optional(np.atleast_1d), on_setattr=[setters.convert, set_wavelengths])
    wavelengths: ArrayLike = field(default=None, kw_only=True, converter=converters.optional(np.atleast_1d), on_setattr=[setters.convert, set_frequencies])


@mutable(slots=False)
class SpectrumMixinV0_2:
    """
    Implements a 'frequencies' and 'wavelengths' interface.
    
    This version gives frequencies primacy and makes wavelengths a derived property with caching.
    FIXME: wavelengths is no longer in the repr
    FIXME: can set wavelengths directly but frequencies is not updated
    """
    
    c: float = field(default=c, repr=False, kw_only=True)
    frequencies: ArrayLike = field(default=None, kw_only=True, converter=converters.optional(np.atleast_1d), on_setattr=[setters.convert, unset_wavelengths])
    # wavelengths: ArrayLike = field(init=False)
    
    @cached_property
    def wavelengths(self):
        try:
            return convert_wavelength_and_frequency(self.frequencies, self.c)
        except TypeError:
            return None
