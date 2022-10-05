from typing import Literal




class Material:
    """
    Represent a material.
    """
    
    def index(self, f: float, component: Literal[1, 2, 3] = 1):
        """
        Return the complex refractive index at frequency f, which may be a float or an array of floats.
        
        For anisotropic materials the component is 1, 2, or 3. (Not currently supported.)
        """
        pass
    
class LumericalMaterial(Material):
    """
    Represent and control a material in the Lumerical materials database.
    """
    
    
class LumericalOscillator(LumericalMaterial):
    """
    Represent and control a Lorentz Oscillator type material in the Lumerical materials database.
    """
    
