====================
multilayer-simulator
====================

Set up, run and post-process data from multilayer simulations with a variety of backends.

This project was first conceived to build a Python interface to conduct simulations of multilayers using Lumerical, and that is still the main focus.
A design goal is to introduce sufficient flexibility so that other optical physics solvers can be swapped in as backends.
Specific targets to be kept in mind include the `PyLlama <https://github.com/VignoliniLab/PyLlama>`_ and `py_matrix <https://github.com/gevero/py_matrix>`_ projects.

For a showcase of the features, please see the `example notebook <example.ipynb>`.

Features
--------

* Define a workflow and API specification for the task of performing optical modeling of 1D structures.
* Integrate with Lumerical's Python API to leverage its material database and models, and STACK solver.

Installation
------------

To 'install':

Download either Anaconda or Miniconda: https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html


Download multilayer-simulator repository e.g. from Github with
::
    git clone git@github.com:AQ18/multilayer-simulator.git

From Anaconda Prompt working in the multilayer-simulator repository, run this to create the conda environment:
::
    conda env create --file environment.yml --name multilayer-simulator

To start using the notebooks and edit the code, run
::
    conda activate multilayer-simulator
    jupyter lab

If using Lumerical STACK or FDTD as a backend
.............................................

For reasons unknown, Lumerical's Python API is not presented as a package, and so pip install doesn't work. Instead, add the directory to the system path, either permanently via system settings, or specifically for this environment using
::
    conda develop "C:\Program Files\Lumerical\v212\api\python" -n multilayer_simulator

(Replace the path with whatever is appropriate for your installation, of course. The version number will probably not be the same.)
You can check that this has been successful by running the very simple ``test_lumapi_installation.py`` script. Look at the comments in the script for advice on what to do if it breaks.



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
