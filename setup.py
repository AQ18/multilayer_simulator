#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Alex Qiu",
    author_email='alex.qiu@bristol.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Set up, run and post-process data from multilayer simulations with a variety of backends.",
    entry_points={
        'console_scripts': [
            'multilayer_simulator=multilayer_simulator.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='multilayer_simulator',
    name='multilayer_simulator',
    packages=find_packages(include=['multilayer_simulator', 'multilayer_simulator.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AQ18/multilayer_simulator',
    version='0.1.0',
    zip_safe=False,
)
