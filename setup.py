from setuptools import setup, find_packages

setup(
    name='WireModeling',
    version='0.1.0',
    description='Modelisation of wires',
    packages=find_packages(),
    install_requires=[
        'numpy','scikit-learn','pandas','matplotlib','scipy'
    ],
)