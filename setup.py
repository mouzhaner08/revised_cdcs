# setup.py
from setuptools import setup, find_packages

setup(
    name='revised_cdcs',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
    ],
    author='Zhaner Mou',
    description='Python implementation of confidence sets for causal orderings',
    include_package_data=True,
)