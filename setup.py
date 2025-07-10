from setuptools import setup, find_packages

setup(
    name='revised_cdcs',
    version='0.1',
    description='A Python implementation of confidence sets for causal orderings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Zhaner Mou',
    url='https://github.com/mouzhaner08/revised_cdcs',
    packages=find_packages(include=['revised_cdcs', 'revised_cdcs.*']),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)