from setuptools import setup, find_packages
import os

# Read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (PyPI compatible)
install_requires = [
    "numpy>=1.19.0",
    "scipy>=1.6.0",
    "pandas>=1.2.0",
    "scikit-learn>=0.24.0",
    "tqdm>=4.50.0",
    "networkx>=3.0",
    "matplotlib>=3.0.0"
]

# Optional dependencies
extras_require = {
    "dev": [
        "ipykernel>=6.0",  # Notebook support
        "pytest>=7.0",     # Testing
        "black>=22.0"      # Formatting
    ],
    "docs": [
        "sphinx>=5.0",
        "myst-parser>=0.18"
    ]
}

setup(
    name="revised_cdcs",
    version="0.1.0",
    author="Zhaner Mou",
    author_email="zhmou@ucsd.edu",
    description="A Python implementation of confidence sets for causal orderings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mouzhaner08/revised_cdcs",
    packages=find_packages(include=["revised_cdcs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "cdcs=revised_cdcs.cli:main",
        ],
    },
)