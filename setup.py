"""
Setup configuration for Generic Flexibility Model.

This package provides a framework for modeling and optimizing flexibility assets
in energy systems from a utility company perspective.
"""

from setuptools import setup, find_packages

# Read long description from README
try:
    with open("docs/README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="flex_model",
    version="0.1.0",
    author="Mathias Niffeler",
    author_email="mathias.niffeler@empa.ch",
    description="Framework for modeling and optimizing flexibility assets in energy systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatNif/GenericFlexiblityModel",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "visualization": [
            "plotly>=5.14.0",
            "pandas>=2.0.0",
        ],
        "dashboard": [
            "plotly>=5.14.0",
            "pandas>=2.0.0",
            "streamlit>=1.28.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "all": [
            "plotly>=5.14.0",
            "pandas>=2.0.0",
            "streamlit>=1.28.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
