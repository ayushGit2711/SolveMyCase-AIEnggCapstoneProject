from setuptools import setup, find_packages

_subpackages = [f"solvemycase.{pkg}" for pkg in find_packages()]

setup(
    name="solvemycase",
    version="0.1.0",
    package_dir={"solvemycase": "."},
    packages=["solvemycase"] + _subpackages,
)
