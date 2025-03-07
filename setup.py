from setuptools import setup, find_packages

setup(
    name="celltosg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "torch", "os"],
    python_requires=">=3.9",
)
