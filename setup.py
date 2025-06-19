from setuptools import setup, find_packages

setup(
    name="celltosg_loader",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "torch"],
    python_requires=">=3.9",
)
