from setuptools import setup

setup(
    name="celltosg",
    version="0.1.0",
    py_modules=["train", "dataset", "utils", "pretrain", "train-gnn", "data_loader"],
    python_requires=">=3.9",
)
