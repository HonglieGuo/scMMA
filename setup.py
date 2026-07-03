from setuptools import setup, find_packages

setup(
    name="scmma",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "lightning",
        "hydra-core",
        "scanpy",
        "muon",
        "anndata",
        "transformers",
        "peft"
    ],
)
