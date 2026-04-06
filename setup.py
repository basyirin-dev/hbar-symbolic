"""Setup script for hbar-symbolic package."""

from setuptools import find_packages, setup

setup(
    name="hbar-symbolic",
    version="0.1.0",
    description="H-Bar Symbolic: A framework for studying compositional generalization through schema coherence dynamics",
    author="H-Bar Research",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.8.0",
        "optax>=0.1.7",
        "chex>=0.1.8",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.10",
)
