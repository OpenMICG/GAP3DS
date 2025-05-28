from setuptools import setup, find_packages

setup(
    name="GAP3DS",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "trimesh",
        "open3d"
    ]
) 