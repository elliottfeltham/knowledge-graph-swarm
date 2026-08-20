from setuptools import find_packages, setup


setup(
    name="knowledge-graph-swarm",
    version="0.1.0",
    py_modules=["common"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
