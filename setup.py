from setuptools import setup, find_packages

setup(
    name="tfexp",
    version="0.1",
    description=(
        "KISS framework to conduct tensorflow experiments,"
        "described in YAML files."
    ),
    author="Marcel Arpogaus",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml",
        "tensorflow"
    ],
    entry_points=dict(console_scripts=["tfexp = tfexp.cli:cli"]),
)
