#!env python3
# -*- coding: utf-8 -*-

import os
import pytest

import yaml

from tfexp.utils import ExtendedLoader

test_yaml = """
a: !store [b, 2]
c: !include inc1/include_var.yaml
d: $e
f: !sum [$g, !sum [!include inc2/include_seq.yaml]]
h: !join [$i, ' ', world]
"""

include_var_yaml = """
e: !store [e, !product [$b, 2]]
g: !store [g, 2]
i: i<=hello
"""

include_seq_yaml = """
[3, 4, 5]
"""

expected_dict = {
    "a": 2,
    "c": {"e": 4, "g": 2, "i": "hello"},
    "d": 4,
    "f": 14,
    "h": "hello world",
}


@pytest.fixture(scope="session")
def artificial_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("yaml_data")
    with open(tmpdir / "test.yaml", "w+") as f:
        f.write(test_yaml)
    os.mkdir(tmpdir / "inc1")
    os.mkdir(tmpdir / "inc2")
    with open(tmpdir / "inc1" / "include_var.yaml", "w+") as f:
        f.write(include_var_yaml)
    with open(tmpdir / "inc2" / "include_seq.yaml", "w+") as f:
        f.write(include_seq_yaml)

    return tmpdir


def test_artificial_data(artificial_data):
    with open(artificial_data / "test.yaml") as f:
        d = yaml.load(f, Loader=ExtendedLoader)
    assert d == expected_dict, "Fail"
