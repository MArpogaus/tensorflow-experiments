#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : setup.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-03-23 17:32:10
# COPYRIGHT ###################################################################
# Copyright 2020 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTES ######################################################################
#
# This project is following the
# [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 15:25:51
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-03-23 17:32:10
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
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
