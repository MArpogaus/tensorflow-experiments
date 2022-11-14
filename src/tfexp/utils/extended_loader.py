# AUTHOR INFORMATION ##########################################################
# file    : extended_loader.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-05-18 09:29:39
# changed : 2020-11-15 19:41:35
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/)
#
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
###############################################################################

import datetime as dt

# REQUIRED PYTHON MODULES #####################################################
import os
import re
from typing import IO

import numpy as np
import yaml


class ExtendedLoader(yaml.Loader):
    """YAML Loader with additional constructors and global variables."""

    GLOBAL_VARIABLES = {}

    def __init__(self, stream: IO) -> None:
        super().__init__(stream)

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        self.get_global_variable_pattern = re.compile(r"^\$(\w+)$")
        self.set_global_variable_pattern = re.compile(r"^(\w+)\s*<=\s*(.+)$")

        self.add_constructor("!datetime", self._date_time_constructor)
        self.add_constructor("!abspath", self._abspath_constructor)
        self.add_constructor("!product", self._np_product_constructor)
        self.add_constructor("!sum", self._np_sum_constructor)
        self.add_constructor("!join", self._join_constructor)
        self.add_constructor("!pathjoin", self._path_join_constructor)
        self.add_constructor("!abspathjoin", self._abspath_join_constructor)
        self.add_constructor("!include", self._include_constructor)

        self.add_constructor("!store", self._set_global_variable_constructor)
        self.add_constructor("!load", self._get_global_variable_constructor)

        self.add_implicit_resolver("!store", self.set_global_variable_pattern, None)

        self.add_implicit_resolver("!load", self.get_global_variable_pattern, None)

    def _date_time_constructor(self, loader, node):
        value = loader.construct_scalar(node)
        now = dt.datetime.now()
        return value.format(now=now)

    def _abspath_constructor(self, loader, node):
        path = os.path.join(loader._root, self.construct_scalar(node))
        return os.path.abspath(path)

    def _np_product_constructor(self, loader, node):
        value = loader.construct_sequence(node)
        return np.product(value)

    def _np_sum_constructor(self, loader, node):
        value = loader.construct_sequence(node)
        return np.sum(value)

    def _join_constructor(self, loader, node):
        seq = loader.construct_sequence(node)
        return "".join([str(i) for i in seq])

    def _path_join_constructor(self, loader, node):
        seq = loader.construct_sequence(node)
        path = os.path.join(*seq)
        return path

    def _abspath_join_constructor(self, loader, node):
        path = self._path_join_constructor(loader, node)
        path = os.path.join(loader._root, path)
        return os.path.abspath(path)

    def _include_constructor(self, loader, node):
        path = os.path.abspath(os.path.join(loader._root, self.construct_scalar(node)))
        with open(path, "r") as yf:
            ref = yaml.load(yf, Loader=ExtendedLoader)
        return ref

    def _set_global_variable_constructor(self, loader, node):
        if isinstance(node, yaml.ScalarNode):
            scalar = loader.construct_scalar(node)
            match = self.set_global_variable_pattern.match(scalar)
            key, value = match.groups()
        else:
            seq = loader.construct_sequence(node)
            key = seq.pop(0)
            value = seq.pop(0)
        self.GLOBAL_VARIABLES[key] = value

        return value

    def _get_global_variable_constructor(self, loader, node):
        scalar = loader.construct_scalar(node)
        match = self.get_global_variable_pattern.match(scalar)
        key = match.groups()[0]
        value = self.GLOBAL_VARIABLES[key]

        return value


def get_loader(cmd):
    class CMDLoader(ExtendedLoader):
        def __init__(self, stream: IO) -> None:
            self.cmd = cmd

            super().__init__(stream)

            self.add_constructor("!switchcmd", self._switchcmd_constructor)

        def _switchcmd_constructor(self, loader, node):
            cases = self.construct_mapping(node, deep=True)
            return cases.get(self.cmd, cases.get("default", None))

    return CMDLoader
