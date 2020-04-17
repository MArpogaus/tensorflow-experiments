import os
import re
import yaml

import numpy as np
import datetime as dt

from typing import IO


class ExtendedLoader(yaml.Loader):
    """YAML Loader with additional constructors and global variables."""

    GLOBAL_VARIABLES = {}

    def __init__(self, stream: IO) -> None:
        super().__init__(stream)

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        self.get_global_variable_pattern = re.compile(r'^\$(\w+)$')
        self.set_global_variable_pattern = re.compile(r'^(\w+)\s+<=\s+(.+)$')

        self.add_constructor('!datetime', self._date_time_constructor)
        self.add_constructor('!product', self._np_product_constructor)
        self.add_constructor('!sum', self._np_sum_constructor)
        self.add_constructor('!join', self._join_constructor)
        self.add_constructor('!include', self._include_constructor)
        self.add_constructor('!global',
                             self._set_global_variable_constructor)
        self.add_constructor('!load',
                             self._get_global_variable_constructor)

        self.add_implicit_resolver(u'!global',
                                   self.set_global_variable_pattern,
                                   None)

        self.add_implicit_resolver(u'!load',
                                   self.get_global_variable_pattern,
                                   None)

    def _date_time_constructor(self, loader, node):
        value = loader.construct_scalar(node)
        now = dt.datetime.now()
        return value.format(now=now)

    def _np_product_constructor(self, loader, node):
        value = loader.construct_sequence(node)
        return np.product(value)

    def _np_sum_constructor(self, loader, node):
        value = loader.construct_sequence(node)
        return np.sum(value)

    def _join_constructor(self, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def _include_constructor(self, loader, node):
        path = os.path.join(self._root, self.construct_scalar(node))

        with open(path, 'r') as yf:
            ref = yaml.load(yf, Loader=loader.__class__)

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
