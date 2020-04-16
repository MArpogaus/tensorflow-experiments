import os
import re
import yaml

import numpy as np
import datetime as dt

from typing import IO


def _date_time_constructor(loader, node):
    value = loader.construct_scalar(node)
    now = dt.datetime.now()
    return value.format(now=now)


def _np_product_constructor(loader, node):
    value = loader.construct_sequence(node)
    return np.product(value)


def _np_sum_constructor(loader, node):
    value = loader.construct_sequence(node)
    return np.sum(value)


def _join_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def _include_constructor(loader, node):
    path = loader.construct_scalar(node)
    with open(path, 'r') as yf:
        ref = yaml.load(yf, Loader=loader.__class__)

    return ref


def _set_global_variable_constructor(loader, node):
    if isinstance(node, yaml.ScalarNode):
        scalar = loader.construct_scalar(node)
        match = ExtendedLoader.SET_GLOBAL_VARIABLE_PATTERN.match(scalar)
        key, value = match.groups()
    else:
        seq = loader.construct_sequence(node)
        key = seq.pop(0)
        value = seq.pop(0)
    ExtendedLoader.GLOBAL_VARIABLES[key] = value

    return value


def _get_global_variable_constructor(loader, node):
    key = loader.construct_scalar(node)
    key = key.replace('$', '')
    value = ExtendedLoader.GLOBAL_VARIABLES[key]

    return value


class ExtendedLoader(yaml.Loader):
    """YAML Loader with additional constructors and global variables."""

    GLOBAL_VARIABLES = {}
    GET_GLOBAL_VARIABLE_PATTERN = re.compile(r'^\$(\w+)$')
    SET_GLOBAL_VARIABLE_PATTERN = re.compile(r'^(\w+)\s+<=\s+(.+)$')

    def __init__(self, stream: IO) -> None:
        super().__init__(stream)

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        self.add_constructor('!datetime', _date_time_constructor)
        self.add_constructor('!product', _np_product_constructor)
        self.add_constructor('!sum', _np_sum_constructor)
        self.add_constructor('!join', _join_constructor)
        self.add_constructor('!include', _include_constructor)
        self.add_constructor('!global',
                             _set_global_variable_constructor)
        self.add_constructor('!load',
                             _get_global_variable_constructor)

        self.add_implicit_resolver(u'!global',
                                   self.SET_GLOBAL_VARIABLE_PATTERN,
                                   None)

        self.add_implicit_resolver(u'!load',
                                   self.GET_GLOBAL_VARIABLE_PATTERN,
                                   None)
