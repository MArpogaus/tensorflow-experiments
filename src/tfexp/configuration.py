# -*- coding: utf-8 -*-
# @Author: marcel
# @Date:   2020-03-17 13:17:31
# @Last Modified by:   marcel
# @Last Modified time: 2020-03-19 10:05:57
import io
import os
import yaml
import shutil

import tensorflow as tf


def frmt(key, value, width=80):
    """
     helper function for string formatting.

     :param      s     string to be formatted

     :returns:   a left aligned version of the string with 20 char length.
    """
    if type(value) is dict:
        s = ''
        for k in value:
            s += frmt(key + '.' + k, value[k], width)
        return s
    else:
        return "| {: <25}: {}".format(key, value).ljust(width - 1) + '|\n'


class Configuration():
    """docstring for Configuration"""

    def __init__(self,
                 model: tf.keras.Model,
                 data: object,
                 seed: int,
                 experiment_name: str = "unnamed-experiment",
                 model_args: list = [],
                 model_kwds: dict = {},
                 data_kwds: dict = {},
                 compile_kwds: dict = {},
                 fit_kwds: dict = {}):

        # Common --------------------------------------------------------------
        self.experiment_name = experiment_name
        self.seed = seed

        # Model ---------------------------------------------------------------
        self.model = model

        # Dataset -------------------------------------------------------------
        self.data = data
        self.data_kwds = data_kwds

        # Training ------------------------------------------------------------
        self.compile_kwds = compile_kwds
        self.fit_kwds = fit_kwds

    def __repr__(self)->str:
        """Display Configuration values."""
        width = shutil.get_terminal_size((80, 20)).columns

        attr = []
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attr.append((a, getattr(self, a)))
        return f"{f'+--[ Configuration ]'.ljust(width - 1,'-') + '+'}\n" \
               f"{''.join(map(lambda x:frmt(x[0],x[1],width),attr))}"\
               f"{'+' + '-'*(width - 2) + '+'}"

    @classmethod
    def from_yaml(cls, configuration_file: io.TextIOWrapper):
        config = yaml.load(configuration_file, Loader=yaml.Loader)
        return Configuration(**config)

    def to_yaml(self, configuration_file: io.TextIOWrapper):
        yaml.dump(self, Dumper=yaml.Dumper)
