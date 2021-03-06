#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : configuration.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-06 14:47:58
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
# modified time : 2020-04-06 15:24:21
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 14:47:58
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
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
                 data: tuple,
                 experiment_name: str = "unnamed-experiment",
                 model_args: list = None,
                 model_kwargs: dict = None,
                 compile_kwargs: dict = None,
                 fit_kwargs: dict = None,
                 save_path: str = './logs'):

        # Common --------------------------------------------------------------
        self.experiment_name = experiment_name
        self.save_path = os.path.join(save_path, self.experiment_name)

        # Model ---------------------------------------------------------------
        self.model = model

        # Dataset -------------------------------------------------------------
        self.data = data

        # Training ------------------------------------------------------------
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def __str__(self)->str:
        """Display Configuration values."""
        width = shutil.get_terminal_size((80, 20)).columns

        attr = []
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attr.append((a, getattr(self, a)))
        return f"{f'+--[ Configuration ]'.ljust(width - 1,'-') + '+'}\n" \
               f"{''.join(map(lambda x:frmt(x[0],x[1],width),attr))}"\
               f"{'+' + '-'*(width - 1) + '+'}"

    @classmethod
    def from_yaml(cls, configuration_file: io.TextIOWrapper):
        config = yaml.load(configuration_file, Loader=yaml.Loader)
        return Configuration(**config)

    def to_yaml(self, configuration_file: io.TextIOWrapper):
        yaml.dump(self, Dumper=yaml.Dumper)
