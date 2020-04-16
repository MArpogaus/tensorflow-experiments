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
# modified time : 2020-04-16 11:33:53
#  changes made : added load_data function for data preprocessing
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 14:47:58
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import io
import yaml
import shutil

import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2
from tensorflow.python.data.ops.iterator_ops import Iterator


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
                 data_loader: callable,
                 seed: int,
                 dump_cfg_path: str = None,
                 model_args: list = [],
                 model_kwds: dict = {},
                 data_kwds: dict = {},
                 compile_kwds: dict = {},
                 fit_kwds: dict = {},
                 data_preprocessor: callable = None,
                 **kwds):

        # Common --------------------------------------------------------------
        self.seed = seed
        self.dump_cfg_path = dump_cfg_path

        # Model ---------------------------------------------------------------
        self.model = model

        # Dataset -------------------------------------------------------------
        self.data_loader = data_loader
        self.data_kwds = data_kwds
        self.data_preprocessor = data_preprocessor

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

    def load_data(self):
        data = self.data_loader(**self.data_kwds)

        # DATA FORMAT #########################################################
        if isinstance(data, tuple):
            if len(data) == 1:
                # Data format: (train_x,[train_y])
                train_data = data
                test_data = None
                val_data = None
            if len(data) == 2:
                # Data format: ((train_x,[train_y]),
                #               (test_x,[test_y]))
                train_data = data[0]
                test_data = data[1]
                val_data = None
            elif len(data) == 3:
                # Data format: ((train_x,[train_y]),
                #               (val_x,[val_y]),
                #               (test_x,[test_y]))
                train_data = data[0]
                val_data = data[1]
                test_data = data[2]
            else:
                raise ValueError(f"unexpected structure of dataset")

        elif isinstance(data, dict):
            train_data = data['train']
            test_data = data.get('test', None)
            val_data = data.get('validate', None)

        elif isinstance(data, (DatasetV1, DatasetV2, Iterator)):
            train_data = data
            test_data = None
            val_data = None
        else:
            raise ValueError(f"Dataset type '{type(data)}' unsupported")

        # Pack data in expected format
        data = {
            'train': train_data,
            'validate': val_data,
            'test': test_data
        }

        # PREPROCESSING #######################################################
        if self.data_preprocessor is not None:
            data = self.data_preprocessor(data)

        return data
