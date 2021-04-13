#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : configuration.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-04-06 14:47:58
# changed : 2020-11-17 10:28:53
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

# REQUIRED PYTHON MODULES #####################################################
import sys
from pprint import pformat

import tensorflow as tf


class Configuration:
    """This class stores all states required for experiments"""

    def __init__(
        self,
        model: tf.keras.Model,
        data_loader: callable,
        seed: int,
        name: str = None,
        use_mlflow: bool = True,
        model_checkpoints: str = None,
        fit_kwds: dict = {},
        data_loader_kwds: dict = {},
        compile_kwds: dict = {},
        evaluate_kwds: dict = {},
    ):

        # COMMON ##############################################################
        self.seed = seed
        self.name = name or model.name
        self.use_mlflow = use_mlflow and "mlflow" in sys.modules

        # MODEL ###############################################################
        self.model = model
        self.model_checkpoints = model_checkpoints or ""

        # DATASET #############################################################
        self.data_loader = data_loader
        self.data_loader_kwds = data_loader_kwds

        # KERAS ###############################################################
        self.fit_kwds = fit_kwds
        self.compile_kwds = compile_kwds
        self.evaluate_kwds = evaluate_kwds

    def __repr__(self) -> str:
        """Display Configuration values."""

        def format(x):
            return f"  {x[0]}={pformat(x[1],indent=4)}"

        attr = []
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attr.append((a, getattr(self, a)))
        return self.__class__.__name__ + "(\n" + ",\n".join(map(format, attr)) + "\n)"
