#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-04-09 12:39:02 (Marcel Arpogaus)
# changed : 2021-04-22 18:03:08 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
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
import importlib
from .utils import cmd_helper


def get_framework(framework):
    framework = framework.replace(".", "")
    try:
        return importlib.import_module("tfexp.frameworks." + framework)
    except ModuleNotFoundError:
        raise (ValueError(f"framework '{framework}' not supported"))


def fit(args, **kwds):
    framework = get_framework("tensorflow")
    return cmd_helper(framework.fit, framework, args, **kwds)


def evaluate(args, **kwds):
    framework = get_framework("tensorflow")
    return cmd_helper(framework.evaluate, framework, args, **kwds)
