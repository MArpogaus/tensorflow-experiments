#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : cli.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-06 15:21:06
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
# modified time : 2020-04-16 22:04:09
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 15:21:06
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import argparse

from . import train, test, predict


def cli():
    p = argparse.ArgumentParser(
        description='Help me to conduct my experiments')
    subparsers = p.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train the model')
    train_parser.add_argument('config', type=argparse.FileType(mode='r'))
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument('config', type=argparse.FileType(mode='r'))
    test_parser.set_defaults(func=test)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(func=predict)

    args = p.parse_args()
    args.func(args)
