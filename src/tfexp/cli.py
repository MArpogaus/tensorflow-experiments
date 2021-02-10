#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : cli.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-04-06 15:21:06
# changed : 2020-11-25 13:28:15
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
import os
import argparse

from . import train, test, predict


def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"{path} is not a valid path")


def cli():
    p = argparse.ArgumentParser(
        description='Help me to conduct my experiments')
    subparsers = p.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train the model')
    train_parser.add_argument('configs', type=dir_path, nargs='+')
#    train_parser.add_argument('config', type=argparse.FileType(mode='r'))
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument('config', type=argparse.FileType(mode='r'))
    test_parser.set_defaults(func=test)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(func=predict)

    args = p.parse_args()
    args.func(args)
