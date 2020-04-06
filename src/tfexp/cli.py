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
# modified time : 2020-04-06 15:24:14
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 15:21:06
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import argparse

from .configuration import Configuration

# FUNCTION DEFINITIONS ########################################################


def f_train(args):
    cfg = Configuration.from_yaml(args.config)

    print(cfg)

    (x_train, y_train), (x_test, y_test) = cfg.data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = cfg.model
    # cfg.compile_kwargs['loss']=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(**cfg.compile_kwargs)
    model.fit(x=x_train, y=y_train, **cfg.fit_kwargs)


def f_predict():
    pass


def cli():
    p = argparse.ArgumentParser(
        description='Help me to conduct my experiments')
    subparsers = p.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train the model')
    # Add specific options for option1 here, but here's
    train_parser.add_argument('config', type=argparse.FileType(mode='r'))
    # train_parser.add_argument('--model', type=tf.keras.Model)
    # train_parser.add_argument('--experiment_name', type=str)
    # train_parser.add_argument('--model_args', type=list)
    # train_parser.add_argument('--model_kwargs', type=dict)
    # train_parser.add_argument('--model_compile_kwargs', type=dict)
    # train_parser.add_argument('--model_fit_kwargs', type=dict)
    # train_parser.add_argument('--save_path', type=str)
    # an example
    train_parser.set_defaults(func=f_train)

    predict_parser = subparsers.add_parser('predict')
    # Add specific options for option1 here
    predict_parser.set_defaults(func=f_predict)

    args = p.parse_args()
    args.func(args)
