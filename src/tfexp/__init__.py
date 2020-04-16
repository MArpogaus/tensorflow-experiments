#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : __init__.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-06 15:23:11
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
# CHANGELOG ###################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-04-16 23:01:57
#  changes made : added test routine
# modified by   : Marcel Arpogaus
# modified time : 2020-04-16 11:30:32
#  changes made : refactoring and cleaning
# modified by   : Marcel Arpogaus
# modified time : 2020-04-08 19:00:30
#  changes made : added support for data organized in dict
# modified by   : Marcel Arpogaus
# modified time : 2020-04-08 07:58:38
#  changes made : return configuration and data
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 15:23:11
#  changes made : newly written
###############################################################################
import os
import tensorflow as tf

from .configuration import Configuration


def _prepare(args):
    if type(args) is str:
        cfg_file = open(args, 'r')
    else:
        cfg_file = args.config
    cfg = Configuration.from_yaml(cfg_file)
    cfg_file.close()

    # SEED ####################################################################
    # Set seed to ensure reproducibility
    tf.random.set_seed(cfg.seed)

    # LOAD DATA ###############################################################
    data = cfg.load_data()

    # COMPILE MODEL ###########################################################
    model = cfg.model
    model.compile(**cfg.compile_kwds)

    # LOAD CHECKPOINT #########################################################
    # ref: https://www.tensorflow.org/tutorials/keras/save_and_load
    if os.path.exists(cfg.model_checkpoints):
        # get path to latest checkpoint
        latest_cp = tf.train.latest_checkpoint(cfg.model_checkpoints)
        # Load model
        print(f'restoring model from checkpoint {latest_cp}')
        model.load_weights(latest_cp)
        # Finding the epoch index from which we are resuming

    return cfg, model, data


def train(args):
    cfg, model, data = _prepare(args)
    model.summary()

    if isinstance(data['train'], tuple):
        # Data format: (train_x, train_y)
        fit_kwds = dict(x=data['train'][0],
                        y=data['train'][1],
                        validation_data=data['validate'])
    else:
        # Data format: dataset / generator or unsupervised
        fit_kwds = dict(x=data['train'],
                        validation_data=data['validate'])

    # TRAIN ###################################################################
    history = model.fit(**fit_kwds, **cfg.fit_kwds)

    return history, cfg, data


def test(args):
    cfg, model, data = _prepare(args)
    print(cfg)
    model.summary()

    if isinstance(data['test'], tuple):
        # Data format: (test_x, test_y)
        evaluate_kwds = dict(x=data['test'][0],
                             y=data['test'][1])
    else:
        # Data format: dataset / generator or unsupervised
        evaluate_kwds = dict(x=data['test'])

    # EVALUATE ################################################################
    loss = model.evaluate(**evaluate_kwds, **cfg.evaluate_kwds)

    return loss, cfg, model


def predict():
    pass
