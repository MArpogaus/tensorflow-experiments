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
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-04-08 07:58:38
#  changes made : return configuration and data
# modified by   : Marcel Arpogaus
# modified time : 2020-04-06 15:23:11
#  changes made : newly written
###############################################################################
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops

from sklearn.model_selection import train_test_split

from . import utils
from .configuration import Configuration


def train(args):
    if type(args) is str:
        cfg_file = open(args, 'r')
    else:
        cfg_file = args.config
    cfg = Configuration.from_yaml(cfg_file)
    cfg_file.close()
    print(cfg)

    # SEED ####################################################################
    # Set seed to ensure reproducibility
    tf.random.set_seed(cfg.seed)

    # LOAD DATA ###############################################################
    data = cfg.data.load_data(**cfg.data_kwds)
    if type(data) is tuple:
        train_x, train_y = data[0]
        val_x, val_y = None, None
        test_x, test_y = data[1]
        # VALIDATION DATA #####################################################
        if cfg.validation_split is not None:
            print(f'splitting of {cfg.validation_split*100:.2f}% '
                  'as validation data')
            train_x, val_x, train_y, val_y = train_test_split(
                train_x, train_y,
                test_size=cfg.validation_split,
                shuffle=True,
                random_state=cfg.seed)

        # NORMALIZATION #######################################################
        if cfg.data_preprocessor is not None:
            print(f'preprocessing data')
            dp = cfg.data_preprocessor
            dp.fit(train_x, train_y)
            test_x, test_y = dp.transform(test_x, test_y)
            if cfg.cross_validation is None:
                train_x, train_y = dp.transform(train_x, train_y)
            if val_x is not None:
                val_x, val_y = dp.transform(val_x, val_y)

        fit_kwds = dict(x=train_x,
                        y=train_y)

        if val_x is not None:
            fit_kwds['validation_data'] = (val_x, val_y)
            train_data = (train_x, val_x, train_y, val_y)
        else:
            train_data = (train_x, train_y)
        test_data = (test_x, test_y)

    elif isinstance(data, (dataset_ops.DatasetV1,
                           dataset_ops.DatasetV2,
                           iterator_ops.Iterator)):
        fit_kwds = dict(x=data)

        if cfg.validation_split is not None:
            ValueError(
                f'validation_split not supported with type(data)={type(data)}')

        if cfg.cross_validation is not None:
            ValueError(
                f'cross_validation not supported with type(data)={type(data)}')

        train_data = data
        test_data = None

    else:
        raise ValueError(f"Dataset type '{type(data)}' unsupported")

    model = cfg.model
    model.compile(**cfg.compile_kwds)
    model.summary()

    # TRAIN ###################################################################
    if cfg.cross_validation is not None:
        cv = cfg.cross_validation
        epoch = 0
        history = {}
        for i, (train_idx, val_idx) in enumerate(cv.split(X=train_x)):
            train_x_cv = train_x[train_idx]
            train_y_cv = train_y[train_idx]
            val_x_cv = train_x[val_idx]
            val_y_cv = train_y[val_idx]

            # NORMALIZATION ###################################################
            if cfg.data_preprocessor is not None:
                print(f'preprocessing data')
                dp = cfg.data_preprocessor
                train_x_cv, train_y_cv = dp.fit_transform(train_x_cv.copy(),
                                                          train_y_cv.copy())
                val_x_cv, val_y_cv = dp.transform(val_x_cv.copy(),
                                                  val_y_cv.copy())

            # FIT MODEL #######################################################
            print(f'CV iteration {i}')
            h = model.fit(
                x=train_x_cv,
                y=train_y_cv,
                validation_data=(val_x_cv, val_y_cv),
                initial_epoch=epoch,
                **cfg.fit_kwds)
            history[i] = h
            epoch = max(h.epoch + [0]) + 1

    else:
        # FIT MODEL ###########################################################
        history = model.fit(**fit_kwds, **cfg.fit_kwds)

    # EVALUATE ################################################################
    if test_x is not None:
        model.evaluate(test_x, test_y)

    return history, cfg, train_data, test_data


def predict():
    pass
