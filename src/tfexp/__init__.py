#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-04-06 15:23:11
# changed : 2020-11-25 16:45:52
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
import io
import sys

try:
    import mlflow
except ImportError:
    print("Warning: ackage 'mlflow' is not installed")

import tensorflow as tf

from .configuration import Configuration


def build_model(cfg):
    # COMPILE MODEL ###########################################################
    model = cfg.model
    model.compile(**cfg.compile_kwds)

    # LOAD CHECKPOINT #########################################################
    # ref: https://www.tensorflow.org/tutorials/keras/save_and_load
    if os.path.exists(cfg.model_checkpoints):
        if os.path.isfile(cfg.model_checkpoints):
            cp = cfg.model_checkpoints
        else:
            # get path to latest checkpoint
            cp = tf.train.latest_checkpoint(cfg.model_checkpoints)

        # Load model
        if cp is not None:
            print(f"restoring model from checkpoint {cp}")
            model.load_weights(cp)
    else:
        os.makedirs(cfg.model_checkpoints)

    return model


def get_model_and_data(cfg):
    # SEED ####################################################################
    # Set seed to ensure reproducibility
    tf.random.set_seed(cfg.seed)

    # LOAD DATA ###############################################################
    data = cfg.load_data()

    # LOAD MODEL ##############################################################
    model = build_model(cfg)

    return model, data


def train(args, **kwds):
    if isinstance(args, (str, io.TextIOBase)):
        cfg_files = [args]
    elif isinstance(args, list):
        cfg_files = args
    else:
        cfg_files = args.configs
        if not args.use_mlflow:
            kwds["use_mlflow"] = False

    results = {}
    for cfg_file in cfg_files:
        if os.path.isfile(cfg_file):
            history, cfg, data = _train(cfg_file, **kwds)
            results[cfg.name] = (history, cfg)
        elif os.path.isdir(cfg_file):
            path = cfg_file
            files = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.endswith("yaml")
            ]
            results = {**results, **train(files, **kwds)}
        else:
            raise ValueError("Unsupported type for args: " f"{type(args)}")
    return results


def _train(cfg_file, **kwds):
    print(f"Reading file: {cfg_file}")
    cfg = Configuration.from_yaml(cfg_file, **kwds)
    print(cfg)

    if "mlflow" in sys.modules and cfg.use_mlflow:
        mlflow.autolog()

    model, data = get_model_and_data(cfg)
    model.summary()

    if isinstance(data["train"], tuple):
        # Data format: (train_x, train_y)
        fit_kwds = dict(
            x=data["train"][0], y=data["train"][1], validation_data=data["validate"]
        )
    else:
        # Data format: dataset / generator or unsupervised
        fit_kwds = dict(x=data["train"], validation_data=data["validate"])

    # TRAIN ###################################################################
    history = model.fit(**fit_kwds, **cfg.fit_kwds)

    return history, cfg, data


def test(args):
    cfg, model, data = get_model_and_data(args)
    print(cfg)
    model.summary()

    if isinstance(data["test"], tuple):
        # Data format: (test_x, test_y)
        evaluate_kwds = dict(x=data["test"][0], y=data["test"][1])
    else:
        # Data format: dataset / generator or unsupervised
        evaluate_kwds = dict(x=data["test"])

    # EVALUATE ################################################################
    loss = model.evaluate(**evaluate_kwds, **cfg.evaluate_kwds)

    return loss, cfg, model


def predict():
    pass
