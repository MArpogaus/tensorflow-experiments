#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-04-09 12:39:02 (Marcel Arpogaus)
# changed : 2021-04-09 17:04:06 (Marcel Arpogaus)
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
import os
import io
import sys
import yaml
import tempfile
import argparse
import hashlib

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2
from tensorflow.python.data.ops.iterator_ops import Iterator

from .utils import ExtendedLoader

try:
    import mlflow
except ImportError:
    print("Warning: package 'mlflow' is not installed")

import tensorflow as tf

from .configuration import Configuration


def ensure_list(args, **kwds):
    if isinstance(args, list):
        return args
    elif isinstance(args, argparse.Namespace):
        return args.configs
    else:
        return list(args)


def compile_model(cfg):
    model = cfg.model
    model.compile(**cfg.compile_kwds)

    return model


def load_checkpoint(cfg, model):
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


def load_data(cfg, **kwds):
    data = cfg.data_loader(**{**cfg.data_loader_kwds, **kwds})

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
            raise ValueError("unexpected structure of dataset")

    elif isinstance(data, dict):
        train_data = data.get("train", None)
        test_data = data.get("test", None)
        val_data = data.get("validate", None)

    elif isinstance(data, (DatasetV1, DatasetV2, Iterator)):
        train_data = data
        test_data = None
        val_data = None
    else:
        raise ValueError(f"Dataset type '{type(data)}' unsupported")

    # Pack data in expected format
    data = {"train": train_data, "validate": val_data, "test": test_data}

    return data


def open_cfg_file(cfg_file_path):
    if isinstance(cfg_file_path, str):
        cfg_file = open(cfg_file_path, "r")
    elif isinstance(cfg_file_path, io.TextIOBase):
        cfg_file = cfg_file_path
    else:
        raise ValueError(
            "Unsupported type for configuration_file: " f"{type(cfg_file_path)}"
        )
    return cfg_file


def load_cfg_from_yaml(cfg_file_path, **kwds):
    print(f"Reading file: {cfg_file_path}")
    cfg_file = open_cfg_file(cfg_file_path)
    config = yaml.load(cfg_file, Loader=ExtendedLoader)
    config.update(**kwds)
    cfg = Configuration(**config)

    cfg_file.seek(0)
    sha1 = hashlib.sha1(cfg_file.read().encode("utf-8")).hexdigest()
    cfg_file.close()

    print(cfg)

    return cfg, sha1


def log_cfg(cfg_file_path):
    cfg_file = open_cfg_file(cfg_file_path)

    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "config.yaml")
    with open(tmp_file, "w+") as f:
        f.write(cfg_file.read())
    mlflow.log_artifact(tmp_file)

    cfg_file.close()


def load_model_and_data(cfg):
    # SEED ####################################################################
    # Set seed to ensure reproducibility
    tf.random.set_seed(cfg.seed)

    # LOAD DATA ###############################################################
    data = load_data(cfg)

    # LOAD MODEL ##############################################################
    model = compile_model(cfg)
    model = load_checkpoint(cfg, model)

    return model, data


def gen_cfg_file_mapper(fn, **kwds):
    def map_fn(cfg_file_path):
        if os.path.isfile(cfg_file_path):
            cfg, sha1 = load_cfg_from_yaml(cfg_file_path, **kwds)
            model, data = load_model_and_data(cfg)
            if "mlflow" in sys.modules and cfg.use_mlflow:
                experiment_id = mlflow.set_experiment(cfg.name)
                # mlflow.start_run(nested=True)
                mlflow.start_run(experiment_id=experiment_id, run_name=sha1)
                # TODO: retrive existing run from db
                log_cfg(cfg_file_path)
                mlflow.start_run(experiment_id=experiment_id, nested=True)
                # mlflow.start_run(run_id=0)
                mlflow.autolog(exclusive=False)
            res = fn(cfg, model, data)
            return (cfg_file_path, dict(cfg=cfg, model=model, data=data, res=res))
        elif os.path.isdir(cfg_file_path):
            path = cfg_file_path
            files = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.endswith("yaml")
            ]
            return path, dict(map(gen_cfg_file_mapper(fn, **kwds), files))
        else:
            raise ValueError(
                f"Unsupported type for cfg_file_path: {type(cfg_file_path)}"
            )

    return map_fn


def cmd_helper(cmd, args, **kwds):
    cfg_files = ensure_list(args, **kwds)
    return dict(map(gen_cfg_file_mapper(cmd, **kwds), cfg_files))


def train(args, **kwds):
    return cmd_helper(_train, args, **kwds)


def test(args, **kwds):
    return cmd_helper(_test, args, **kwds)


def _train(cfg, model, data):
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

    return history


def _test(cfg, model, data):
    if isinstance(data["test"], tuple):
        # Data format: (test_x, test_y)
        evaluate_kwds = dict(x=data["test"][0], y=data["test"][1])
    else:
        # Data format: dataset / generator or unsupervised
        evaluate_kwds = dict(x=data["test"])

    # EVALUATE ################################################################
    loss = model.evaluate(**evaluate_kwds, **cfg.evaluate_kwds)

    return loss
