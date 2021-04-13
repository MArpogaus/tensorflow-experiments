#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-04-17 14:35:45
# changed : 2020-11-15 19:41:43
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
import argparse
import contextlib
import io
import os
import sys
import tempfile
from functools import partial
from itertools import starmap

import yaml

try:
    import mlflow
except ImportError:
    print("Warning: package 'mlflow' is not installed")

from ..configuration import Configuration
from .extended_loader import ExtendedLoader


def ensure_list(args, **kwds):
    if isinstance(args, list):
        return args
    elif isinstance(args, argparse.Namespace):
        return args.configs
    else:
        return [args]


def load_cfg_from_yaml(cfg_file, **kwds):
    print(f"Reading file: {cfg_file.name}")
    config = yaml.load(cfg_file, Loader=ExtendedLoader)
    config.update(**kwds)
    cfg = Configuration(**config)

    return cfg


def log_cfg(cfg):
    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "config.txt")
    with open(tmp_file, "w+") as f:
        f.write(repr(cfg))
    mlflow.log_artifact(tmp_file)


def log_res(key, res, step=None):
    if isinstance(res, dict):
        list(starmap(log_res, res.items()))
    elif isinstance(res, list):
        list(starmap(partial(log_res, key), zip(res, range(len(res)))))
    elif isinstance(res, (int, float)):
        mlflow.log_metric(key, float(res), step)
    else:
        print(f"Warning: metric type '{type(res)}' unsupported.")


def run(cfg, fn, framework, **kwds):
    # CONFIG ##################################################################
    print(cfg)

    # MLFLOW ##################################################################
    if "mlflow" in sys.modules and cfg.use_mlflow:
        query = f"tags.mlflow.runName = '{cfg.name}'"
        results = mlflow.search_runs(filter_string=query, output_format="list")
        if len(results) > 0:
            run_id = results[0].info.run_id
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=cfg.name)
        mlflow.start_run(run_name=fn.__name__, nested=True)
        log_cfg(cfg)
        mlflow.autolog(exclusive=False)

    # SEED ####################################################################
    # Set seed to ensure reproducibility
    print(f"Setting random seed to: {cfg.seed}")
    framework.set_seed(cfg.seed)

    # LOAD DATA ###############################################################
    print("Loading data...")
    data = framework.load_data(cfg)

    # LOAD MODEL ##############################################################
    print("Loading model...")
    model = framework.compile_model(cfg)
    model = framework.load_checkpoint(cfg, model)

    res = fn(cfg, model, data)
    if "mlflow" in sys.modules and cfg.use_mlflow:
        log_res(fn.__name__, res)
        mlflow.end_run()
        mlflow.end_run()
    return cfg, dict(model=model, data=data, res=res)


def cfg_generator(cfg_files, **kwds):
    for cfg_or_file in cfg_files:
        if isinstance(cfg_or_file, Configuration):
            yield cfg_or_file
        elif isinstance(cfg_or_file, io.TextIOBase):
            with contextlib.closing(cfg_or_file) as cfg_file:
                yield load_cfg_from_yaml(cfg_file, **kwds)
        elif os.path.isfile(cfg_or_file) and cfg_or_file.endswith(".yaml"):
            with open(cfg_or_file, "r") as cfg_file:
                yield load_cfg_from_yaml(cfg_file, **kwds)
        elif os.path.isdir(cfg_or_file):
            path = cfg_or_file
            files = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.endswith(".yaml")
            ]
            yield from cfg_generator(files)
        else:
            raise ValueError(f"Configuration file '{cfg_or_file}' not found.'")


def cmd_helper(cmd, framework, args, **kwds):
    cfg_files = ensure_list(args, **kwds)
    cfgs = cfg_generator(cfg_files)
    return dict(map(partial(run, fn=cmd, framework=framework, **kwds), cfgs))
