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
from contextlib import contextmanager
from functools import partial
from itertools import starmap

import yaml

try:
    import mlflow
except ImportError:
    print("Warning: package 'mlflow' is not installed")

from ..configuration import Configuration
from .extended_loader import get_loader


def ensure_list(args):
    if isinstance(args, list):
        return args
    elif isinstance(args, argparse.Namespace):
        return args.configs
    else:
        return [args]


def load_cfg_from_yaml(cmd, cfg_file, **kwds):
    print(f"Reading file: {cfg_file.name}")
    config = yaml.load(cfg_file, Loader=get_loader(cmd))
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


def log_cfg_values(cfg, keys):
    for k in keys:
        if k in cfg.mlflow.keys():
            log_fn = getattr(mlflow, k)
            log_fn(cfg.mlflow[k])


@contextmanager
def mlflow_tracking(cfg, name):
    if "mlflow" in sys.modules and cfg.mlflow["enable"]:
        query = f"tags.mlflow.runName = '{cfg.name}'"
        results = mlflow.search_runs(filter_string=query, output_format="list")
        if len(results) > 0:
            run_id = results[0].info.run_id
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=cfg.name)
        child_run = mlflow.start_run(run_name="-".join((cfg.name, name)), nested=True)
        log_cfg(cfg)
        mlflow.autolog(exclusive=False)
        mlflow.log_param("name", cfg.name)
        log_cfg_values(cfg, ["log_params"])
        try:
            yield child_run
        finally:
            log_cfg_values(cfg, ["log_artifacts", "log_artifact"])

            mlflow.end_run()  # child_run
            mlflow.end_run()  # parent_run
    else:
        yield None


def run(cfg, fn, framework):
    # CONFIG ##################################################################
    print(cfg)

    # MLFLOW ##################################################################
    with mlflow_tracking(cfg, fn.__name__) as active_run:

        # SEED ################################################################
        # Set seed to ensure reproducibility
        print(f"Setting random seed to: {cfg.seed}")
        framework.set_seed(cfg.seed)

        # LOAD DATA ###########################################################
        print("Loading data...")
        data = cfg.data_loader(**cfg.data_loader_kwds)
        data = framework.prepare_data(data)

        # LOAD MODEL ##########################################################
        print("Loading model...")
        model = cfg.model
        model = framework.compile_model(model, **cfg.compile_kwds)
        model = framework.load_checkpoint(model, cfg.model_checkpoints)

        res = fn(model, data, **getattr(cfg, fn.__name__ + "_kwds"))
        if active_run:
            log_res(fn.__name__, res)
    return cfg, dict(model=model, data=data, res=res)


def cfg_generator(cmd, cfg_files, **kwds):
    for cfg_or_file in cfg_files:
        if isinstance(cfg_or_file, Configuration):
            yield cfg_or_file
        elif isinstance(cfg_or_file, io.TextIOBase):
            with contextlib.closing(cfg_or_file) as cfg_file:
                yield load_cfg_from_yaml(cmd, cfg_file, **kwds)
        elif os.path.isfile(cfg_or_file) and cfg_or_file.endswith(".yaml"):
            with open(cfg_or_file, "r") as cfg_file:
                yield load_cfg_from_yaml(cmd, cfg_file, **kwds)
        elif os.path.isdir(cfg_or_file):
            path = cfg_or_file
            files = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.endswith(".yaml")
            ]
            yield from cfg_generator(cmd, files)
        else:
            raise ValueError(f"Configuration file '{cfg_or_file}' not found.'")


def process_cli_args(args):
    if isinstance(args, argparse.Namespace):
        return {"mlflow": {"enable": args.no_mlflow}}
    else:
        return {}


def cmd_helper(cmd, framework, args, **kwds):
    cfg_files = ensure_list(args)
    kwds.update(process_cli_args(args))
    cfgs = cfg_generator(cmd.__name__, cfg_files, **kwds)
    return dict(map(partial(run, fn=cmd, framework=framework), cfgs))
