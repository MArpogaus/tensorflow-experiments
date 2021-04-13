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
import hashlib
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
    cfg_file = open_cfg_file(cfg_file_path)
    config = yaml.load(cfg_file, Loader=ExtendedLoader)
    config.update(**kwds)
    cfg = Configuration(**config)

    cfg_file.seek(0)
    sha1 = hashlib.sha1(cfg_file.read().encode("utf-8")).hexdigest()
    cfg_file.close()

    return cfg, sha1


def log_cfg(cfg_file_path):
    cfg_file = open_cfg_file(cfg_file_path)

    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "config.yaml")
    with open(tmp_file, "w+") as f:
        f.write(cfg_file.read())
    mlflow.log_artifact(tmp_file)

    cfg_file.close()


def log_res(key, res, step=None):
    if isinstance(res, dict):
        list(starmap(log_res, res.items()))
    elif isinstance(res, list):
        list(starmap(partial(log_res, key), zip(res, range(len(res)))))
    elif isinstance(res, (int, float)):
        mlflow.log_metric(key, float(res), step)
    else:
        print(f"Warning: metric type '{type(res)}' unsupported.")


def run(cfg_file_path, fn, framework, **kwds):
    # CONFIG ##################################################################
    print(f"Reading file: {cfg_file_path}")
    cfg, sha1 = load_cfg_from_yaml(cfg_file_path, **kwds)
    print(cfg)

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

    # MLFLOW ##################################################################
    if "mlflow" in sys.modules and cfg.use_mlflow:
        experiment_id = mlflow.set_experiment(cfg.name)
        query = f"tags.mlflow.runName = '{sha1}'"
        results = mlflow.search_runs(
            experiment_ids=experiment_id, filter_string=query, output_format="list"
        )
        if len(results) > 0:
            run_id = results[0].info.run_id
            mlflow.start_run(experiment_id=experiment_id, run_id=run_id)
        else:
            mlflow.start_run(experiment_id=experiment_id, run_name=sha1)
            log_cfg(cfg_file_path)
        mlflow.start_run(run_name=fn.__name__, nested=True)
        mlflow.autolog(exclusive=False)
    res = fn(cfg, model, data)
    if "mlflow" in sys.modules and cfg.use_mlflow:
        log_res(fn.__name__, res)
        mlflow.end_run()
        mlflow.end_run()
    return dict(cfg=cfg, model=model, data=data, res=res)


def gen_cfg_file_mapper(fn, framework, **kwds):
    def map_fn(cfg_file_path):
        if os.path.isfile(cfg_file_path):
            res = run(cfg_file_path, fn, framework, **kwds)
            return (cfg_file_path, res)
        elif os.path.isdir(cfg_file_path):
            path = cfg_file_path
            files = [
                os.path.join(path, file)
                for file in os.listdir(path)
                if file.endswith("yaml")
            ]
            return path, dict(map(gen_cfg_file_mapper(fn, framework, **kwds), files))
        else:
            raise ValueError(f"Configuration file '{cfg_file_path}' not found.'")

    return map_fn


def cmd_helper(cmd, framework, args, **kwds):
    cfg_files = ensure_list(args, **kwds)
    return dict(map(gen_cfg_file_mapper(cmd, framework, **kwds), cfg_files))
