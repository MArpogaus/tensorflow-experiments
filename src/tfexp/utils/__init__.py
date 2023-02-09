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
import logging
import numbers
import os
import tempfile
import traceback
from contextlib import contextmanager
from functools import partial
from itertools import starmap

import yaml

from ..configuration import Configuration
from .extended_loader import get_loader

# PRIVATE VARIABLES ###########################################################
__LOGGER__ = logging.getLogger(__name__)


# IMPORT MLFLOW ###############################################################
try:
    import mlflow

    __MLFLOW_AVAILABLE__ = True
except ImportError:
    __MLFLOW_AVAILABLE__ = False


# FUNCTION DEFINITIONS #######################################################
def ensure_list(args):
    if isinstance(args, list):
        return args
    elif isinstance(args, argparse.Namespace):
        return args.configs
    else:
        return [args]


# inspired by: https://stackoverflow.com/questions/20656135
def deep_merge_dicts(d1, d2):
    res = d1.copy()
    for key, value in d2.items():
        if isinstance(value, dict):
            res[key] = deep_merge_dicts(value, res.setdefault(key, {}))
        else:
            res[key] = value

    return res


def load_cfg_from_yaml(cmd, cfg_file, **kwds):
    __LOGGER__.info(f"Reading file: {cfg_file.name}")
    config = yaml.load(cfg_file, Loader=get_loader(cmd))
    config = deep_merge_dicts(config, kwds)
    cfg = Configuration(**config)

    return cfg


def log_cfg(cfg):
    mlflow.log_text(repr(cfg), "config.txt")


def log_res(key, res, step=None):
    if isinstance(res, dict):
        list(starmap(log_res, res.items()))
    elif isinstance(res, list):
        list(starmap(partial(log_res, key), zip(res, range(len(res)))))
    elif isinstance(res, numbers.Number):
        mlflow.log_metric(key, float(res), step)
    else:
        __LOGGER__.info(f"Warning: metric type '{type(res)}' unsupported.")


def call_mlflow_log_functions(fns):
    for fn, v in fns.items():
        log_fn = getattr(mlflow, fn)
        if isinstance(v, list):
            list(map(log_fn, v))
        elif isinstance(v, dict):
            if "kwds" in v.keys() and v.pop("kwds"):
                log_fn(**v)
            else:
                log_fn(v)
        else:
            log_fn(v)


@contextmanager
def mlflow_tracking(cfg, name):
    if cfg.mlflow.get("enable", False):
        __LOGGER__.debug("mlflow tracking enabled")
        if __MLFLOW_AVAILABLE__:
            __LOGGER__.info("Starting mlflow tracking")
            run_id = os.environ.get("MLFLOW_RUN_ID", False)
            if run_id:
                mlflow.start_run()

            mlflow.autolog()
            if "experiment_name" in cfg.mlflow.keys():
                mlflow.set_experiment(cfg.mlflow["experiment_name"])
            run = mlflow.start_run(
                run_name="-".join((cfg.name, name)),
                nested=mlflow.active_run() is not None,
            )
            log_cfg(cfg)
            mlflow.log_param("name", cfg.name)
            mlflow.set_tag("tfexp_cmd", name)
            call_mlflow_log_functions(cfg.mlflow.get("before_run", {}))

            status = "FINISHED"
            exc = None
            try:
                yield run
            except Exception as e:
                with tempfile.NamedTemporaryFile(
                    prefix="traceback", suffix=".txt"
                ) as tmpf:
                    with open(tmpf.name, "w+") as f:
                        f.write(traceback.format_exc())
                    mlflow.log_artifact(tmpf.name)
                status = "FAILED"
                exc = e
            finally:
                call_mlflow_log_functions(cfg.mlflow.get("after_run", {}))

                mlflow.end_run(status=status)
                __LOGGER__.info("Finished mlflow tracking")
                if exc:
                    raise exc
        else:
            __LOGGER__.warn(
                "mlflow tracking enabled, but package 'mlflow' is not installed"
            )
            yield None
    else:
        __LOGGER__.debug("mlflow tracking disabled")
        yield None


def run(cfg, fn, framework):
    # CONFIG ##################################################################
    __LOGGER__.info(cfg)

    # MLFLOW ##################################################################
    with mlflow_tracking(cfg, fn.__name__) as active_run:
        # SEED ################################################################
        # Set seed to ensure reproducibility
        __LOGGER__.info(f"Setting random seed to: {cfg.seed}")
        framework.set_seed(cfg.seed)

        # LOAD DATA ###########################################################
        __LOGGER__.info("Loading data...")
        data = cfg.data_loader(**cfg.data_loader_kwds)
        data = framework.prepare_data(data)

        # LOAD MODEL ##########################################################
        __LOGGER__.info("Loading model...")
        model = cfg.model(**cfg.model_kwds)
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
