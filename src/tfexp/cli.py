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
import argparse
import logging
import os
import signal
import sys

from . import evaluate, fit


def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def confirm_prompt(question: str) -> bool:
    reply = None
    question_str = f"{question} (y,n): "
    while reply not in ("y", "n"):
        reply = input(question_str).lower()
    return reply == "y"


original_sigint = signal.getsignal(signal.SIGINT)


def signal_handler(*args):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    if confirm_prompt("Do you want to cancel the running experiment?"):
        logging.info("Exiting due to user interrupt")
        sys.exit(1)

    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, signal_handler)


def cli():
    parser = argparse.ArgumentParser(
        description="Helps you to conduct and track your ml experiments"
    )
    parser.add_argument(
        "--log-file",
        type=argparse.FileType("a"),
        help="Path to logfile",
    )
    parser.add_argument("--log-level", default="info", help="Provide logging level.")

    subparsers = parser.add_subparsers()

    fit_parser = subparsers.add_parser("fit", help="fit the model")
    fit_parser.add_argument("configs", type=dir_path, nargs="+")
    fit_parser.add_argument(
        "--no-mlflow",
        help="use mlflow to record metrics",
        action="store_false",
    )
    fit_parser.set_defaults(func=fit)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate the model")
    evaluate_parser.add_argument("configs", type=dir_path, nargs="+")
    evaluate_parser.add_argument(
        "--no-mlflow",
        help="use mlflow to record metrics",
        action="store_false",
    )
    evaluate_parser.set_defaults(func=evaluate)

    signal.signal(signal.SIGINT, signal_handler)
    args = parser.parse_args()

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    if args.log_file is not None:
        handlers += [logging.StreamHandler(args.log_file)]
    # Configure logging
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    args.func(args)
