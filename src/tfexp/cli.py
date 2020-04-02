import argparse

from . import utils
from . import train, predict


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
    # train_parser.add_argument('--model_kwds', type=dict)
    # train_parser.add_argument('--model_compile_kwds', type=dict)
    # train_parser.add_argument('--model_fit_kwds', type=dict)
    # train_parser.add_argument('--save_path', type=str)
    # an example
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser('predict')
    # Add specific options for option1 here
    predict_parser.set_defaults(func=predict)

    args = p.parse_args()
    args.func(args)
