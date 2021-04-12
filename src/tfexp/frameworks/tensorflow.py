import os
import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2
from tensorflow.python.data.ops.iterator_ops import Iterator


def set_seed(seed):
    tf.random.set_seed(seed)


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


def train(cfg, model, data):
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


def test(cfg, model, data):
    if isinstance(data["test"], tuple):
        # Data format: (test_x, test_y)
        evaluate_kwds = dict(x=data["test"][0], y=data["test"][1])
    else:
        # Data format: dataset / generator or unsupervised
        evaluate_kwds = dict(x=data["test"])

    # EVALUATE ################################################################
    loss = model.evaluate(**evaluate_kwds, **cfg.evaluate_kwds)

    return loss
