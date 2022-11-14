import logging
import os

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2
from tensorflow.python.data.ops.iterator_ops import Iterator


def set_seed(seed):
    tf.random.set_seed(seed)


def prepare_data(data):
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


def compile_model(model, **compile_kwds):
    model.compile(**compile_kwds)

    return model


def load_checkpoint(model, model_checkpoints):
    # ref: https://www.tensorflow.org/tutorials/keras/save_and_load
    if os.path.exists(model_checkpoints):
        if os.path.isfile(model_checkpoints):
            cp = model_checkpoints
        else:
            # get path to latest checkpoint
            cp = tf.train.latest_checkpoint(model_checkpoints)

        # Load model
        if cp is not None:
            logging.info(f"restoring model from checkpoint {cp}")
            model.load_weights(cp)
    elif model_checkpoints != "":
        os.makedirs(model_checkpoints)

    return model


def fit(model, data, **fit_kwds):
    if isinstance(data["train"], tuple):
        # Data format: (train_x, train_y)
        fit_kwds.update(
            dict(
                x=data["train"][0], y=data["train"][1], validation_data=data["validate"]
            )
        )
    else:
        # Data format: dataset / generator or unsupervised
        fit_kwds.update(dict(x=data["train"], validation_data=data["validate"]))

    # TRAIN ###################################################################
    history = model.fit(**fit_kwds)

    return history


def evaluate(model, data, dataset="test", **evaluate_kwds):
    if isinstance(data[dataset], tuple):
        # Data format: (test_x, test_y)
        evaluate_kwds.update(dict(x=data[dataset][0], y=data[dataset][1]))
    else:
        # Data format: dataset / generator or unsupervised
        evaluate_kwds.update(dict(x=data[dataset]))

    evaluate_kwds["return_dict"] = evaluate_kwds.get("return_dict", True)
    # EVALUATE ################################################################
    loss = model.evaluate(**evaluate_kwds)

    return loss
