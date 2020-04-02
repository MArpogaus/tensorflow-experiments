import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from .configuration import Configuration

from . import utils


def train(args):
    cfg = Configuration.from_yaml(args.config)
    print(cfg)

    # Setting seed to ensure reproducibility
    tf.random.set_seed(cfg.seed)

    data = cfg.data.load_data(**cfg.data_kwds)
    if type(data) is tuple:
        train_x, train_y = data[0]
        test_x, test_y = data[1]
        fit_kwds = dict(x=train_x,
                        y=train_y,
                        validation_data=(test_x, test_y))
    elif isinstance(data, (dataset_ops.DatasetV1,
                           dataset_ops.DatasetV2,
                           iterator_ops.Iterator)):
        fit_kwds = dict(x=data)
    else:
        raise ValueError(f"Dataset type '{type(data)}' unsupported")

    model = cfg.model
    # cfg.compile_kwds['loss']=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(**cfg.compile_kwds)

    # from sklearn.model_selection import TimeSeriesSplit

    #     cv = TimeSeriesSplit(n_splits=n_splits)
    #     for i, (train_index, test_index) in enumerate(cv.split(X=x)):
    model.fit(**fit_kwds, **cfg.fit_kwds)


def predict():
    pass
