import yaml
import numpy as np
import datetime as dt


def date_time_path_formatter(loader, node):
    value = loader.construct_scalar(node)
    now = dt.datetime.now()
    return value.format(now=now)


def np_product(loader, node):
    value = loader.construct_sequence(node)
    return np.product(value)


def np_sum(loader, node):
    value = loader.construct_sequence(node)
    return np.sum(value)


def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def include(loader, node):
    path = loader.construct_scalar(node)
    with open(path, 'r') as yf:
        ref = yaml.load(yf, Loader=loader.__class__)

    return ref


yaml.add_constructor('!datetime_path', date_time_path_formatter)
yaml.add_constructor('!product', np_product)
yaml.add_constructor('!sum', np_sum)
yaml.add_constructor('!join', join)
yaml.add_constructor('!include', include)
