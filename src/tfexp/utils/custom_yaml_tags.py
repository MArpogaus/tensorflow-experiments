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


yaml.add_constructor('!datetime_path', date_time_path_formatter)
yaml.add_constructor('!product', np_product)
yaml.add_constructor('!sum', np_sum)
