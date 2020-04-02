import yaml
import datetime as dt

yaml_tag = u'!path_with_ts'


def DateTimePathFormatter(path):
    now = dt.datetime.now()
    return path.format(now=now)


def constructor(loader, node):
    value = loader.construct_scalar(node)
    return DateTimePathFormatter(value)


yaml.add_constructor(yaml_tag, constructor)
