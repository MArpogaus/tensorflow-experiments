# TensorFlow Experiments

A simple framework to run
[TensorFlow](https://tensorflow.org)
experiments described in
[YAML](https://yaml.org/)
files.

## Install

**The packages is not released to PyPI, but feel free to clone it from this repo if your interested**

Use the
[pip package manger](https://www.tensorflow.org/install/pip)
to install the

```
$ pip install git+https://github.com/MArpogaus/tensorflow-experiments.git
```

## MNIST Example

in `./examples` you can find the minimal working example using the
[MNIST dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)
of handwrite digits

```yaml
seed: 42

model: !!python/object/apply:tensorflow.keras.models.Sequential
  args:
    - !!python/list
      - !!python/object/apply:tensorflow.keras.layers.Flatten
        kwds: {input_shape: !!python/tuple [28, 28]}
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [128]
        kwds: {activation: relu}
      - !!python/object/apply:tensorflow.keras.layers.Dropout
        args: [0.2]
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [10]

data: !!python/name:tensorflow.python.keras.datasets.mnist

compile_kwds:
  loss: !!python/object/apply:tensorflow.python.keras.losses.SparseCategoricalCrossentropy
    kwds: {from_logits: true}
  metrics: [accuracy]
  optimizer: adam

fit_kwds:
  epochs: 5
  shuffle: True
  validation_split: 0.1

```

## License

[Apache License 2.0](LICENSE)
