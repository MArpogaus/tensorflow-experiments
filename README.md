# TensorFlow Experiments

A straight-forward framework to run
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
experiment_name: MNIST
model: !!python/name:mnist_model.model
data: !!python/name:tensorflow.python.keras.datasets.mnist
compile_kwargs:
  loss: !!python/object/apply:tensorflow.python.keras.losses.SparseCategoricalCrossentropy
    kwds: {from_logits: true}
  metrics:
  - accuracy
  optimizer: adam
fit_kwargs:
  epochs: 5
save_path: ./logs/MNIST
```

## License

[Apache License 2.0](LICENSE)
