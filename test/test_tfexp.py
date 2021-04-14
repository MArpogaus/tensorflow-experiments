import tfexp
import pytest

from tfexp.configuration import Configuration

import tensorflow as tf


cfg = Configuration(
    seed=42,
    name="mnist",
    model=tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ]
    ),
    data_loader=tf.keras.datasets.mnist.load_data,
    compile_kwds={
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "metrics": ["accuracy"],
        "optimizer": "adam",
    },
    fit_kwds={"batch_size": 32, "epochs": 2, "shuffle": True, "validation_split": 0.1},
)


densenet_cfg_yaml_template_str = """
seed: &seed 42

mlflow:
  log_artifacts: $base_path
  log_params:
    hidden_units: &units {units}

name: &name !join [ mnist-, *units]
model_checkpoints: !join [ !store [base_path, !abspathjoin [ './logs/', *name]], /mcp]


model: !!python/object/apply:tensorflow.keras.models.Sequential
  args:
    - !!python/list
      - !!python/object/apply:tensorflow.keras.layers.Flatten
        kwds: {{input_shape: !!python/tuple [28, 28]}}
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [*units]
        kwds: {{activation: relu}}
      - !!python/object/apply:tensorflow.keras.layers.Dropout
        args: [0.2]
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [10]

data_loader: !!python/name:tensorflow.python.keras.datasets.mnist.load_data

compile_kwds:
  loss: !!python/object/apply:tensorflow.python.keras.losses.SparseCategoricalCrossentropy
    kwds: {{from_logits: true}}
  metrics: [accuracy]
  optimizer: adam

fit_kwds:
  epochs: 2
  batch_size: 32
  shuffle: True
  validation_split: 0.1
  callbacks:
    - !!python/object/apply:tensorflow.keras.callbacks.ModelCheckpoint
      kwds:
        filepath: !join [$base_path, !datetime '/mcp/{{now:%Y%m%d-%H%M%S}}']
        monitor:  val_loss
        mode: auto
        verbose: 0
        save_best_only: true
        save_weights_only: true
    - !!python/object/apply:tensorflow.keras.callbacks.CSVLogger
        kwds:
          filename: !join [$base_path, !datetime '/{{now:%Y%m%d-%H%M%S}}.log']
          append: false
"""


@pytest.fixture(scope="session")
def cfgs(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("cfgs")
    for units in range(32, 64, 128):
        yaml = densenet_cfg_yaml_template_str.format(units=units)
        with open(tmpdir / f"mnist-{units}.yaml", "w+") as f:
            f.write(yaml)
    return tmpdir


def test_train():
    tfexp.train(cfg)


def test_test():
    tfexp.test(cfg)


def test_batch_train(cfgs):
    tfexp.train(cfgs)


def test_batch_test(cfgs):
    tfexp.test(cfgs)
