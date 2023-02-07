import pytest
import tensorflow as tf

import tfexp
from tfexp.configuration import Configuration

cfg = Configuration(
    seed=42,
    name="mnist",
    model=tf.keras.Sequential,
    model_kwds=dict(
        layers=[
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

name: &name !join [ mnist-, &units {units}]
model_checkpoints: !pathjoin [ !store [base_path, !abspathjoin [ 'logs', *name]], mcp]

mlflow:
  log_artifacts: $base_path
  log_params:
    hidden_units: *units

model: !!python/name:tensorflow.keras.models.Sequential
model_kwds:
  layers: !!python/list
      - !!python/object/apply:tensorflow.keras.layers.Flatten
        kwds: {{input_shape: !!python/tuple [28, 28]}}
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [*units]
        kwds: {{activation: relu}}
      - !!python/object/apply:tensorflow.keras.layers.Dropout
        args: [0.2]
      - !!python/object/apply:tensorflow.keras.layers.Dense
        args: [10]

data_loader: !!python/name:keras.api._v2.keras.datasets.mnist.load_data

compile_kwds:
  loss:
    !!python/object/apply:tensorflow.python.keras.losses.SparseCategoricalCrossentropy
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
        filepath: !pathjoin [$base_path, !datetime 'mcp/{{now:%Y%m%d-%H%M%S}}']
        monitor:  val_loss
        mode: auto
        verbose: 0
        save_best_only: true
        save_weights_only: true
    - !!python/object/apply:tensorflow.keras.callbacks.CSVLogger
        kwds:
          filename: !pathjoin [$base_path, !datetime '{{now:%Y%m%d-%H%M%S}}.log']
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


def test_fit():
    tfexp.fit(cfg)


def test_evaluate():
    tfexp.evaluate(cfg)


def test_batch_fit(cfgs):
    tfexp.fit(cfgs)


def test_batch_evaluate(cfgs):
    tfexp.evaluate(cfgs)
