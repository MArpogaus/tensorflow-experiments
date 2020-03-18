import argparse
import tensorflow as tf
from .configuration import Configuration


def f_train(args):

    #cfg = Configuration.from_yaml(args.config)
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    cfg = Configuration(
        experiment_name='MNIST',
        model=tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ]),
        # model_args,
        # model_kwargs,
        model_compile_kwargs=dict(optimizer='adam',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                      from_logits=True),
                                  metrics=['accuracy']),
        model_fit_kwargs=dict(x=x_train, y=y_train, epochs=5),
        save_path='./logs')
    print(cfg)

    model = cfg.model

    model.compile(**cfg.model_compile_kwargs)

    model.fit(**cfg.model_fit_kwargs)


def f_predict():
    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Help me to conduct my experiments')
    subparsers = p.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train the model')
    # Add specific options for option1 here, but here's
    train_parser.add_argument('config', type=argparse.FileType(mode='r'))
    # train_parser.add_argument('--model', type=tf.keras.Model)
    # train_parser.add_argument('--experiment_name', type=str)
    # train_parser.add_argument('--model_args', type=list)
    # train_parser.add_argument('--model_kwargs', type=dict)
    # train_parser.add_argument('--model_compile_kwargs', type=dict)
    # train_parser.add_argument('--model_fit_kwargs', type=dict)
    # train_parser.add_argument('--save_path', type=str)
    # an example
    train_parser.set_defaults(func=f_train)

    predict_parser = subparsers.add_parser('predict')
    # Add specific options for option1 here
    predict_parser.set_defaults(func=f_predict)

    args = p.parse_args()
    args.func(args)
