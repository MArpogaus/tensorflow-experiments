# -*- coding: utf-8 -*-
# @Author: marcel
# @Date:   2020-03-17 19:22:35
# @Last Modified by:   marcel
# @Last Modified time: 2020-03-17 19:23:01
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
]
