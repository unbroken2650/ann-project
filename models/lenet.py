
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras import models, layers

class LeNetModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh',
                                   input_shape=[28, 28, 1], padding='same'),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),
            tf.keras.layers.Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh', padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(62, activation="softmax")
        ])
        return model

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
