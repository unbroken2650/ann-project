import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras import models, layers

class midModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(filters=6, kernel_size=3, strides=1, activation='tanh',
                                   input_shape=[28, 28, 1], padding='same'),
            layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),
            layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='tanh', padding='valid'),
            layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),
            layers.Conv2D(filters=120, kernel_size=3, strides=1, activation='tanh', padding='valid'),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(62, activation="softmax")
        ])
        return model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer, loss, metrics)

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def summary(self):
        return self.model.summary()
