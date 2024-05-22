from keras import models, layers

class CNNModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(62, activation='softmax')  # 47 classes in EMNIST Balanced
        ])
        return model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer, loss, metrics)

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None, class_weight=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weight)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def summary(self):
        return self.model.summary()
