import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras import models, layers,optimizers, callbacks


class CNNModel:
    def __init__(self):
        self.model = self.build_model()
        
    def residual_block(self, x, filters, kernel_size=3, stride=1):
        y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)
        y = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)

        if stride != 1 or x.shape[-1] != filters:
            x = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)

        out = layers.Add()([x, y])
        out = layers.PReLU()(out)
        return out
    
    def build_model(self):
        input = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(0.33)(x)
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)

        x = self.residual_block(x, 64)
        x = layers.Dropout(0.33)(x)
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)

        x = self.residual_block(x, 128)
        x = layers.Dropout(0.33)(x)
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)

        x = self.residual_block(x, 256)
        x = layers.Dropout(0.33)(x)
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(0.6)(x)

        output = layers.Dense(62, activation='softmax')(x)

        model = models.Model(inputs=input, outputs=output)
        
        return model
    
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer, loss, metrics)

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
