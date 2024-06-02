from keras import models, layers, optimizers, callbacks


class ResModel:
    def __init__(self, num_classes=62, initial_filters=32, dropout_rate=0.3, final_dropout_rate=0.6, activation='ReLU', num_residual_units=3):
        self.num_classes = num_classes
        self.initial_filters = initial_filters
        self.dropout_rate = dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.activation = activation
        self.num_residual_units = num_residual_units
        self.model = self.build_model()

    def residual_block(self, x, filters, kernel_size=3, stride=1):
        y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = self._get_activation()(y)
        y = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)

        if stride != 1 or x.shape[-1] != filters:
            x = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)

        out = layers.Add()([x, y])
        out = self._get_activation()(out)
        return out

    def build_model(self):
        input = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(self.initial_filters, (3, 3), padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = self._get_activation()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)

        filters = self.initial_filters * 2
        for _ in range(self.num_residual_units):
            x = self.residual_block(x, filters)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
            filters *= 2

        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = self._get_activation()(x)
        x = layers.Dropout(self.final_dropout_rate)(x)

        output = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=input, outputs=output)

        return model

    def _get_activation(self):
        if self.activation == 'swish':
            return layers.Activation('swish')
        elif self.activation == 'prelu':
            return layers.PReLU(shared_axes=[1, 2])
        else:
            return layers.Activation(self.activation)

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None, class_weight=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data,
                              batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                              class_weight=class_weight)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def summary(self):
        return self.model.summary()
