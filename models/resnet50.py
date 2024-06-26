from keras import models, layers
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

class ResNetModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        input_tensor = Input(shape=(32, 32, 1))
        base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        output = Dense(62, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer, loss, metrics)


    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None, class_weight=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weight)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def summary(self):
        return self.model.summary()
