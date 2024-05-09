
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

class ResNetModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        input_tensor = Input(shape=(32, 32, 3))
        base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        output = Dense(62, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, validation_data, epochs=50, batch_size=300, callbacks=None):
        return self.model.fit(x_train, y_train, validation_data=validation_data,batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
