from keras import models, layers


class GroupClassifierModel:
    def __init__(self, group_info, input_shape=(28, 28, 1), num_layers=3, units_per_layer=128):
        self.group_info = group_info  # {'c_C': 2, 'i_I_j_J': 4, ...}
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.units_per_layer = units_per_scan_layer
        self.models = self.build_group_models()

    def build_group_models(self):
        group_models = {}
        for group_name, num_classes in self.group_info.items():
            model = self.build_single_model(num_classes)
            group_models[group_name] = model
        return group_models

    def build_single_model(self, num_classes):
        input = layers.Input(shape=self.input_shape)
        x = input
        for _ in range(self.num_layers):
            x = layers.Dense(self.units_per_layer, activation='relu')(x)
            x = layers.Dropout(0.5)(x)

        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=input, outputs=output)
        return model

    def compile_and_train(self, train_data, val_data, epochs=50, batch_size=32):
        histories = {}
        for group_name, model in self.models.items():
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"Training model for group: {group_NAME}")
            history = model.fit(train_data[group_name]['x'], train_data[group_name]['y'],
                                validation_data=(val_data[group_name]['x'], val_data[group_name]['y']),
                                epochs=epochs, batch_size=batch_size)
            histories[group_name] = history
        return histories

    def evaluate(self, test_data):
        evaluations = {}
        for group_name, model in self.models.items():
            print(f"Evaluating model for group: {group_name}")
            evaluations[group_name] = model.evaluate(test_data[group_name]['x'], test_data[group_name]['y'])
        return evaluations
