from datetime import datetime
from models.simple_cnn import CNNModel
from models.residual import ResModel
from models.resnet50 import ResNetModel
from models.lenet import LeNetModel
from keras import callbacks, optimizers, layers, losses
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import seaborn as sns
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

result_path = './results'
_time = datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S')


train_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-train.csv'
test_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

x_train = train_data.iloc[:, 1:].to_numpy().reshape((num_train_samples, 28, 28, 1))
x_test = test_data.iloc[:, 1:].to_numpy().reshape((num_test_samples, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(train_data.iloc[:, 0], 62)
y_test = tf.keras.utils.to_categorical(test_data.iloc[:, 0], 62)

y_train_int = train_data.iloc[:, 0].to_numpy()
y_test_int = test_data.iloc[:, 0].to_numpy()

_, _, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_valid, y_train_int, y_valid_int = train_test_split(x_train, y_train_int, test_size=0.1, random_state=42)


def create_callbacks(model_name):
    current_time = int(time.time())
    checkpoint_path = f"./checkpoints/checkpoints_{model_name}/weights.{current_time}.hdf5"
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                             min_delta=0.0001, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    return [early_stopping, reduce_lr, checkpoint]


result_path = './results'


def save_results(results_df):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    results_df.to_csv(os.path.join(result_path, f'result_{_time}.csv'), index=False)


def save_history(histories, filename):
    with open(os.path.join(result_path, filename), 'w') as f:
        json.dump(histories, f)


history = []
training_time = []

callbacks_final = create_callbacks('loss')

best_activation = layers.Activation('swish')
best_learning_rate = 2e-4
best_optimizer = optimizers.legacy.Nadam(learning_rate=best_learning_rate, beta_1=0.9, beta_2=0.999)

results = {'Loss Function': ['Cross Entropy Loss', 'Focal Loss'], 'Loss': [], 'Accuracy': [], 'Training Time': []}

EPOCHS = 50
BATCH_SIZE = 500


class CategoricalFocalLoss(losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        alpha = tf.constant(self.alpha, dtype=tf.float32)
        gamma = tf.constant(self.gamma, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)


focal_loss = CategoricalFocalLoss(gamma=2.0, alpha=0.25)

model_default = ResModel(num_residual_units=3, activation=best_activation)
model_focal = ResModel(num_residual_units=3, activation=best_activation)
model_default.compile(optimizer=best_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_focal.compile(optimizer=best_optimizer, loss=focal_loss, metrics=['accuracy'])


def _train_losses(model, x_train, y_train, validation_data, x_test, y_test):
    start_time = time.time()
    hist = model.train(x_train, y_train, validation_data=validation_data, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, callbacks=[callbacks_final])
    end_time = time.time()

    history.append(hist)

    loss, accuracy = model.evaluate(x_test, y_test)
    results['Loss'].append(loss)
    results['Accuracy'].append(accuracy)
    results['Training Time'].append(end_time - start_time)


_train_losses(model_default, x_train, y_train_int, (x_valid, y_valid_int), x_test, y_test_int)
_train_losses(model_focal, x_train, y_train, (x_valid, y_valid), x_test, y_test)

results_df = pd.DataFrame(results)
results_df.to_csv(f'{result_path}/result_losses_{_time}.csv', index=False)

for i, h in enumerate(history):
    history_dict = {key: list(map(float, value)) for key, value in h.history.items()}
    save_history(history_dict, f'history_losses_{i}_{_time}.json')

colors = ["red", "blue", "green", "purple", "gold", "orange", "cyan", "magenta", "brown", "pink"]


plt.figure(figsize=(12, 6))

plt.subplot(121)
for idx, hist in enumerate(history):
    plt.plot(hist.history['loss'], label=f'{results["Loss Function"][idx]}', color=colors[idx])
    plt.title(f'Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, EPOCHS])
    plt.grid(True)
    plt.legend()
plt.subplot(122)
for idx, hist in enumerate(history):
    plt.plot(hist.history['accuracy'], label=f'{results["Loss Function"][idx]}', color=colors[idx])
    plt.title(f'Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, EPOCHS])
    plt.ylim([0.8, 1])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig(f"{result_path}/result_losses_{_time}.png")
plt.show()

metrics = ["Loss", "Accuracy", "Training Time"]
for metric in metrics:
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Loss Function", y=metric, hue="Loss Function", data=results_df, palette="viridis", legend=False)
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xlabel('Loss Function')
    plt.tight_layout()
    plt.savefig(f'{result_path}/{metric.lower().replace(" ", "_")}_comparison_lr_{_time}.png')
    plt.show()
