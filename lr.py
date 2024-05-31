# %% [markdown]
# # ann-project
#
# 5조
# 한수호, 강구현, 김민규, 홍준기

# %% [markdown]
# # Setup

# %%
from datetime import datetime
from models.simple_cnn import CNNModel
from models.residual import ResModel
from models.resnet50 import ResNetModel
from models.lenet import LeNetModel
from keras import callbacks, optimizers, layers
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
import pickle
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# %%
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# %%
result_path = './results'
_time = datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S')

# %% [markdown]
# dataset

# %%
train_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-train.csv'
test_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Data dimensions and sizes
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

# Prepare data
x_train = train_data.iloc[:, 1:].to_numpy().reshape((num_train_samples, 28, 28, 1))
x_test = test_data.iloc[:, 1:].to_numpy().reshape((num_test_samples, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(train_data.iloc[:, 0], 62)  # 62 classes for EMNIST ByClass
y_test = tf.keras.utils.to_categorical(test_data.iloc[:, 0], 62)

# Integer labels for sparse categorical crossentropy
y_train_int = train_data.iloc[:, 0].to_numpy()
y_test_int = test_data.iloc[:, 0].to_numpy()

# Split the training data into training and validation sets
_, _, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_valid, y_train_int, y_valid_int = train_test_split(x_train, y_train_int, test_size=0.1, random_state=42)

# Prepare data for ResNet
x_train_resized = tf.image.resize(x_train, [32, 32])
x_valid_resized = tf.image.resize(x_valid, [32, 32])
x_test_resized = tf.image.resize(x_test, [32, 32])

# %% [markdown]
# checkpoints

# %%


def create_callbacks(model_name):
    current_time = int(time.time())
    checkpoint_path = f"./checkpoints/checkpoints_{model_name}/weights.{current_time}.hdf5"
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                             min_delta=0.0001, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    return [early_stopping, reduce_lr, checkpoint]


callbacks_lenet = create_callbacks('lenet')
callbacks_resnet = create_callbacks('resnet')
callbacks_ours_1 = create_callbacks('ours_1')
callbacks_ours_2 = create_callbacks('ours_2')
callbacks_final = create_callbacks('final')
result_path = './results'

# %% [markdown]
# test results

# %%


def save_results(results_df):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    results_df.to_csv(os.path.join(result_path, f'result_{_time}.csv'), index=False)

# %% [markdown]
# saving history

# %%


def save_history(histories, filename):
    with open(os.path.join(result_path, filename), 'w') as f:
        json.dump(histories, f)

# %% [markdown]
# ## 2.4. Learning Rate


# %%
history = []
training_time = []

# %%
callbacks_final = create_callbacks('lr')

# %%
best_optimizer_1 = optimizers.legacy.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
best_optimizer_2 = optimizers.legacy.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
best_optimizer_3 = optimizers.legacy.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
best_optimizer_4 = optimizers.legacy.Nadam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
best_optimizer_5 = optimizers.legacy.Nadam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999)
best_activation = layers.Activation('swish')

# %%
our_model_1lr = ResModel(num_residual_units=3, activation=best_activation)
our_model_2lr = ResModel(num_residual_units=3, activation=best_activation)
our_model_3lr = ResModel(num_residual_units=3, activation=best_activation)
our_model_4lr = ResModel(num_residual_units=3, activation=best_activation)
our_model_5lr = ResModel(num_residual_units=3, activation=best_activation)

# %%
our_model_1lr.compile(optimizer=best_optimizer_1, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
our_model_2lr.compile(optimizer=best_optimizer_2, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
our_model_3lr.compile(optimizer=best_optimizer_3, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
our_model_4lr.compile(optimizer=best_optimizer_4, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
our_model_5lr.compile(optimizer=best_optimizer_5, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
EPOCHS = 50
BATCH_SIZE = 500

# %%
results = {'Learning Rate': ['1e-2', '1e-3', '1e-4', '1e-5', '1e-6'], 'Loss': [], 'Accuracy': [], 'Training Time': []}

# %%


def _train_lr(model, optimizer, x_train, y_train, validation_data):
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    hist = model.train(x_train, y_train, validation_data=validation_data, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, callbacks=[callbacks_final])
    end_time = time.time()

    history.append(hist)

    loss, accuracy = model.evaluate(x_test, y_test_int)
    results['Loss'].append(loss)
    results['Accuracy'].append(accuracy)
    results['Training Time'].append(end_time - start_time)


# %%
start_time = time.time()
_train_lr(our_model_1lr, best_optimizer_1, x_train, y_train_int, (x_valid, y_valid_int))
end_time = time.time()
training_time.append(end_time-start_time)
callbacks_final = create_callbacks('lr_1')

start_time = time.time()
_train_lr(our_model_2lr, best_optimizer_2, x_train, y_train_int, (x_valid, y_valid_int))
end_time = time.time()
training_time.append(end_time-start_time)
callbacks_final = create_callbacks('lr_2')


start_time = time.time()
_train_lr(our_model_3lr, best_optimizer_3, x_train, y_train_int, (x_valid, y_valid_int))
end_time = time.time()
training_time.append(end_time-start_time)
callbacks_final = create_callbacks('lr_3')

start_time = time.time()
_train_lr(our_model_4lr, best_optimizer_4, x_train, y_train_int, (x_valid, y_valid_int))
end_time = time.time()
training_time.append(end_time-start_time)
callbacks_final = create_callbacks('lr_4')

start_time = time.time()
_train_lr(our_model_5lr, best_optimizer_5, x_train, y_train_int, (x_valid, y_valid_int))
end_time = time.time()
training_time.append(end_time-start_time)
callbacks_final = create_callbacks('lr_5')

# %%
results_df = pd.DataFrame(results)
results_df.to_csv(f'{result_path}/result_lr_{_time}.csv', index=False)

# %%
for i, h in zip(range(1, 4), history):
    history_dict = {key: list(map(float, value)) for key, value in h.history.items()}  # Ensure values are float
    save_history(history_dict, f'history_lr_{i}_{_time}.json')

# %%
colors = ["red", "blue", "green", "purple", "gold", "orange"]

plt.figure(figsize=(12, 6))

plt.subplot(121)
for idx, hist in enumerate(history):
    plt.plot(hist.history['loss'], label=f'{results["Learning Rate"][idx]}', color=colors[idx])
    plt.title(f'Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, EPOCHS])
    plt.grid(True)
    plt.legend()
plt.subplot(122)
for idx, hist in enumerate(history):
    plt.plot(hist.history['accuracy'], label=f'{results["Learning Rate"][idx]}', color=colors[idx])
    plt.title(f'Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, EPOCHS])
    plt.ylim([0.8, 1])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig(f"{result_path}/result_lr_{_time}.png")
plt.show()

# %%
metrics = ["Loss", "Accuracy", "Training Time"]
for metric in metrics:
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Learning Rate", y=metric, hue="Learning Rate", data=results_df, palette="viridis", legend=False)
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xlabel('Learning Rate')
    plt.tight_layout()
    plt.savefig(f'{result_path}/{metric.lower().replace(" ", "_")}_comparison_lr_{_time}.png')
    plt.show()

# %% [markdown]
# ## 2.5. LR Scheduler

# %% [markdown]
# ## 2.6. Loss function
