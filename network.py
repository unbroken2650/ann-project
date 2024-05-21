import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import time
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from keras import callbacks
import time
from models.lenet import LeNetModel
from models.resnet50 import ResNetModel
from models.ours import CNNModel

BATCH_SIZE = 500
EPOCHS = 50

# Load data
train_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-train.csv'
test_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-test.csv'

chunk_size = 10000
train_data_iter = pd.read_csv(train_file_path, chunksize=chunk_size)
train_data = pd.concat([chunk for chunk in tqdm(train_data_iter, desc='Loading training data')])
test_data_iter = pd.read_csv(test_file_path, chunksize=chunk_size)
test_data = pd.concat([chunk for chunk in tqdm(test_data_iter, desc='Loading test data')])

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

# Callbacks for checkpoints and learning rate reduction
checkpoint_path_lenet = f"./checkpoints_lenet/weights.{int(time.time())}.hdf5"
checkpoint_path_resnet = f"./checkpoints_resnet/weights.{int(time.time())}.hdf5"
checkpoint_path_cnn = f"./checkpoints_cnn/weights.{int(time.time())}.hdf5"
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
checkpoint_lenet = callbacks.ModelCheckpoint(filepath=checkpoint_path_lenet, monitor='val_loss', save_best_only=True, verbose=1)
checkpoint_resnet = callbacks.ModelCheckpoint(filepath=checkpoint_path_resnet, monitor='val_loss', save_best_only=True, verbose=1)
checkpoint_cnn = callbacks.ModelCheckpoint(filepath=checkpoint_path_cnn, monitor='val_loss', save_best_only=True, verbose=1)

# Initialize models
lenet_model = LeNetModel()
resnet_model = ResNetModel()
cnn_model = CNNModel()

# Compile models
lenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train models
history = []
training_time=[]

start_time = time.time()
history.append(lenet_model.train(x_train, y_train_int, validation_data=(x_valid, y_valid_int), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reduce_lr, checkpoint_lenet]))
end_time = time.time()
training_time.append(end_time-start_time)

start_time = time.time()
history.append(resnet_model.train(x_train_resized, y_train_int, validation_data=(x_valid_resized, y_valid_int), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reduce_lr, checkpoint_resnet]))
end_time = time.time()
training_time.append(end_time-start_time)

start_time = time.time()
history.append(cnn_model.train(x_train, y_train_int, validation_data=(x_valid, y_valid_int), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reduce_lr, checkpoint_cnn]))
end_time = time.time()
training_time.append(end_time-start_time)


# Evaluate models
loss_lenet, acc_lenet = lenet_model.evaluate(x_test, y_test_int)
loss_resnet, acc_resnet = resnet_model.evaluate(x_test, y_test_int)
loss_cnn, acc_cnn = cnn_model.evaluate(x_test, y_test_int)

# Results DataFrame
results = {
    "Model": ["LeNet-5", "ResNet-50", "CNN"],
    "Loss": [loss_lenet, loss_resnet, loss_cnn],
    "Accuracy": [acc_lenet, acc_resnet, acc_cnn],
    "Training Time": training_time
}
result_path = './results'
results_df = pd.DataFrame(results)
results_df.to_csv(f'{result_path}/result{time.time()}.csv', index=False)

colors = ["red", "blue", "green"]

plt.figure(figsize=(10, 5))

plt.subplot(121)
for idx, hist in enumerate(history):
    plt.plot(hist.history['loss'], label=f'{results["Model"][idx]}', color=colors[idx])
    plt.title(f'Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, EPOCHS])
    plt.grid(True)
    plt.legend()
plt.subplot(122)
for idx, hist in enumerate(history):
    plt.plot(hist.history['accuracy'], label=f'{results["Model"][idx]}', color=colors[idx])
    plt.title(f'Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, EPOCHS])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig(f"{result_path}/result{time.time()}.png")