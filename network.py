
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras import models, layers
import pandas as pd
from tqdm.auto import tqdm

from models.lenet import LeNetModel
from models.resnet50 import ResNetModel

# Load data
train_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-train.csv'
test_file_path = '../../../../mnt/sda/suhohan/emnist/emnist-byclass-test.csv'

chunk_size = 10000
train_data_iter = pd.read_csv(train_file_path, chunksize=chunk_size)
train_data = pd.concat([chunk for chunk in tqdm(train_data_iter, desc='Loading training data')])
test_data_iter = pd.read_csv(test_file_path, chunksize=chunk_size)
test_data = pd.concat([chunk for chunk in tqdm(test_data_iter, desc='Loading test data')])

print(train_data.shape, test_data.shape)

# 데이터의 차원과 크기를 정확히 파악
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

# 데이터 준비
x_train = train_data.iloc[:, 1:].to_numpy().reshape((num_train_samples, 28, 28, 1))
x_test = test_data.iloc[:, 1:].to_numpy().reshape((num_test_samples, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(train_data.iloc[:, 0], 62)  # 클래스 수는 EMNIST ByClass 기준 62개
y_test = tf.keras.utils.to_categorical(test_data.iloc[:, 0], 62)

# ResNet을 위한 데이터 준비
x_train_resized = tf.image.resize(x_train, [32, 32])
x_test_resized = tf.image.resize(x_test, [32, 32])
x_train_rgb = tf.repeat(x_train_resized, 3, axis=3)
x_test_rgb = tf.repeat(x_test_resized, 3, axis=3)

# 콜백 설정
checkpoint_path_lenet = f"./checkpoints_lenet/weights.{time.time()}.hdf5"
checkpoint_path_resnet = f"./checkpoints_resnet/weights.{time.time()}.hdf5"


# 모델 생성 및 훈련
lenet_model = LeNetModel()
resnet_model = ResNetModel()
lenet_model.compile()
resnet_model.compile()
lenet_model.train(x_train, y_train, validation_data=(x_test, y_test), batch_size=1000)
resnet_model.train(x_train_rgb, y_train, validation_data=(x_test_rgb, y_test), batch_size=1000)

# 모델 평가
loss_lenet, acc_lenet = lenet_model.evaluate(x_test, y_test)
loss_resnet, acc_resnet = resnet_model.evaluate(x_test_rgb, y_test)

# 결과 표시
results = {"Model": ["LeNet-5", "ResNet-50"], "Loss": [loss_lenet, loss_resnet], "Accuracy": [acc_lenet, acc_resnet]}
results_df = pd.DataFrame(results)