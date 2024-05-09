import numpy as np
import tensorflow as tf

# test_data를 불러오는 코드 (이미 위에서 로드했다고 가정)
# 예를 들어, test_data = pd.read_csv('path_to_test_data.csv')

# 첫 번째 이미지 선택
sample_image = test_data.iloc[0, 1:].values  # 첫 번째 열은 레이블이므로 제외
sample_label = test_data.iloc[0, 0]

# 이미지 데이터를 28x28x1 형태로 변형 (LeNet 모델을 위한 형태)
sample_image = sample_image.reshape(1, 28, 28, 1)  # (1, 28, 28, 1) -> 첫 번째 차원은 배치 크기

# 데이터 타입 변경 및 정규화
sample_image = sample_image.astype('float32') / 255.0

# 모델 로드
model = tf.keras.models.load_model('./path_to_your_model.h5')

# 모델 예측
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions, axis=1)

# 결과 출력
print("Actual label:", sample_label)
print("Predicted class:", predicted_class)
print("Prediction probabilities:", predictions)
