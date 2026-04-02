# 필요한 라이브러리 임포트
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import mnist

# 1. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리: 0~255 범위의 픽셀 값을 0과 1 사이로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 신경망 모델 구축
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 배열로 변환
    Dense(128, activation='relu'),  # 첫 번째 Dense 레이어, 128개의 뉴런
    Dense(10, activation='softmax')  # 출력 레이어, 10개의 클래스(0-9)
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
