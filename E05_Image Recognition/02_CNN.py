import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CIFAR-10 클래스 이름
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# 3. CNN 모델 구축
model = Sequential([
    # 합성곱 블록 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),   # 32×32 → 16×16

    # 합성곱 블록 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),   # 16×16 → 8×8

    # 완전 연결층
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax'),  # 10개 클래스 출력
])

# 4. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 모델 훈련
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=2 
)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# 7. 테스트 결과 출력
print("=" * 45)
print("테스트 결과")
print(f"  테스트 손실(Loss): {test_loss:.4f}")
print(f"  테스트 정확도:     {test_acc * 100:.2f}%")
print("=" * 45)

# 8. dog.jpg 예측 
IMG_PATH = 'image/dog.jpg'

if os.path.exists(IMG_PATH):
    from keras.preprocessing import image
    # 이미지 로드 및 전처리
    img = image.load_img(IMG_PATH, target_size=(32, 32))  # CIFAR-10 크기로 리사이즈
    img_arr = image.img_to_array(img) / 255.0             # 정규화
    img_input = np.expand_dims(img_arr, axis=0)           # (1, 32, 32, 3)

    # 예측
    pred = model.predict(img_input, verbose=0)
    pred_class = np.argmax(pred)
    confidence = pred[0][pred_class] * 100

    # 결과 출력 (예측된 클래스와 신뢰도를 출력)
    print(f"\n dog.jpg 예측 결과: {CLASS_NAMES[pred_class]}")
    print(f"신뢰도: {confidence:.1f}%")
else:
    print(f"\n '{IMG_PATH}' 파일을 찾을 수 없습니다.")
    print("   dog.jpg를 코드와 같은 폴더에 넣어주세요.")
