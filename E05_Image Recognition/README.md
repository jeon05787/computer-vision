## 01_간단한 이미지 분류기 구현

### 설명
*  손글씨 숫자 이미지 (MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현

### 요구사항

*  MNIST 데이터셋을로드
* 데이터를 훈련세트와 테스트 세트로 분할
*  간단한 신경망 모델을 구축
*  모델을 훈련시키고 정확도를 평가
*  tensorflow.keras.datasets에서 MNIST 데이터셋을 불러올 수 있음
*  ​Sequential 모델과 Dense 레이어를 활용하여 신경망을 구성
*  손글씨 숫자 이미지는 28x28 픽셀크기의 흑백이미지

### 전체코드
```python
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

```

### 주요코드
- **`(x_train, y_train), (x_test, y_test) = mnist.load_data()`**:  MNIST 데이터셋 로
  

-  **`(x_train, y_train), (x_test, y_test) = mnist.load_data()`**: 데이터 전처리, 0~255 범위의 픽셀 값을 0과 1 사이로 정규화합니다.
  

-  **`model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 배열로 변환
    Dense(128, activation='relu'),  # 첫 번째 Dense 레이어, 128개의 뉴런
    Dense(10, activation='softmax')  # 출력 레이어, 10개의 클래스(0-9)
])`**:  신경망 모델 구축, Sequential 모델을 사용하여 간단한 신경망을 구성합니다, Flatten: 2D 이미지를 1D로 변환합니다. Dense: Fully connected layer로, 첫 번째 레이어는 128개의 뉴런, 두 번째 레이어는 10개의 뉴런(출력 클래스)입니다.

- **`model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])`**:  Adam optimizer와 sparse categorical crossentropy 손실 함수를 사용하여 모델을 컴파일합니다.
정확도(accuracy)를 평가 지표로 설정합니다

-  **`model.fit(x_train, y_train, epochs=5)`**:  훈련 데이터를 사용하여 5 에포크 동안 모델을 학습시킵니다.
-  **`test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")`**:  테스트 데이터로 모델을 평가하고, 테스트 정확도를 출력합니다.
### 실행결과
<img width="706" height="274" alt="image" src="https://github.com/user-attachments/assets/75041a9e-d56c-442f-a345-014ebd0d4099" />


## 02_CIFAR-10 데이터셋을 활용한 CNN 모델 구축

### 설명

* CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행

### 요구사항

* CIFAR-10 데이터셋을 로드
* 데이터 전처리(정규화 등)를 수행
* CNN 모델을 설계하고 훈련
* 모델의 성능을 평가하고, 테스트 이미지(dog.jpg)에 대한 예측을 수행
* tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 불러올 수 있음
* Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN을 구성
* 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있음


### 전체코드
```python
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

    # 결과 출력
    print(f"\n dog.jpg 예측 결과: {CLASS_NAMES[pred_class]} ({confidence:.1f}%)")
else:
    print(f"\n '{IMG_PATH}' 파일을 찾을 수 없습니다.")
    print("   dog.jpg를 코드와 같은 폴더에 넣어주세요.")


```

### 주요코드
- **`(x_train, y_train), (x_test, y_test) = cifar10.load_data()`**: CIFAR-10 데이터셋을 로드하여 훈련 데이터와 테스트 데이터를 나누어 불러옵니다.
  
-  **`x_train = x_train / 255.0
x_test  = x_test / 255.0`**:  훈련 데이터와 테스트 데이터의 픽셀 값을 0과 1 사이로 정규화하여 모델 훈련 속도를 높입니다.
  
-  **`CNN 모델 구축`**:  Conv2D, MaxPooling2D, Dropout 등을 이용하여 CNN 모델을 설계합니다.
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
```

- **`model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`**:  Adam optimizer와 sparse categorical crossentropy 손실 함수를 사용하여 모델을 컴파일함, 모델의 정확도(accuracy)를 평가 지표로 설정합니다.

-  **`모델 훈련`**: 10 에포크 동안 훈련 데이터를 사용하여 모델을 학습시킵니다.
검증 데이터는 **훈련 데이터의 10%**를 사용합니다.
```python
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=2 
)
```
- **`dog.jpg 예측`**:  dog.jpg 이미지를 CIFAR-10 크기로 리사이즈한 후 정규화하고 예측을 수행합니다.
예측된 클래스와 확신도를 출력합니다.
```python
img = image.load_img(IMG_PATH, target_size=(32, 32))  # CIFAR-10 크기로 리사이즈
img_arr = image.img_to_array(img) / 255.0             # 정규화
img_input = np.expand_dims(img_arr, axis=0)           # (1, 32, 32, 3)
pred = model.predict(img_input, verbose=0)
pred_class = np.argmax(pred)
confidence = pred[0][pred_class] * 100
```

### 실행결과
<img width="956" height="560" alt="스크린샷 2026-04-02 152330" src="https://github.com/user-attachments/assets/555d516f-6376-45ec-b31d-850b7bec7eb9" />
