import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CIFAR-10 클래스 이름 (이미지에 대한 분류 라벨)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 데이터 로드: CIFAR-10 데이터셋을 로드 (학습용 데이터와 테스트용 데이터로 나누어짐)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리: 데이터 정규화 (픽셀 값 범위: 0~255 → 0~1로 변환)
# 이 작업은 모델 학습을 더 효율적으로 하고, 훈련 속도를 빠르게 만듦
x_train = x_train / 255.0  # 학습 데이터의 픽셀 값을 0~1 범위로 정규화
x_test  = x_test / 255.0   # 테스트 데이터의 픽셀 값을 0~1 범위로 정규화

# 3. CNN 모델 구축: Sequential 모델을 사용하여 신경망 층을 순차적으로 쌓아 모델을 구성
model = Sequential([
    # 합성곱 블록 1: 32개의 필터를 사용한 Conv2D 레이어와 MaxPooling2D 레이어
    # Conv2D는 이미지의 특징을 추출하는 합성곱 레이어로, 필터 수와 커널 크기 설정
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)), 
    # 첫 번째 Conv2D 레이어: 32개의 필터, 3x3 크기의 커널, ReLU 활성화 함수, 'same' 패딩
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    # 두 번째 Conv2D 레이어: 32개의 필터, 3x3 크기의 커널, ReLU 활성화 함수, 'same' 패딩
    MaxPooling2D(pool_size=(2, 2)),   # MaxPooling2D: 2x2 풀링, 출력 크기 32×32 → 16×16

    # 합성곱 블록 2: 64개의 필터를 사용한 Conv2D 레이어와 MaxPooling2D 레이어
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    # 첫 번째 Conv2D 레이어: 64개의 필터, 3x3 크기의 커널, ReLU 활성화 함수, 'same' 패딩
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    # 두 번째 Conv2D 레이어: 64개의 필터, 3x3 크기의 커널, ReLU 활성화 함수, 'same' 패딩
    MaxPooling2D(pool_size=(2, 2)),   # MaxPooling2D: 2x2 풀링, 출력 크기 16×16 → 8×8

    # 완전 연결층 (Fully Connected Layer): 이미지를 1차원 벡터로 평탄화하고, Dense 레이어 추가
    Flatten(),  # 2D 이미지를 1D 벡터로 변환
    Dense(512, activation='relu'),  # 512개의 유닛을 가진 Dense 레이어, ReLU 활성화 함수
    Dense(10, activation='softmax'),  # 출력 레이어, 10개의 유닛 (CIFAR-10의 클래스 수), Softmax 함수
])

# 4. 모델 컴파일: 모델 학습을 위한 최적화, 손실 함수, 평가 지표 설정
model.compile(
    optimizer='adam',  # Adam 옵티마이저 사용 (효율적인 학습)
    loss='sparse_categorical_crossentropy',  # 다중 클래스 분류 문제에 적합한 손실 함수
    metrics=['accuracy']  # 모델 성능을 평가하기 위한 지표, 정확도 사용
)

# 5. 모델 훈련: 10 에폭 동안 학습
history = model.fit(
    x_train, y_train,  # 훈련 데이터와 레이블을 사용
    epochs=10,  # 훈련을 10번 반복 (에폭 수)
    batch_size=64,  # 한 번에 학습할 샘플 수 (배치 크기)
    validation_split=0.1,  # 10%의 데이터를 검증용 데이터로 사용
    verbose=2  # 훈련 과정을 자세히 출력 (진행 상태 표시)
)

# 6. 모델 평가: 테스트 데이터셋을 사용하여 모델 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# 7. 테스트 결과 출력: 모델이 테스트 데이터셋에서의 성능을 출력
print("=" * 45)
print("테스트 결과")
print(f"  테스트 손실(Loss): {test_loss:.4f}")  # 테스트 데이터셋에서의 손실 값 출력
print(f"  테스트 정확도:     {test_acc * 100:.2f}%")  # 테스트 데이터셋에서의 정확도 출력
print("=" * 45)

# 8. dog.jpg 예측: 주어진 이미지를 예측하여 어떤 클래스인지 출력
IMG_PATH = 'image/dog.jpg'  # 예측할 이미지 경로

# 예측할 이미지 파일 경로가 존재하는지 확인
if os.path.exists(IMG_PATH):
    from keras.preprocessing import image
    # 이미지 로드 및 전처리: CIFAR-10에 맞게 크기 조정 후 정규화
    img = image.load_img(IMG_PATH, target_size=(32, 32))  # CIFAR-10 이미지 크기로 리사이즈
    img_arr = image.img_to_array(img) / 255.0             # 이미지 배열로 변환 후 정규화
    img_input = np.expand_dims(img_arr, axis=0)           # 모델에 맞게 차원 추가 (1, 32, 32, 3)

    # 예측: 모델을 사용하여 이미지에 대한 분류 예측 수행
    pred = model.predict(img_input, verbose=0)
    pred_class = np.argmax(pred)  # 예측된 클래스의 인덱스 추출
    confidence = pred[0][pred_class] * 100  # 해당 클래스에 대한 신뢰도 계산

    # 결과 출력 (예측된 클래스와 신뢰도를 출력)
    print(f"\n dog.jpg 예측 결과: {CLASS_NAMES[pred_class]}")  # 예측된 클래스 이름 출력
    print(f"신뢰도: {confidence:.1f}%")  # 예측의 신뢰도 출력
else:
    # 이미지 파일이 없을 경우 에러 메시지 출력
    print(f"\n '{IMG_PATH}' 파일을 찾을 수 없습니다.")
    print("   dog.jpg를 코드와 같은 폴더에 넣어주세요.")
