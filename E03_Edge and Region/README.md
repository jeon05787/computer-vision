## 01_소벨 에지 검출 및 결과 시각화

### 설명

* edgeDetectionImage 이미지를 그레이스케일로 변환
* Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출
* 검출된 에지 강도 이미지를 시각화

### 요구사항

* cv.imread()를 사용하여 이미지를 불러옴
* cv.cvtColor()를 사용하여 그레이스케일로 변환
* cvSobel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출
* cv.magnitude()를 사용하여 에지 강도 계산
* Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
* cv.Sobel()의 ksize는 3 또는 5로 설정
* cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환
* plt.imshow()에서 cmap=‘gray’를 사용하여 흑백으로 시각화


### 전체코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지를 불러오기
image_path = 'image\edgeDetectionImage.jpg'  # 이미지 경로 설정
image = cv.imread(image_path)

# 2. 이미지를 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 3. Sobel 필터를 사용하여 x축과 y축 방향 에지 검출
sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)  # x축 방향
sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)  # y축 방향

# 4. 에지 강도 계산
edge_magnitude = cv.magnitude(sobel_x, sobel_y)

# 5. 에지 강도 이미지를 uint8로 변환
edge_magnitude = cv.convertScaleAbs(edge_magnitude)

# 6. 원본 이미지와 에지 강도 이미지를 나란히 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 에지 강도 이미지
plt.subplot(1, 2, 2)
plt.imshow(edge_magnitude, cmap='gray')
plt.title("Edge Magnitude (Sobel)")
plt.axis('off')

plt.tight_layout()
plt.show()


```

### 주요코드
- **`image = cv.imread(image_path)`**: 이미지 불러오기
- **`gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  `**: 그레이스케일로 변환
- Sobel 필터를 사용한 에지 검출(x축, y축) <br>
  **`sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3) `**:  x축 방향 에지 검출 <br>
  **`sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3) `**:  y축 방향 에지 검출
- **`edge_magnitude = cv.magnitude(sobel_x, sobel_y)`**:  x, y 방향의 에지를 합성하여 강도 계산
- **`edge_magnitude = cv.convertScaleAbs(edge_magnitude)`**:   에지 강도를 uint8로 변환
- **`plt.subplot(1, 2, 1) `**: 원본 이미지 표시
- **`plt.subplot(1, 2, 2) `**: 에지 강도 이미지 표시
- **`plt.imshow(edge_magnitude, cmap='gray')` `** 를 통한 이미지 시각화

### 실행결과
<img width="1498" height="832" alt="image" src="https://github.com/user-attachments/assets/7749066c-a0b6-4155-a42a-ffa737b6206f" />

## 02_캐니 에지 및 허프 변환을 이용한 직선 검출

### 설명
* dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성
* 허프 변환을 사용하여 이미지에서 직선 검출
* 검출된 직선을 원본 이미지에서 빨간색으로 표시

### 요구사항
* cv.Canny()를 사용하여 에지 맵 생성
* cv.HoughtLinesP()를 사용하여 직선 검출
* cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림
* Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
* cv.Canny()에서 threshold1과 threshold2는 100과 200으로 설정
* cv.HoughLinesP()에서 rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선
* cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정


### 전체코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image_path = 'image\dabo.jpg'  # 이미지 경로 설정
image = cv.imread(image_path)

# 2. 이미지를 그레이스케일로 변환
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 3. 캐니 에지 검출
edges = cv.Canny(gray_image, 100, 200)

# 4. 허프 변환을 사용하여 직선 검출
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# 5. 원본 이미지에 직선 그리기
image_with_lines = image.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색, 두께 2로 직선 그리기

# 6. 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 직선이 그려진 이미지
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(image_with_lines, cv.COLOR_BGR2RGB))
plt.title("Image with Detected Lines")
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 주요코드
- **`image = cv.imread(image_path)`**: 이미지 불러오기
-  **`gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY`**: 그레이스케일로 변환
- Canny Edge Detection <br>
   **`edges = cv.Canny(gray_image, 100, 200)`**: Canny 알고리즘을 사용하여 에지 검출
- Hough 변환을 사용한 직선 검출 <br>
  **`lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)`**
- 원본 이미지에 검출된 직선 그리기
  ```python
   image_with_lines = image.copy()  # 이미지 복사
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)`**:  빨간색 직선 그리기
   ```
### 실행결과
<img width="1497" height="830" alt="image" src="https://github.com/user-attachments/assets/b117391b-8063-4ed0-b57d-b4b1628ccf8c" />



## 03_GrabCut을 이용한 대화식 영역 분할 및 객체 추출

### 설명

* coffee cup 이미지로 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
* 객체 추출 결과를 마스크 형태로 시각화
* 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력

### 요구사항

* cv.grabCut()를 사용하여 대화식 분할을 수행
* 초기 사각형 영역은 (x, y, width, height) 형식으로 설
* 마스크를 사용하여 원본 이미지에서 배경을 제거
* matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화
* 마스크 값은 cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD를 사용
* np.where()를 사용하여 마스크 값을 0 또는 1로 변경한 후 원본 이미지에 곱하여 배경을 제거
  
### 전체코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
# - cv.imread(): 이미지를 BGR 형식으로 읽어옴
img = cv.imread('image\\coffee cup.jpg')

# - OpenCV는 BGR, matplotlib은 RGB를 사용하므로 변환 필요
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2. 마스크 초기화
# - GrabCut에서 사용할 마스크 배열 생성
# - 이미지와 동일한 크기의 2D 배열 (초기값: 0 → 확실한 배경)
mask = np.zeros(img.shape[:2], np.uint8)

# 3. GrabCut 모델 초기화
# - 내부적으로 GMM(Gaussian Mixture Model)에 사용됨
# - 사용자가 직접 수정하지 않는 내부 버퍼 역할
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델

# 4. 초기 사각형(rect) 설정
# - (x, y, width, height)
# - 이 영역 내부를 "전경 후보"로 간주
# - 너무 작으면 객체가 잘림 → 크게 잡는 것이 중요
h, w = img.shape[:2]
rect = (10, 10, w-20, h-20)  # 거의 전체 이미지 포함

# 5. GrabCut 알고리즘 실행
# - mask를 업데이트하면서 foreground/background를 분리
# - 10번 반복 수행하여 더 정교하게 분할
# - cv.GC_INIT_WITH_RECT: rect 기반 초기화
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

# 6. 마스크 값 변환
# GrabCut 결과 값 의미:
# 0: 확실한 배경 (GC_BGD)
# 1: 확실한 전경 (GC_FGD)
# 2: 배경일 가능성 (GC_PR_BGD)
# 3: 전경일 가능성 (GC_PR_FGD)

# - 배경(0,2) → 0
# - 전경(1,3) → 1
# → 이진 마스크 생성
mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,  # 배경
    1   # 전경
).astype('uint8')

# 7. 배경 제거
# - mask2를 3채널로 확장하여 RGB 이미지에 적용
# - 전경(1)은 유지, 배경(0)은 제거됨
result = img_rgb * mask2[:, :, np.newaxis]

# 8. 결과 시각화 (matplotlib)
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# 마스크 이미지 (흑백)
plt.subplot(1, 3, 2)
plt.title('Mask Image')
plt.imshow(mask2, cmap='gray')
plt.axis('off')

# 배경 제거 결과
plt.subplot(1, 3, 3)
plt.title('Background Removed')
plt.imshow(result)
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 주요코드
- **`rect = (10, 10, w-20, h-20)`**:  GrabCut에서 사용할 초기 사각형 영역을 설정한다. 이 영역 내부는 전경(객체) 후보로 간주되며, 객체를 충분히 포함하도록 크게 설정하는 것이 중요하다.
-  **`cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)`**:GrabCut 알고리즘을 실행하여 이미지에서 전경과 배경을 분리한다.
rect를 기반으로 초기 분할을 수행하며, 반복 횟수(10)를 통해 분할 결과를 점진적으로 개선한다.    
-  **`mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')`**:  GrabCut 결과 마스크를 이진 마스크(0 또는 1)로 변환한다.
배경(확실한 배경 + 배경일 가능성)은 0, 전경(확실한 전경 + 전경일 가능성)은 1로 변환한다.
- **`result = img_rgb * mask2[:, :, np.newaxis]`**:  이진 마스크를 원본 이미지에 적용하여 배경을 제거한다.
마스크를 3채널로 확장한 뒤 곱셈을 수행하여 전경만 남기고 배경은 제거한다.
-  **`mask = np.zeros(img.shape[:2], np.uint8)`**:  GrabCut에서 사용할 초기 마스크를 생성한다.
모든 픽셀을 배경으로 초기화한 뒤 알고리즘이 이를 업데이트한다.
- GrabCut → 마스크 생성 → 이진화 → 이미지에 적용 → 배경 제거

### 실행결과
<img width="1877" height="705" alt="image" src="https://github.com/user-attachments/assets/85ee5a22-9357-410f-bd8f-a79bb29a090f" />



