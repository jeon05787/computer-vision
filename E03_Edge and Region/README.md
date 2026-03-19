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
- Sobel 필터를 사용한 에지 검출(x축, y축)
- **`sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3) `**:  x축 방향 에지 검출
- **`sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3) `**:  y축 방향 에지 검출
- **`edge_magnitude = cv.magnitude(sobel_x, sobel_y)`**:  x, y 방향의 에지를 합성하여 강도 계산
- **`edge_magnitude = cv.convertScaleAbs(edge_magnitude)`**:   에지 강도를 uint8로 변환
- **`plt.subplot(1, 2, 1) `**: 원본 이미지 표시
- **`plt.subplot(1, 2, 2) `**: 에지 강도 이미지 표시
- **`plt.imshow(edge_magnitude, cmap='gray')` `** 를 통한 이미지 시각화

### 실행결과

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
- Canny Edge Detection
-  **`edges = cv.Canny(gray_image, 100, 200)`**: Canny 알고리즘을 사용하여 에지 검출
- Hough 변환을 사용한 직선 검출
- **`lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)`**
- 원본 이미지에 검출된 직선 그리기
-  **`image_with_lines = image.copy()  # 이미지 복사
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)`**:  빨간색 직선 그리기
   
### 실행결과



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
image_path = 'image\coffee cup.JPG'  # coffee cup 이미지 경로 설정
image = cv.imread(image_path)

# 2. 초기 사각형 영역 설정 (x, y, width, height)
rect = (50, 50, 450, 290)  # 예시로 설정된 영역, 사용자에 맞게 변경 가능

# 3. bgdModel과 fgdModel 초기화
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 4. GrabCut 알고리즘을 사용하여 대화식 분할 수행
mask = np.zeros(image.shape[:2], np.uint8)  # 마스크 초기화
cv.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

# 5. 마스크 값 변경: (배경은 0, 객체는 1로 설정)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 6. 원본 이미지에서 배경을 제거하고 객체만 남기기
result = image * mask2[:, :, np.newaxis]

# 7. 원본 이미지, 마스크 이미지, 배경 제거 이미지를 나란히 시각화
plt.figure(figsize=(18, 6))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 마스크 이미지
plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title("Mask Image")
plt.axis('off')

# 배경 제거 이미지
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Foreground Image")
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 주요코드
- **`코드`**:  
  
-  **`코드`**:  
  
-  **`코드`**:  

- **`코드`**:  

-  **`코드`**:  

### 실행결과



