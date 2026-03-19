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
