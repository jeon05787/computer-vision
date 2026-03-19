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
