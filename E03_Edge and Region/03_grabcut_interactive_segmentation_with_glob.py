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
