import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
# - cv.imread(): 이미지를 BGR 형식으로 읽어옴
img = cv.imread('image\\coffee cup.jpg')

# - OpenCV는 BGR, matplotlib은 RGB를 사용하므로 색상 변환 필요
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2. 마스크 초기화
# - GrabCut 알고리즘에서 사용할 마스크 배열 생성
# - 이미지와 동일한 크기의 2차원 배열 (높이 x 너비)
# - 초기값 0: 모든 픽셀을 '확실한 배경(GC_BGD)'으로 설정
mask = np.zeros(img.shape[:2], np.uint8)

# 3. GrabCut 모델 초기화
# - 내부적으로 GMM(Gaussian Mixture Model)에 사용되는 배열
# - 사용자가 직접 의미를 해석할 필요는 없으며, 알고리즘 내부에서 자동 갱신됨
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델

# 4. 초기 사각형(rect) 설정
# - 형식: (x, y, width, height)
# - 해당 영역 내부를 전경(객체) 후보로 간주
# - 객체(컵)가 충분히 포함되도록 설정해야 정확한 분할 가능
h, w = img.shape[:2]
rect = (50, 50, w-100, h-100)

# 5. GrabCut 알고리즘 실행
# - 주어진 rect를 기반으로 전경/배경을 분리
# - mask는 이 과정에서 자동으로 업데이트됨
# - 반복 횟수(10)를 통해 결과를 점진적으로 개선
# - cv.GC_INIT_WITH_RECT: 사각형 기반 초기화 방식
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

# 6. 마스크 이진화
# GrabCut 마스크 값 의미:
# 0: 확실한 배경 (GC_BGD)
# 1: 확실한 전경 (GC_FGD)
# 2: 배경일 가능성 (GC_PR_BGD)
# 3: 전경일 가능성 (GC_PR_FGD)
#
# - 배경(0, 2)은 0으로 변환
# - 전경(1, 3)은 1로 변환
# → 최종적으로 0과 1로 이루어진 이진 마스크 생성
mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,
    1
).astype('uint8')

# 7. 노이즈 제거 및 마스크 보정
# - morphology 연산을 이용하여 마스크 품질 개선
# - 커널: 5x5 크기의 구조 요소
kernel = np.ones((5, 5), np.uint8)

# - MORPH_CLOSE: 작은 구멍을 메우고 내부를 채움
mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)

# - MORPH_OPEN: 작은 잡음(노이즈) 제거
mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)

# 8. 배경 제거
# - mask2를 (H, W, 1) 형태로 확장하여 RGB 이미지에 적용
# - 전경(1)은 유지되고, 배경(0)은 제거됨
result = img_rgb * mask2[:, :, np.newaxis]

# 9. 결과 시각화
# - 원본 이미지, 마스크 이미지, 배경 제거 결과를 나란히 출력
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
