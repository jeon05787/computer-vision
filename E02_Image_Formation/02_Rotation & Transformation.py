import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread("image/rose.png")

if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# 이미지 크기 절반으로 줄이기
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# 이미지 크기
h, w = img.shape[:2]

# 이미지 중심
center = (w // 2, h // 2)

# 회전 + 스케일 행렬 생성
# +30도 회전, scale=0.8
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 평행이동 추가
# x축 +80, y축 -40
M[0, 2] += 80
M[1, 2] += -40

# affine transformation 적용
transformed = cv2.warpAffine(img, M, (w, h))

# 결과 출력
cv2.imshow("Original", img)
cv2.imshow("Rotated + Scaled + Translated", transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()
