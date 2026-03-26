import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv.imread('image\mot_color70.jpg')

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점 및 디스크립터 검출
keypoints, descriptors = sift.detectAndCompute(img, None)

# 특징점을 원본 이미지에 시각화
img_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 원본 이미지와 특징점이 표시된 이미지 나란히 출력
plt.figure(figsize=(12, 6))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 특징점이 시각화된 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))
plt.title("KeyPoints Detected")
plt.axis('off')

# 출력
plt.show()
