import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 읽기
img1 = cv.imread('image\mot_color70.jpg')
img2 = cv.imread('image\mot_color83.jpg', cv.IMREAD_COLOR)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점 및 디스크립터 검출
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 매칭 객체 생성 (Brute-Force Matcher)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# 특징점 매칭
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과를 거리 기준으로 정렬
matches = sorted(matches, key = lambda x:x.distance)

# 매칭 결과 시각화
img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
plt.title("SIFT Feature Matching")
plt.axis('off')
plt.show()
