## 01_SIFT를 이용한특징점검출및시각화

### 설명

*  주어진이미지(mot_color70.jpg)를이용하여SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을 검출하고이를시각화

### 요구사항
* cv.SIFT_create()를 사용하여 SIFT 객체를 생성
* detectAndCompute()를 사용하여 특징점을 검출
* cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
* matplotlib을 이용하여 원본 이미지와 특징점이 시각화 된 이미지를 나란히 출력
* SIFT_create()의 매개변수를 변경하며 특징점 검출 결과를 비교
* 특징점이 너무 많다면 nfeatures 값을 조정하여 제한
* cv.drawKeypoints()의 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 특징점의
방향과크기도표시

### 전체코드
```python
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
```

### 주요코드
- **`cv.imread('mot_color70.jpg')`**:  주어진 이미지를 불러옵니다
-  **`sift = cv.SIFT_create()`**:  SIFT 객체 생성
-  **`keypoints, descriptors = sift.detectAndCompute(img, None)`**: 이미지에서 특징점을 추출하고 그에대한 디스크립터를 계산합니다. keypoints는 특징점의 위치를 , descriptors 은 각 특징점의 설명자를 나타냅니다.  
- **`cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)`**:검출된 특징점들을 이미지 위에 그립니다.  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 사용하면 특징점의 방향과 크기까지 표시됩니다.
-  **`plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))`**:  Matplotlib을 사용하여 이미지를 시각화합니다.

### 실행결과
<img width="1495" height="486" alt="image" src="https://github.com/user-attachments/assets/d16bcf99-4451-4f89-bace-7a2be3317ab8" />

## 02_SIFT를 이용한 두 영상 간 특징점 매칭

### 설명

* 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화

### 요구사항

* cv.imread()를 사용하여 두 개의 이미지를 불러옴
* cv.SIFT_create()를 사용하여 특징점을 추출
* cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭
* cv.drawMatches()를 사용하여 매칭 결과를 시각화
* matplotlib을 이용하여 매칭 결과를 출력
* BFMatcher(cv.NORM_L2, crossCheck=True)를 사용하면 간단한 매칭이 가능
* FLANN 기반 매칭을 원하면 cv.FlannBasedMatcher()를 사용
* knnMatch()와 DMatch 객체를 활용하여 최근접 이웃 거리 비율을 적용하면 매칭 정확도를 높일 수 있음


### 전체코드
```python
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
```

### 주요코드
- **`img1 = cv.imread('mot_color70.jpg', cv.IMREAD_COLOR), img2 = cv.imread('mot_color80.jpg', cv.IMREAD_COLOR)`**:   두 이미지를 불러옵니다.
-  **`sift = cv.SIFT_create()`**: SIFT 객체를 생성하여 특징점을 검출합니다.
-  **`특징점 및 디스크립터 추출`**:  keypoints1, descriptors1와 keypoints2, descriptors2를 통해 각각 두 이미지에서 특징점과 디스크립터를 추출합니다.
- **`cv.BFMatcher(),knnMatch()`**: 두 이미지 간의 특징점을 매칭합니다.
-  **`cv.findHomography()`**:  두 이미지 간의 호모그래피 행렬을 계산하고, 이 정보를 사용해 이미지를 정합합니다
- **`cv.warpPerspective()`**: 첫 번째 이미지를 두 번째 이미지와 맞춰 변환합니다.


### 실행결과
<img width="1500" height="495" alt="image" src="https://github.com/user-attachments/assets/328f642c-ce85-4604-b375-058423a8851b" />

## 03_호모그래피를 이용한 이미지 정합 (Image Alignment)

### 설명

* SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
* 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택

### 요구사항

* cv.imread()를 사용하여 두 개의 이미지를 불러옴
* cv.SIFT_create()를 사용하여 특징점을 검출
* cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
* cv.findHomography()를 사용하여 호모그래피 행렬을 계산
* cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
* 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
* cv.findHomography()에서 cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음
* cv.warpPerspective()를 사용할 때 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
* knnMatch()로 두 개의 최근접 이웃을 구한 뒤, 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별


### 전체코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# ──────────────────────────────────────────────
# 0. 이미지 경로 설정 
# ──────────────────────────────────────────────
IMG1_PATH ='image\img1.jpg' 
IMG2_PATH ='image\img2.jpg' 

# ──────────────────────────────────────────────
# 1. 이미지 로드
# ──────────────────────────────────────────────
img1 = cv.imread(IMG1_PATH)
img2 = cv.imread(IMG2_PATH)

if img1 is None or img2 is None:
    print(f"[ERROR] 이미지를 불러올 수 없습니다.\n"
          f"  img1: {IMG1_PATH}\n  img2: {IMG2_PATH}\n"
          "경로를 확인하거나 IMG1_PATH / IMG2_PATH 변수를 수정하세요.")
    sys.exit(1)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# ──────────────────────────────────────────────
# 2. SIFT 특징점 검출 및 기술자 추출
# ──────────────────────────────────────────────
sift = cv.SIFT_create()  # SIFT 객체 생성

kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지
kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지

print(f"[INFO] img1 특징점 수: {len(kp1)}")
print(f"[INFO] img2 특징점 수: {len(kp2)}")

# ──────────────────────────────────────────────
# 3. BFMatcher + knnMatch + Lowe's ratio test
# ──────────────────────────────────────────────
bf = cv.BFMatcher(cv.NORM_L2)  # Brute-Force 매칭 객체 생성

raw_matches = bf.knnMatch(des2, des1, k=2)   # img2 → img1 방향으로 매칭

RATIO_THRESH = 0.7  # Lowe's 비율 테스트를 위한 임계값 설정
good_matches = []   # 좋은 매칭을 담을 리스트

# Lowe's 비율 테스트 적용 (두 번째 최근접 이웃의 거리가 첫 번째보다 0.7배 이내일 때만 좋은 매칭으로 판단)
for m, n in raw_matches:
    if m.distance < RATIO_THRESH * n.distance:
        good_matches.append(m)

print(f"[INFO] 전체 매칭 수: {len(raw_matches)}  →  좋은 매칭 수: {len(good_matches)}")

if len(good_matches) < 4:
    print("[ERROR] 호모그래피 계산에 필요한 최소 매칭 수(4)가 부족합니다.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. 호모그래피 계산 (RANSAC)
# ──────────────────────────────────────────────
src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # img2의 특징점
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # img1의 특징점

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=5.0)

inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
print(f"[INFO] RANSAC 인라이어 수: {len(inlier_matches)}")

# ──────────────────────────────────────────────
# 5. Perspective Warp → 파노라마 크기 출력
# ──────────────────────────────────────────────
panorama_w = w1 + w2  # 두 이미지의 가로 크기 합
panorama_h = max(h1, h2)  # 두 이미지의 세로 크기 중 더 큰 값

warped = cv.warpPerspective(img2, H, (panorama_w, panorama_h))

result = warped.copy()  # warped 이미지를 result에 복사
result[0:h1, 0:w1] = img1  # 첫 번째 이미지를 합성

# ── 검은 영역 자동 크롭 ──────────────────────
gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray_result, 1, 255, cv.THRESH_BINARY)  # 내용이 있는 부분을 추출
coords = cv.findNonZero(thresh)  # 내용이 있는 픽셀 좌표 찾기
x, y, w_crop, h_crop = cv.boundingRect(coords)  # 내용이 있는 부분의 바운딩 박스 계산

PAD = 5
x = max(0, x - PAD)
y = max(0, y - PAD)
w_crop = min(panorama_w - x, w_crop + 2 * PAD)
h_crop = min(panorama_h - y, h_crop + 2 * PAD)

result = result[y:y+h_crop, x:x+w_crop]

# ──────────────────────────────────────────────
# 6. 매칭 결과 시각화
# ──────────────────────────────────────────────
match_vis = cv.drawMatches(
    img2, kp2, img1, kp1,
    inlier_matches[:50],          # 최대 50개만 표시
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ──────────────────────────────────────────────
# 7. 결과 출력 (나란히)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("SIFT + Homography Image Alignment", fontsize=15, fontweight="bold")

axes[0].imshow(cv.cvtColor(match_vis, cv.COLOR_BGR2RGB))
axes[0].set_title(f"Matching Result  (inliers: {len(inlier_matches)})", fontsize=12)
axes[0].axis("off")

axes[1].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
axes[1].set_title("Warped & Aligned Image (Panorama)", fontsize=12)
axes[1].axis("off")

plt.tight_layout()
plt.show()


```

### 주요코드
- **`sift = cv.SIFT_create()`**: SIFT 알고리즘 객체를 생성합니다. 이 객체를 사용하여 이미지에서 특징점과 디스크립터를 추출합니다.
-  **`sift.detectAndCompute()`**: 각 이미지에서 특징점(keypoints)과 그에 대한 디스크립터(descriptors)를 추출합니다.
-  **`cv.BFMatcher()와 knnMatch()`**: 두 이미지 간의 특징점들을 매칭합니다. Lowe's 비율 테스트를 통해 좋은 매칭만 필터링하여 선택합니다.
- **`cv.findHomography()`**:  두 이미지 간의 호모그래피 행렬을 계산합니다. RANSAC을 사용하여 이상치를 제거하고, 유효한 매칭만 사용하여 호모그래피를 계산합니다.
-  **`cv.warpPerspective()`**:  두 번째 이미지를 첫 번째 이미지와 맞게 변환합니다. 변환된 이미지를 왼쪽에 합성하여 파노라마 이미지를 만듭니다. 
-  **`cv.drawMatches()`**:  두 이미지 간의 매칭된 특징점들을 시각화합니다.


### 실행결과


