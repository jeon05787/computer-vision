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
