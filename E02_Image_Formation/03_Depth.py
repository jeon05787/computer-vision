import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("image/left.png")
right_color = cv2.imread("image/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
f = 700.0  # 초점거리
B = 0.12   # 카메라 간 거리

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# StereoBM 결과는 16배 스케일된 정수형이므로 16으로 나눠야 함
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)

# disparity가 0 이하이면 depth 계산 불가
valid_mask = disparity > 0
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]

    # disparity > 0인 유효한 값만
    roi_valid_disp = roi_disp[roi_disp > 0]
    roi_valid_depth = roi_depth[roi_depth > 0]

    mean_disp = float(np.mean(roi_valid_disp)) if roi_valid_disp.size > 0 else np.nan
    mean_depth = float(np.mean(roi_valid_depth)) if roi_valid_depth.size > 0 else np.nan

    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력 - 가장 가까운 / 가장 먼 ROI만 출력
# -----------------------------
valid_results = {k: v for k, v in results.items() if not np.isnan(v["mean_depth"])}

if len(valid_results) > 0:
    # 가장 가까운 ROI는 depth가 작은 것
    nearest_roi = min(valid_results.items(), key=lambda x: x[1]["mean_depth"])

    # 가장 먼 ROI는 depth가 큰 것
    farthest_roi = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])

    print("\n=== 가장 가까운 ROI, 가장 먼 ROI ===")
    print(f"가장 가까운 ROI: {nearest_roi[0]} ({nearest_roi[1]['mean_depth']:.2f})")
    print(f"가장 먼 ROI    : {farthest_roi[0]} ({farthest_roi[1]['mean_depth']:.2f})")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 7. Left / Right 이미지를 나란히 출력
# -----------------------------
combined = np.hstack((left_vis, disparity_color))

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "combined.png"), combined)

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Original and Disparity Map", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
