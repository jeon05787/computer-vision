import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성 (결과 이미지 저장을 위한 폴더 생성)
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)  # 만약 폴더가 없으면 새로 생성

# 좌/우 이미지 불러오기
left_color = cv2.imread("image/left.png")  # 왼쪽 이미지 읽기
right_color = cv2.imread("image/right.png")  # 오른쪽 이미지 읽기

# 이미지가 제대로 불러와졌는지 확인
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 이미지 파일이 없으면 에러 발생

# 카메라 파라미터 설정
f = 700.0  # 초점 거리(focal length)
B = 0.12  # 두 카메라 간 거리 (baseline)

# ROI (Region of Interest) 설정: 특정 영역을 지정하여 그 지역만 처리
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환 (두 이미지를 그레이스케일로 변환하여 처리)
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)  # 왼쪽 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 이미지를 그레이스케일로 변환

# -----------------------------
# 1. Disparity 계산 (시차 계산)
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # StereoBM 객체 생성 (시차 계산을 위한 객체)

# StereoBM 결과는 16배 스케일된 정수형으로 반환되므로 16으로 나누어야 한다.
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # 시차를 계산하고 부동소수점으로 변환

# -----------------------------
# 2. Depth 계산 (깊이 계산)
# Z = fB / d (깊이 = 초점거리 * 두 카메라 간 거리 / 시차)
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)  # 깊이 맵 초기화 (시차 맵과 같은 크기)

# 시차가 0 이하일 경우, 깊이 계산 불가하므로 이를 제외하고 계산
valid_mask = disparity > 0
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효한 시차 값을 이용해 깊이 계산

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

# 지정한 ROI 영역에 대해 평균 시차와 평균 깊이를 계산
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]  # ROI 내 시차 값 추출
    roi_depth = depth_map[y:y+h, x:x+w]  # ROI 내 깊이 값 추출

    roi_valid_disp = roi_disp[roi_disp > 0]  # 유효한 시차 값만 필터링
    roi_valid_depth = roi_depth[roi_depth > 0]  # 유효한 깊이 값만 필터링

    # 유효한 값이 있으면 평균을 계산하고, 없으면 NaN 처리
    mean_disp = float(np.mean(roi_valid_disp)) if roi_valid_disp.size > 0 else np.nan
    mean_depth = float(np.mean(roi_valid_depth)) if roi_valid_depth.size > 0 else np.nan

    # 계산된 결과를 딕셔너리에 저장
    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI별 평균 Disparity / Depth ===")
for name, values in results.items():
    print(f"{name}")
    print(f"  Mean disparity: {values['mean_disparity']:.2f}")  # 평균 시차 출력
    print(f"  Mean depth    : {values['mean_depth']:.2f}")  # 평균 깊이 출력

# 가장 가까운 / 가장 먼 ROI 찾기
valid_results = {k: v for k, v in results.items() if not np.isnan(v["mean_depth"])}  # NaN이 아닌 값만 필터링

if len(valid_results) > 0:
    # 가장 가까운 ROI (가장 작은 깊이)
    nearest_roi = min(valid_results.items(), key=lambda x: x[1]["mean_depth"])
    # 가장 먼 ROI (가장 큰 깊이)
    farthest_roi = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])

    print("\n=== 거리 비교 ===")
    print(f"가장 가까운 ROI: {nearest_roi[0]} ({nearest_roi[1]['mean_depth']:.2f})")
    print(f"가장 먼 ROI    : {farthest_roi[0]} ({farthest_roi[1]['mean_depth']:.2f})")

# -----------------------------
# 5. disparity 시각화 (시차 맵을 컬러로 표시)
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()  # 시차 맵 복사
disp_tmp[disp_tmp <= 0] = np.nan  # 0 이하 값은 NaN으로 설정

if np.all(np.isnan(disp_tmp)):  # 시차 값이 모두 NaN이면 예외 처리
    raise ValueError("유효한 disparity 값이 없습니다.")

# 시차 값의 최소값과 최대값을 기준으로 정규화
d_min = np.nanpercentile(disp_tmp, 5)  # 5% percentile
d_max = np.nanpercentile(disp_tmp, 95)  # 95% percentile

if d_max <= d_min:  # 최소값과 최대값이 같으면 오류 방지
    d_max = d_min + 1e-6

# 시차 값을 0에서 1 사이로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)  # 0과 1 사이로 클리핑

# 시차 값에 대해 컬러 맵을 적용
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # JET 컬러맵 적용

# -----------------------------
# 6. depth 시각화 (깊이 맵을 컬러로 표시)
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):  # 유효한 깊이 값이 있을 경우
    depth_valid = depth_map[valid_mask]  # 유효한 깊이 값 추출

    # 깊이 값의 최소값과 최대값을 기준으로 정규화
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:  # 최소값과 최대값이 같으면 오류 방지
        z_max = z_min + 1e-6

    # 깊이 값을 0에서 1 사이로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)  # 0과 1 사이로 클리핑

    # 깊이는 거리가 멀수록 값이 크므로 반전시켜야 한다
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # JET 컬러맵 적용

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

# ROI 영역을 이미지에 표시
for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ROI 영역 사각형 그리기
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 이름 텍스트 추가

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 오른쪽 이미지에도 동일하게 표시
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장 (결과 이미지 저장)
# -----------------------------
cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)

# -----------------------------
# 9. 출력 (이미지 화면에 표시)
# -----------------------------
cv2.imshow("Left with ROI", left_vis)
cv2.imshow("Right with ROI", right_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기
