## 01_체크보드 기반 카메라 캘리브레이션

### 설명

* 이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터 추정
* 체크보드 패턴이 촬영된 여러 장의 이미지를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하여 왜곡 보정

### 요구사항

* 모든 이미지에서 체크보드 코너를 검출
* 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
* cv2.calibrateCamera()를 사용하여 카메라 내부 행렬 k와 왜곡 계수를 구함
* cv2.undistort()를 사용하여 왜곡 보정한 결과를 시각화

### 체크보드란?

* 카메라 캘리브레이션을 위해 사용하는 흑백 격자 패턴
* Corner이 규칙적인 격자 구조
* 실제 좌표를 정확히 알고 있는 패턴이기 때문에 이미지에서 검출된 코너와 실제 좌표의 대응 관계를 이용해 카메라의
내부 파라미터와 렌즈 왜곡을 계산할 수 있음


### 전체코드
```python
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (실제 좌표와 이미지 좌표 매칭을 위한 기준)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건 (정확도를 높이기 위한 최대 반복 횟수 및 정확도 기준 설정)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성 (체크보드의 각 코너의 실제 좌표 설정, 3D 좌표에서 z는 0으로 고정)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 좌표를 저장할 리스트
objpoints = []  # 3D 좌표
imgpoints = []  # 2D 좌표

# 이미지 경로 불러오기
images = glob.glob("image/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"이미지를 읽을 수 없음: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
    img_size = gray.shape[::-1]  # 이미지 크기

    # 체크보드의 코너를 찾는다
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:  # 코너가 검출되면
        # 코너 정밀화 (정확도를 높이기 위한 서브픽셀 정밀화)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 실제 3D 좌표와 이미지 좌표를 매칭
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 검출된 코너를 이미지에 표시
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Corners", img)
        cv2.waitKey(300)
    else:
        print(f"코너 검출 실패: {fname}")

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

print("Camera Matrix K:")
print(K)  # 카메라 행렬

print("\nDistortion Coefficients:")
print(dist)  # 왜곡 계수

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        continue

    undistorted = cv2.undistort(img, K, dist)  # 왜곡 보정

    # 원본 이미지와 보정된 이미지를 나란히 표시
    combined = np.hstack((img, undistorted))

    cv2.imshow("Original (Left) vs Undistorted (Right)", combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()
```

### 주요코드
- **`cv2.findChessboardCorners()`**:  
  이미지에서 체크보드 코너를 찾고, 성공하면 `ret=True`와 검출된 코너 좌표를 반환합니다.

- **`cv2.cornerSubPix()`**:  
  `cv2.findChessboardCorners()`로 찾은 코너의 정확도를 서브픽셀 수준으로 개선합니다.

- **`cv2.calibrateCamera()`**:  
  여러 이미지에서 3D-2D 좌표를 사용하여 카메라 내부 행렬과 왜곡 계수를 계산합니다.

- **`cv2.undistort()`**:  
  계산된 카메라 파라미터를 사용하여 왜곡된 이미지를 보정합니다.

- **이미지 표시 및 비교**:  
  `np.hstack()`을 사용하여 원본 이미지와 보정된 이미지를 가로로 이어 붙여 시각화합니다.

### 실행결과
<img width="1599" height="638" alt="스크린샷 2026-03-12 153519" src="https://github.com/user-attachments/assets/afbeac0b-a9c0-45c0-bff0-9f78ead3c5bf" />

<img width="633" height="177" alt="image" src="https://github.com/user-attachments/assets/79247f06-7a00-4a17-8e35-caf25178df5d" />

## 02_이미지 Rotation & Transformation

### 설명
* 한 장의 이미지에 회전, 크기 조절, 평행이동을 적용

### 요구사항
* 이미지의 중심 기준으로 +30도 회전
* 회전과 동시에 크기를 0.8로 조절
* 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동

### 전체코드
```python
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
```

### 주요코드
# 주요 코드 설명

- **`cv2.imread()`**:  
  이미지를 파일에서 읽어옵니다. 이미지 경로가 잘못되면 에러 메시지를 출력하고 종료합니다.

- **`cv2.resize()`**:  
  이미지를 크기 절반으로 줄입니다. 가로와 세로 크기를 각각 절반으로 조정합니다.

- **`img.shape[:2]`**:  
  이미지의 높이와 너비를 얻어옵니다. `img.shape`는 `(height, width, channels)` 형식으로 되어 있습니다.

- **`(w // 2, h // 2)`**:  
  이미지의 중심 좌표를 계산합니다. `w`와 `h`는 이미지의 너비와 높이입니다.

- **`cv2.getRotationMatrix2D()`**:  
  회전 + 스케일 행렬을 생성합니다. 회전각은 30도, 스케일은 0.8로 설정합니다.

- **`M[0, 2] += 80`**:  
  회전 행렬에 평행이동을 추가합니다. x축으로 +80, y축으로 -40만큼 이동합니다.

- **`cv2.warpAffine()`**:  
  이미지에 Affine 변환을 적용합니다. 회전, 스케일, 평행이동이 모두 적용됩니다.

- **`cv2.imshow()`**:  
  원본 이미지와 변환된 이미지를 화면에 표시합니다.

- **`cv2.waitKey(0)`**:  
  키보드 입력을 기다립니다. 아무 키나 누르면 창을 닫습니다.

- **`cv2.destroyAllWindows()`**:  
  모든 OpenCV 창을 닫습니다.
  
### 실행결과
<img width="1494" height="534" alt="스크린샷 2026-03-12 154642" src="https://github.com/user-attachments/assets/fd9d2e71-a9d7-4069-9fb0-8e0876d9d6c8" />

## 03_Stereo Disparity 기반 Depth 추정

### 설명

* 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용해 깊이를 추정
* 두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산하여 물체가 카메라에서 얼마나 떨어져
있는지(depth)를 구할 수 있음

### 요구사항

* 입력 이미지를 그레이스케일로 변환한 뒤 cv2.StereoBM_create()를 사용하여 disparity map 계산
* Disparity > 0인 픽셀만 사용하여 depth map 계산
* ROI Painting, Frog, Teddy 각각에 대해 평균 disparity와 평균 depth를 계산
* 세 ROI 중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해석
* Disparity가 클 수록 물체는 더 가까움
* Disparity map의 결과는 시각화하기 전에 정규화가 필요할 수 있음

### 전체코드
```python
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
```

### 주요코드
# 주요 코드 설명

- **`Path("./outputs")`**:  
  결과 이미지를 저장할 폴더를 생성합니다. 폴더가 없으면 새로 만듭니다.

- **`cv2.imread()`**:  
  좌측과 우측 이미지를 읽어옵니다. 이미지가 없으면 `FileNotFoundError`를 발생시킵니다.

- **`cv2.cvtColor()`**:  
  이미지를 그레이스케일로 변환합니다. 시차 계산을 위해 두 이미지를 그레이스케일로 처리합니다.

- **`cv2.StereoBM_create()`**:  
  시차 계산을 위한 `StereoBM` 객체를 생성합니다. `numDisparities`와 `blockSize`는 시차 계산의 정확도와 범위를 설정합니다.

- **`stereo.compute()`**:  
  두 그레이스케일 이미지를 입력으로 받아 시차 맵을 계산합니다. 결과는 16배로 스케일링된 정수형이므로 부동소수점으로 변환 후 16으로 나눕니다.

- **`depth_map`**:  
  시차를 사용하여 깊이 맵을 계산합니다. 깊이는 `Z = fB / d` 공식에 따라 계산됩니다. 시차 값이 0 이하일 경우 계산이 불가하므로 이를 제외합니다.

- **`rois`**:  
  이미지에서 관심 영역(ROI)을 설정하여, 특정 영역에 대해 시차와 깊이 값을 계산합니다.

- **`np.mean()`**:  
  각 ROI 영역에 대해 평균 시차와 평균 깊이를 계산합니다. 유효한 값만 필터링하여 평균을 구합니다.

- **`cv2.applyColorMap()`**:  
  시차 맵과 깊이 맵을 컬러맵(JET)을 사용하여 시각화합니다. 가까운 값은 빨간색, 먼 값은 파란색으로 표시됩니다.

- **`cv2.rectangle()`**:  
  이미지에 관심 영역(ROI)을 사각형으로 표시합니다.

- **`cv2.putText()`**:  
  이미지에 ROI 이름을 표시합니다.

- **`cv2.imwrite()`**:  
  처리된 이미지를 출력 폴더에 저장합니다.

- **`cv2.imshow()`**:  
  원본 이미지와 변환된 이미지를 시각화하여 화면에 표시합니다.

- **`cv2.waitKey(0)`**:  
  사용자가 키보드를 누를 때까지 화면을 유지합니다.

- **`cv2.destroyAllWindows()`**:  
  모든 OpenCV 창을 닫습니다.

### 실행결과
<img width="1127" height="972" alt="스크린샷 2026-03-12 155342" src="https://github.com/user-attachments/assets/28550ec5-4e24-4a78-9349-80fb3422d00b" />

=== ROI별 평균 Disparity / Depth ===
Painting
  Mean disparity: 19.06
  Mean depth    : 4.42
Frog
  Mean disparity: 33.60
  Mean depth    : 2.51
Teddy
  Mean disparity: 22.42
  Mean depth    : 3.89

=== 거리 비교 ===
가장 가까운 ROI: Frog (2.51)
가장 먼 ROI    : Painting (4.42)

