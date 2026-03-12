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
