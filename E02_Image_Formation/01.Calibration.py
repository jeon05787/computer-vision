import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 실제 좌표와 이미지 좌표 저장
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 검출 결과 시각화
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
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        continue

    undistorted = cv2.undistort(img, K, dist)

    # 원본 / 보정 결과 나란히 붙이기
    combined = np.hstack((img, undistorted))

    cv2.imshow("Original (Left) vs Undistorted (Right)", combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()
