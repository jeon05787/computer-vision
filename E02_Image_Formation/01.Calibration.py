import cv2
import numpy as np
import glob

# 체크보드 내부 코너의 개수 (가로 9개, 세로 6개로 설정)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (단위: 밀리미터)
square_size = 25.0

# 코너 정밀화 조건 (정확도를 높이기 위한 최대 반복 횟수 및 정확도 기준 설정)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성: 체크보드의 각 코너의 실제 3D 좌표를 생성
# z 값은 0으로 고정, x, y 값은 각 칸마다 증가
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # 실제 크기(mm)를 반영하여 objp 크기 설정

# 실제 좌표와 이미지 좌표를 저장할 리스트 생성
objpoints = []  # 3D 좌표 저장
imgpoints = []  # 2D 좌표 저장

# 이미지 경로 불러오기 (파일 이름이 'left'로 시작하는 모든 이미지 파일)
images = glob.glob("image/calibration_images/left*.jpg")

img_size = None  # 이미지 크기 초기화

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)  # 이미지 읽기

    if img is None:
        print(f"이미지를 읽을 수 없음: {fname}")
        continue  # 이미지 읽기 실패 시 넘어감

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환
    img_size = gray.shape[::-1]  # 이미지 크기 저장 (가로, 세로 순서로 설정)

    # cv2.findChessboardCorners 함수로 체크보드 코너를 찾는다
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:  # 코너가 검출되었으면
        # 코너의 정확도를 높이기 위해 서브픽셀 정밀화 수행
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 실제 3D 좌표(objpoints)와 이미지 좌표(imgpoints)를 리스트에 추가
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 검출된 코너를 이미지에 그려서 확인
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Corners", img)  # 코너 표시된 이미지 출력
        cv2.waitKey(300)  # 300ms 동안 이미지 확인
    else:
        print(f"코너 검출 실패: {fname}")  # 코너 검출 실패 시 출력

cv2.destroyAllWindows()  # 모든 창 닫기

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# cv2.calibrateCamera 함수로 카메라의 내부 파라미터를 추정
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,  # 3D 좌표 (실제 좌표)
    imgpoints,  # 2D 좌표 (이미지 좌표)
    img_size,  # 이미지 크기
    None,  # 카메라 행렬 초기값 (None으로 두면 자동 계산)
    None   # 왜곡 계수 초기값 (None으로 두면 자동 계산)
)

# 카메라 행렬과 왜곡 계수 출력
print("Camera Matrix K:")
print(K)  # 카메라 행렬 (Intrinsic Parameters)

print("\nDistortion Coefficients:")
print(dist)  # 왜곡 계수 (Distortion Coefficients)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 왜곡된 이미지를 보정하여 원본 이미지와 비교하는 작업
for fname in images:
    img = cv2.imread(fname)  # 이미지 읽기

    if img is None:
        continue  # 이미지가 없으면 넘어감

    # cv2.undistort 함수로 왜곡 보정
    undistorted = cv2.undistort(img, K, dist)

    # 원본 이미지와 보정된 이미지를 나란히 합쳐서 출력
    combined = np.hstack((img, undistorted))

    # 두 이미지를 화면에 출력
    cv2.imshow("Original (Left) vs Undistorted (Right)", combined)
    cv2.waitKey(0)  # 키 입력을 기다림

cv2.destroyAllWindows()  # 모든 창 닫기
