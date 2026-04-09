"""
Mediapipe FaceMesh를 활용한 얼굴 랜드마크 추출 및 시각화
- 468개 랜드마크를 실시간 영상에 점으로 표시
- ESC 키로 종료
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import urllib.request
import os

# 모델 파일 경로 지정
MODEL_PATH = "face_landmarker.task"

# 모델 파일 다운로드 함수
def download_model():
    # 모델 파일이 존재하지 않으면 다운로드
    if not os.path.exists(MODEL_PATH):
        print("[INFO] 모델 다운로드 중...")
        # 모델 URL
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        # URL로부터 모델 다운로드
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[INFO] 다운로드 완료")

def main():
    # 모델 다운로드
    download_model()

    # FaceLandmarker 초기화 (Mediapipe 새 API 사용)
    options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),  # 모델 경로 지정
        running_mode=RunningMode.IMAGE,  # 이미지를 처리하는 모드
        num_faces=2,  # 최대 두 개의 얼굴을 동시에 추적
        min_face_detection_confidence=0.5,  # 얼굴 감지 최소 신뢰도
        min_tracking_confidence=0.5  # 얼굴 추적 최소 신뢰도
    )

    # 웹캠 캡처
    cap = cv2.VideoCapture(0)  # 0번 카메라 열기
    if not cap.isOpened():  # 웹캠 열기 실패 시 종료
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    print("[INFO] 실행 중 ... ESC 키로 종료")

    # FaceLandmarker 생성
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:  # 프레임을 제대로 읽지 못하면 종료
                break

            # 영상 크기 (세로, 가로)
            H, W = frame.shape[:2]

            # BGR을 RGB로 변환 (Mediapipe는 RGB 이미지를 사용)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # 얼굴 랜드마크 검출
            results = landmarker.detect(mp_image)

            # 얼굴 랜드마크가 검출되면
            if results.face_landmarks:
                # 각 얼굴에 대해 랜드마크 그리기
                for face_landmarks in results.face_landmarks:
                    for lm in face_landmarks:
                        # 랜드마크 좌표는 0~1로 정규화 되어 있음
                        # 이를 화면의 픽셀 좌표로 변환
                        x = int(lm.x * W)  # x 좌표
                        y = int(lm.y * H)  # y 좌표
                        # 랜드마크 위치에 작은 원을 그려서 시각화
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 시각화된 영상 표시
            cv2.imshow("Face Landmark", frame)

            # ESC 키 입력 시 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 프로그램 실행 시작점
if __name__ == "__main__":
    main()
