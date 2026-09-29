import cv2 as cv
import sys

# 이미지 불러오기
img = cv.imread('C:/Users/COM/Desktop/computer_vision/image/soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 원본 보존용 복사본 (리셋/ROI 추출에 사용)
clone = img.copy()

#  ROI 선택에 필요한 변수들
start = None   # 드래그 시작점 (x, y)
end = None     # 드래그 끝점 (x, y)
drawing = False  # 드래그 중 여부

def select_roi(event, x, y, flags, param):
    """
    마우스 이벤트 콜백:
    - 왼쪽 버튼 누르면 시작점 저장
    - 드래그 중이면 임시 이미지에 사각형을 계속 그려서 시각화
    - 버튼을 놓으면 ROI를 잘라서 별도 창에 표시
    """
    global start, end, drawing, img, clone

    # 왼쪽 버튼 눌렀을 때: 시작점 저장 + 드래그 시작
    if event == cv.EVENT_LBUTTONDOWN:
        start = (x, y)
        drawing = True

    # 마우스 이동 중: 드래그 중이면 사각형을 "미리보기"로 표시
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing and start is not None:
            temp = clone.copy()  # 원본 위에 덧그리지 않기 위해 임시 복사본 사용
            cv.rectangle(temp, start, (x, y), (0, 255, 0), 2)
            cv.imshow("image", temp)

    # 왼쪽 버튼 뗐을 때: 끝점 저장 + 드래그 종료 + ROI 추출/표시
    elif event == cv.EVENT_LBUTTONUP:
        end = (x, y)
        drawing = False

        # 좌표 정렬 (드래그 방향이 어느 쪽이든 ROI가 정상 추출되도록)
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])

        # ROI 추출 (numpy 슬라이싱)
        roi = clone[y1:y2, x1:x2]

        # 원본 이미지에는 선택 영역 사각형 확정해서 보여주기
        img = clone.copy()
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow("image", img)

        # 요구사항: 마우스를 놓으면 ROI를 별도 창에 출력
        if roi.size > 0:
            cv.imshow("ROI", roi)

# 윈도우 생성 및 마우스 콜백 등록
cv.namedWindow("image")
cv.setMouseCallback("image", select_roi)

# 키 입력 처리 루프
while True:
    cv.imshow("image", img)
    key = cv.waitKey(1) & 0xFF

    # r 키: 리셋 (처음 상태로 되돌리기)
    if key == ord('r'):
        img = clone.copy()
        start = None
        end = None
        # ROI 창도 떠있으면 닫아주기 (선택)
        cv.destroyWindow("ROI") if cv.getWindowProperty("ROI", 0) >= 0 else None

    # s 키: ROI 저장 (roi.png로 저장)
    elif key == ord('s'):
        if start and end:
            x1, y1 = min(start[0], end[0]), min(start[1], end[1])
            x2, y2 = max(start[0], end[0]), max(start[1], end[1])
            roi = clone[y1:y2, x1:x2]

            if roi.size > 0:
                cv.imwrite("roi.png", roi)
                print("ROI saved as roi.png")

    # q 키: 종료
    elif key == ord('q'):
        break

cv.destroyAllWindows()
