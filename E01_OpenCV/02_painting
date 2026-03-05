import cv2 as cv
import sys

# 이미지 로드
img = cv.imread('C:/Users/COM/Desktop/computer_vision/image/soccer.jpg')
# 이미지 로드 실패 시 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 마우스로 그리는 중인지 여부
drawing = False
#붓 크기 초기값 5
brush_size = 5

#  마우스 이벤트 콜백 함수
#    event: 발생한 이벤트 종류
#    x, y : 마우스 좌표
#    flags: 마우스 버튼 상태(눌림 여부 등)
def draw(event, x, y, flags, param):
    global drawing, brush_size

    # 좌클릭 누르는 순간: 그리기 시작 + 파란색 점(원) 찍기
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)

    # 우클릭 누르는 순간: 그리기 시작 + 빨간색 점(원) 찍기
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)

    # 마우스 이동 중일 때: 드래그 중이면 계속 원을 찍어서 선처럼 보이게 함
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 왼쪽 버튼 드래그 중이면 파란색으로 연속 그리기
            if flags & cv.EVENT_FLAG_LBUTTON:
                cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
            # 오른쪽 버튼 드래그 중이면 빨간색으로 연속 그리기
            elif flags & cv.EVENT_FLAG_RBUTTON:
                cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)
    # 좌/우 버튼을 떼는 순간: 그리기 종료
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False
    # 변경된 이미지를 즉시 화면에 갱신
    cv.imshow('Drawing', img)

#윈도우 생성 및 최초 이미지 출력
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

# 'Drawing' 창에서 마우스 이벤트가 발생하면 draw() 함수를 호출하도록 등록
cv.setMouseCallback('Drawing', draw)
# 키보드 입력 처리 루프 (요구사항: waitKey(1) & 루프 안에서 처리)
while True:
    key = cv.waitKey(1) & 0xFF
    # '+' 입력 시 붓 크기 1 증가 (최대 15 제한)
    if key == ord('+'):
        brush_size = min(15, brush_size + 1)
    # '-' 입력 시 붓 크기 1 감소 (최소 1 제한)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)
    # 'q' 입력 시 종료
    elif key == ord('q'):
        break
# 창 닫기
cv.destroyAllWindows()
