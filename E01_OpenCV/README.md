## 01_이미지 불러오기 및 그레이스케일 변환
| **설명** |
* OpenCV를사용하여 이미지를 불러오고 화면에 출력
* 원본이미지와 그레이스케일로 변환된 이미지를 나란히 표시
###
|**요구사항**| 
*  cv.imread()를 사용하여 이미지로드
*  cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
*  np.hstack() 함수를 이용해 원본이미지와 그레이스케일 이미지를 가로로 연결하여 출력
* cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무키나 누르면 창이 닫히도록 할것
### 전체코드
```python
import cv2 as cv
import numpy as np
import sys

# 이미지 파일 읽기
img = cv.imread('C:/school/2026-1/computer_vision/image/soccer.jpg')

#이미지가 정상적으로 로드되지 않았으면 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

#BGR 컬러 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#np.hstack()로 원본과 gray(1채널)를 붙이려면 채널 수가 같아야 함
# gray → BGR로 변환 (채널 맞추기)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 가로 연결 (원본 왼쪽, 그레이스케일 오른쪽)
result = np.hstack((img, gray_bgr))
result_small = cv.resize(result,dsize=(0,0),fx=0.5,fy=0.5) #반으로 축소

#결과 창에 출력
cv.imshow('Original vs Gray', result_small)

#키 입력을 기다렸다가 아무키나 누르면 종료
cv.waitKey(0)
#열린 모든 OpenCV창 닫기
cv.destroyAllWindows()
```
## 🛠️ 주요 코드
1. cv.imread() 
   * OpenCV에서 이미지를 불러오는 함수이다.
   * 지정한 경로의 이미지를 읽어 NumPy 배열 형태의 이미지 데이터로 반환한다.
   * OpenCV는 이미지를 기본적으로 BGR 색상 형식으로 읽는다.
2. cv.cvtColor() 
   * 이미지의 색상 공간(Color Space)을 변환하는 함수이다.
   * 여기서는 BGR 컬러 이미지를 그레이스케일 이미지로 변환한다.
   * 그레이스케일 이미지는 밝기 정보만 가지며 단일 채널 이미지이다
3. np.hstack()
   * NumPy의 배열 연결 함수로, 여러 배열을 **가로 방향(horizontal)**으로 이어 붙인다.
   * 이를 통해 원본 이미지와 그레이스케일 이미지를 나란히 표시할 수 있다.
4. cv.imshow()
   * 이미지를 화면 창에 출력하는 함수이다.
   * 첫 번째 인자는 창 이름, 두 번째 인자는 출력할 이미지이다.
5. cv.waitKey()
    * 키 입력을 기다리는 함수이다.
    *  0을 입력하면 사용자가 아무 키나 누를 때까지 프로그램이 대기한다.
6. cv.destroyAllWindows() 
   * OpenCV로 생성된 모든 이미지 창을 닫는 함수이다.
   
## 결과화면 01_gray.py
<img width="1432" height="498" alt="Image" src="https://github.com/user-attachments/assets/f08dc0b3-3cd2-42a7-9d22-3e3f38076e62" /> <br>

## 02_페인팅 붓 크기 조절 기능 추가
| **설명** |
* 마우스 입력으로 이미지 위에 붓질 
* 키보드 입력을 이용해 붓의 크기를 조절하는 기능 추가
###
|**요구사항**| 
*  초기 붓 크기는 5를 사용
* (+) 입력시붓크기1 증가, (-) 입력 시 붓크기1 감소
*  붓 크기는 최소1, 최대 15로 제한
*  좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기
*  q키를 누르면 영상창이 종료

### 전체코드
```python
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
```
  
## 🛠️ 주요 코드
1. cv.setMouseCallback('Drawing', draw)
   * 특정 창(여기서는 'Drawing')에서 발생하는 마우스 이벤트를 처리할 함수를 등록하는 부분이다.
   * 등록해두면 클릭/드래그/이동 같은 이벤트가 생길 때마다 draw()가 자동으로 호출됨.
2. draw(event, x, y, flags, param) 콜백 함수
   * event로 “지금 무슨 일이 벌어졌는지(좌클릭 다운/업, 우클릭 다운/업, 이동 등)”를 구분하고,
   x, y 좌표에 cv.circle()로 원을 찍어서 붓질 효과를 만든다
3. cv.circle(img, (x, y), brush_size, color, -1)
   * 현재 마우스 위치 (x, y)에 반지름 = brush_size인 원을 그림.
   * 마지막 인자 -1 은 원을 채워서 그림.
4. EVENT_MOUSEMOVE + drawing + flags
   * 드래그로 연속 그리기 핵심
   * 클릭하면 drawing = True
   * 이동 이벤트가 계속 들어오면 그때마다 원을 찍음 → 선처럼 보임
   * flags & EVENT_FLAG_LBUTTON / RBUTTON으로 어떤 버튼 드래그인지 구분
6. cv.waitKey(1)로 키보드 입력 처리
   * 루프에서 계속 키 입력을 확인
   * +면 붓 크기 1 증가 - 시 붓 크기 1 감소 q면 종

## 결과화면 02_painting.py
<img width="1433" height="947" alt="Image" src="https://github.com/user-attachments/assets/247aa64a-fa7a-4468-a57c-e3161f1f0eed" /> <br>

## 03_마우스로 영역 선택 및 ROI(관심영역) 추출
| **설명** |
* 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택 
* 선택한 영역만 따로 저장하거나 표시
###
|**요구사항**| 
*  이미지를 불러오고 화면에 출력 
*  cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
*  마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
* r 키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
* s 키를 누르면 선택한 영역을 이미지 파일로 저장

### 전체코드
```python
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
```

## 🛠️ 주요 코드
1. cv.setMouseCallback("image", select_roi)
   * 'image' 창에서 발생하는 **마우스 이벤트(클릭/드래그/이동 등)**를 select_roi() 함수로 전달하도록 등록하는 역할.
2. 드래그 중 사각형 “미리보기” (EVENT_MOUSEMOVE)
   * temp = clone.copy()
   * cv.rectangle(temp, start, (x, y), (0, 255, 0), 2)
   * cv.imshow("image", temp)
   * 드래그 할 때마다 원본을 직접 그려버리면 화면이 누적되어 지저분해짐
   * clone.copy()를 통해 임시이미지를 만들고 "현재 드래그 상태"를 보여줌.
3. 마우스 놓는 순간 ROI 추출 (numpy 슬라이싱)
   * roi = clone[y1:y2, x1:x2]
   * 이미지가 배열에서 원하는 영역만 잘라내는 핵심
4. 좌표 정렬 (드래그 방향 보정)
   * x1, y1 = min(start[0], end[0]), min(start[1], end[1])
   * x2, y2 = max(start[0], end[0]), max(start[1], end[1])
   * 실제로는 왼쪽/위쪽으로도 드래그 할 수 있어서 항상 (좌상단, 우하단) 형태로 정렬해야함
5. cv.imwrite()
   * 현재 ROI 이미지를 파일로 저장하는 함수

## 결과화면 03_mouse.py
<img width="1429" height="950" alt="Image" src="https://github.com/user-attachments/assets/139cd777-4ae8-4082-8209-6e104aefc7f5" /> <br>
