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
