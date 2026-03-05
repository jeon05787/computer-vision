import cv2 as cv
import numpy as np
import sys

img = cv.imread('C:/Users/COM/Desktop/computer_vision/image/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# gray → BGR로 변환 (채널 맞추기)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 가로 연결
result = np.hstack((img, gray_bgr))

cv.imshow('Original vs Gray', result)

cv.waitKey(0)
cv.destroyAllWindows()
