## 01_이미지불러오기 및 그레이스케일변환
* **설명:** <br> • OpenCV를사용하여이미지를불러오고화면에출력 <br> • 원본이미지와그레이스케일로변환된이미지를나란히표시<br>
* **요구사항:** <br> • cv.imread()를 사용하여 이미지 로드 <br>
• cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환 <br>
• np.hstack() 함수를 이용해 원본이미지와 그레이스케일 이미지를 가로로 연결하여 출력<br>
• cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무키나 누르면 창이 닫히도록 할 것 <br>

## 🛠️ 주요 코드

## 결과화면 01_gray.py
<img width="1432" height="498" alt="Image" src="https://github.com/user-attachments/assets/f08dc0b3-3cd2-42a7-9d22-3e3f38076e62" /> <br>

## 02_페인팅 붓 크기 조절 기능 추가
* **설명:** <br> •마우스입력으로이미지위에붓질 <br> • 키보드입력을이용해붓의크기를조절하는기능추가<br>
* **요구사항:** <br> •  초기붓크기는5를사용 <br>
•  + 입력시붓크기1 증가, -입력시붓크기1 감소 <br>
•붓크기는최소1, 최대15로제한 <br>
• 좌클릭=파란색, 우클릭=빨간색, 드래그로연속그리기 <br>
• q키를누르면영상창이종료 <br>

## 🛠️ 주요 코드
## 결과화면 02_painting.py
<img width="1433" height="947" alt="Image" src="https://github.com/user-attachments/assets/247aa64a-fa7a-4468-a57c-e3161f1f0eed" /> <br>

## 03_마우스로 영역 선택 및 ROI(관심영역) 추출
| **설명** |
* 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택 
* 선택한 영역만 따로 저장하거나 표시
  <br>
|**요구사항**| 
*  이미지를 불러오고 화면에 출력 
*  cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
*  마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
*• r 키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
*• s 키를 누르면 선택한 영역을 이미지 파일로 저장
## 🛠️ 주요 코드

## 결과화면 03_mouse.py
<img width="1429" height="950" alt="Image" src="https://github.com/user-attachments/assets/139cd777-4ae8-4082-8209-6e104aefc7f5" /> <br>
