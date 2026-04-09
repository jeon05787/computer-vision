"""
SORT 알고리즘을 활용한 다중 객체 추적기
"""

import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ──────────────────────────────────────────────
# 1. 유틸리티 함수
# ──────────────────────────────────────────────

def iou(bb_test, bb_gt):
    # 두 바운딩 박스의 좌상단/우하단 좌표를 이용해
    # 겹치는 영역(intersection)의 좌표를 계산
    xx1 = max(bb_test[0], bb_gt[0]);  yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2]);  yy2 = min(bb_test[3], bb_gt[3])

    # 겹치는 영역의 너비와 높이
    # 겹치지 않으면 음수가 될 수 있으므로 0과 비교해서 방지
    w = max(0., xx2 - xx1);           h = max(0., yy2 - yy1)

    # 교집합 넓이
    inter = w * h

    # 각각의 박스 넓이 계산
    area_a = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_b = (bb_gt[2]-bb_gt[0])     * (bb_gt[3]-bb_gt[1])

    # 합집합 넓이 = A 넓이 + B 넓이 - 교집합 넓이
    union  = area_a + area_b - inter

    # IoU = 교집합 / 합집합
    # union이 0이면 나눗셈 오류를 피하기 위해 0.0 반환
    return inter / union if union > 0 else 0.0

def bbox_to_z(bbox):
    # [x1, y1, x2, y2] 형식의 바운딩 박스를
    # 칼만 필터 측정 벡터 z = [중심 x, 중심 y, 면적 s, 종횡비 r] 형식으로 변환
    w = bbox[2]-bbox[0];  h = bbox[3]-bbox[1]
    return np.array([[bbox[0]+w/2.], [bbox[1]+h/2.], [w*h], [w/float(h)]])

def z_to_bbox(x):
    # 칼만 필터의 상태 벡터 x에서
    # [중심 x, 중심 y, 면적 s, 종횡비 r] 정보를 꺼내
    # 다시 [x1, y1, x2, y2] 형식의 박스로 복원
    x = x.flatten()

    # 면적(s)과 종횡비(r)를 바탕으로 너비/높이 복원
    # abs를 씌워 음수 때문에 sqrt 에러 나는 것 방지
    w = float(np.sqrt(abs(x[2] * x[3])))
    h = float(x[2]) / w if w != 0 else 0

    # 중심 좌표를 좌상단/우하단 좌표로 변환해서 반환
    return [float(x[0])-w/2., float(x[1])-h/2.,
            float(x[0])+w/2., float(x[1])+h/2.]

def associate(dets, trks, iou_thr=0.3):
    # dets: 현재 프레임의 검출 결과들
    # trks: 이전까지 존재하던 트래커들의 예측 박스
    # iou_thr: 검출 결과와 트래커를 같은 객체로 볼 최소 IoU 기준

    # 트래커가 하나도 없으면
    # 모든 검출 결과는 unmatched detection으로 처리
    if len(trks) == 0:
        return [], list(range(len(dets))), []

    # 검출 박스와 트래커 예측 박스 사이의 IoU 행렬 생성
    # mat[d][t] = det d 와 trk t 의 IoU
    mat = np.array([[iou(d, t) for t in trks] for d in dets])

    # Hungarian Algorithm(선형 할당)을 이용해
    # 전체적으로 가장 좋은 매칭 조합을 찾음
    # linear_sum_assignment는 최소 비용을 찾으므로
    # IoU를 최대화하려면 -mat를 넣음
    ri, ci = linear_sum_assignment(-mat)

    # matched: 최종적으로 매칭된 (det index, trk index)
    # u_dets: 매칭되지 않은 detection들
    # u_trks: 매칭되지 않은 tracker들
    matched, u_dets, u_trks = [], [], []

    # Hungarian 결과를 돌면서 IoU 기준 이상이면 진짜 매칭으로 인정
    # 기준 미만이면 매칭 실패로 간주
    for d, t in zip(ri, ci):
        if mat[d, t] >= iou_thr:
            matched.append((d, t))
        else:
            u_dets.append(d); u_trks.append(t)

    # 아예 할당 자체가 되지 않은 detection들을 unmatched에 추가
    u_dets += [d for d in range(len(dets)) if d not in ri]

    # 아예 할당 자체가 되지 않은 tracker들을 unmatched에 추가
    u_trks += [t for t in range(len(trks)) if t not in ci]

    return matched, u_dets, u_trks

# ──────────────────────────────────────────────
# 2. 칼만 필터 기반 단일 객체 추적기
# ──────────────────────────────────────────────

class KalmanBoxTracker:
    # 전체 tracker 인스턴스 수를 세기 위한 클래스 변수
    # 생성될 때마다 ID 부여에 사용
    count = 0

    def __init__(self, bbox):
        # 상태 벡터 차원 7, 측정 벡터 차원 4의 칼만 필터 생성
        # 일반적으로 SORT에서는
        # 상태: [x, y, s, r, vx, vy, vs]
        # 측정: [x, y, s, r]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태 전이 행렬 F
        # 위치/면적은 속도에 따라 다음 프레임으로 이동한다고 가정
        self.kf.F = np.eye(7); self.kf.F[:3, 4:7] = np.eye(3)

        # 관측 행렬 H
        # 측정값은 상태 벡터의 앞 4개 요소([x, y, s, r])만 관측 가능
        self.kf.H = np.eye(4, 7)

        # 측정 잡음 공분산 R 조정
        # 면적(s), 종횡비(r) 측정에 조금 더 큰 불확실성 부여
        self.kf.R[2:, 2:] *= 10.

        # 오차 공분산 P 조정
        # 속도 성분은 처음엔 잘 모르므로 큰 불확실성 부여
        self.kf.P[4:, 4:] *= 1000.

        # 전체 공분산을 조금 키워 초기 불확실성 반영
        self.kf.P *= 10.

        # 프로세스 잡음 공분산 Q 조정
        # 속도 관련 변화량에 대한 잡음을 작게 설정
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 상태의 앞 4개를 첫 bbox 측정값으로 세팅
        self.kf.x[:4] = bbox_to_z(bbox)

        # tracker 고유 ID 부여
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1

        # hits: 지금까지 업데이트된 총 횟수
        # hit_streak: 연속으로 잘 매칭된 횟수
        # time_since_update: 마지막 업데이트 이후 지난 프레임 수
        self.hits = self.hit_streak = self.time_since_update = 0

    def predict(self):
        # 비정상적인 상태 방지용 조건
        # 면적 관련 값과 속도 조합이 음수가 되는 경우를 막기 위해 보정
        if self.kf.x[6] + self.kf.x[2] <= 0: self.kf.x[6] = 0.

        # 칼만 필터 예측 단계 수행
        self.kf.predict()

        # 업데이트 없이 한 프레임이 지났으므로 1 증가
        self.time_since_update += 1

        # 한 번이라도 update 없이 predict만 거쳤다면
        # 연속 적중 기록(hit_streak)은 끊긴 것으로 처리
        if self.time_since_update > 0: self.hit_streak = 0

        # 예측된 상태를 bbox 형태로 반환
        return z_to_bbox(self.kf.x)

    def update(self, bbox):
        # detection과 성공적으로 매칭되었으므로
        # 마지막 업데이트 이후 시간 초기화
        self.time_since_update = 0

        # 총 적중 수, 연속 적중 수 증가
        self.hits += 1; self.hit_streak += 1

        # 실제 검출값을 이용해 칼만 필터 업데이트
        self.kf.update(bbox_to_z(bbox))

    def get_state(self):
        # 현재 칼만 필터 상태를 bbox 형식으로 반환
        return z_to_bbox(self.kf.x)

# ──────────────────────────────────────────────
# 3. SORT 추적기
# ──────────────────────────────────────────────

class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_thr=0.3):
        # max_age:
        # detection과 매칭되지 않아도 tracker를 몇 프레임까지 유지할지
        self.max_age = max_age

        # min_hits:
        # tracker가 화면에 안정적으로 나타났다고 판단하기 위한
        # 최소 연속 검출 횟수
        self.min_hits = min_hits

        # detection-tracker 매칭에 사용할 IoU 임계값
        self.iou_thr = iou_thr

        # 현재 살아있는 tracker 리스트
        self.trackers = []

        # 현재까지 처리한 프레임 수
        self.frame_count = 0

    def update(self, dets):
        # 프레임 하나 처리 시작
        self.frame_count += 1

        # 각 tracker가 현재 프레임에서 어디 있을지 예측
        trk_preds, bad = [], []
        for i, t in enumerate(self.trackers):
            p = t.predict()

            # 예측값에 NaN이 있으면 비정상 tracker로 간주 후 제거 대상에 기록
            if np.any(np.isnan(p)): bad.append(i)
            else: trk_preds.append(p)

        # 뒤에서부터 제거해야 인덱스 꼬임이 없음
        for i in reversed(bad): self.trackers.pop(i)

        # detection 배열에서 bbox 좌표(x1,y1,x2,y2)만 추출
        # dets는 [x1,y1,x2,y2,conf] 구조라고 가정
        det_boxes = dets[:, :4].tolist() if len(dets) else []

        # detection과 tracker 예측값을 IoU 기준으로 매칭
        matched, u_dets, _ = associate(det_boxes, trk_preds, self.iou_thr)

        # 매칭된 detection으로 기존 tracker 업데이트
        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        # 매칭되지 않은 detection은 새 tracker 생성
        for d in u_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4]))

        # 최종 출력용 결과 리스트
        results = []

        # tracker를 뒤에서부터 순회
        for t in reversed(self.trackers):
            # 바로 직전 프레임에서 업데이트 되었고
            # 충분히 안정화되었거나(min_hits 이상)
            # 혹은 아직 초기 프레임 단계라면 결과로 출력
            if (t.time_since_update < 1 and
                    (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                state = t.get_state()
                results.append([float(state[0]), float(state[1]),
                                 float(state[2]), float(state[3]),
                                 float(t.id + 1)])

            # 너무 오래 업데이트되지 않은 tracker는 제거
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)

        # 결과가 있으면 ndarray로 반환, 없으면 빈 배열 반환
        return np.array(results) if results else np.empty((0, 5))

# ──────────────────────────────────────────────
# 4. YOLOv3 검출기
# ──────────────────────────────────────────────

class YOLOv3Detector:
    def __init__(self, weights, cfg, conf_thr=0.5, nms_thr=0.4):
        # YOLOv3 네트워크 로드
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)

        # OpenCV DNN 백엔드와 CPU 타깃 설정
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # YOLO 출력 레이어 이름 가져오기
        # getUnconnectedOutLayers()는 최종 출력층 인덱스를 반환하므로
        # 실제 레이어 이름 리스트에서 찾아 매핑
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i-1] for i in
                           self.net.getUnconnectedOutLayers().flatten()]

        # confidence threshold:
        # 객체라고 인정할 최소 신뢰도
        self.conf_thr = conf_thr

        # NMS threshold:
        # 겹치는 박스 제거 시 사용할 기준
        self.nms_thr  = nms_thr

    def detect(self, frame):
        # 현재 프레임의 높이/너비
        H, W = frame.shape[:2]

        # 이미지를 YOLO 입력 형태(blob)로 변환
        # 1/255. : 픽셀 정규화
        # (416,416) : YOLOv3 입력 크기
        # swapRB=True : BGR(OpenCV) -> RGB 변환
        # crop=False : 이미지 비율 유지
        blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416),
                                     swapRB=True, crop=False)

        # 네트워크 입력 설정
        self.net.setInput(blob)

        # 출력 레이어 forward 수행
        outs = self.net.forward(self.out_layers)

        # 원시 검출 결과 저장용
        boxes, confs = [], []

        # 각 출력 레이어를 순회
        for out in outs:
            # 각 detection 벡터를 순회
            for det in out:
                # 클래스 확률들 중 최대값을 confidence로 사용
                conf = float(np.max(det[5:]))

                # confidence threshold 미만이면 무시
                if conf < self.conf_thr: continue

                # YOLO 출력은 중심좌표/너비/높이가 0~1 비율값이므로
                # 실제 이미지 좌표계로 변환
                cx, cy, bw, bh = det[0]*W, det[1]*H, det[2]*W, det[3]*H

                # OpenCV NMSBoxes는 [x, y, w, h] 형식 사용
                boxes.append([int(cx-bw/2), int(cy-bh/2), int(bw), int(bh)])
                confs.append(conf)

        # NMS(Non-Maximum Suppression) 적용
        # 겹치는 박스들 중 더 신뢰도 높은 것만 남김
        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thr, self.nms_thr)

        # 최종 반환용 결과
        res = []

        # NMS 후 살아남은 박스만 [x1, y1, x2, y2, conf] 형태로 변환
        for i in (idx.flatten() if len(idx) else []):
            x, y, w, h = boxes[i]
            res.append([x, y, x+w, y+h, confs[i]])

        return res

# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────

# 트래커 ID별로 박스 색을 다르게 보이게 하기 위한 색상 목록
COLORS = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
]

def draw_tracks(frame, tracks):
    # tracks: [x1, y1, x2, y2, id] 배열들
    for t in tracks:
        # 좌표와 ID를 정수형으로 변환
        x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])

        # ID에 따라 색상 순환 선택
        c = COLORS[tid % len(COLORS)]

        # 객체 바운딩 박스 그리기
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)

        # 라벨 문자열 생성
        label = f"ID {tid}"

        # 라벨 텍스트 크기 계산
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # 텍스트 배경 박스 먼저 그림
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), c, -1)

        # 실제 텍스트 그리기
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # 박스와 라벨이 그려진 프레임 반환
    return frame

# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────

def main():
    # 현재 파이썬 파일이 있는 디렉토리 경로
    BASE    = os.path.dirname(os.path.abspath(__file__))

    # 입력 비디오 파일 경로
    VIDEO   = os.path.join(BASE, "slow_traffic_small.mp4")

    # YOLO 가중치 파일 경로
    WEIGHTS = os.path.join(BASE, "yolov3.weights")

    # YOLO 설정 파일 경로
    CFG     = os.path.join(BASE, "yolov3.cfg")

    # YOLOv3 검출기 생성
    detector = YOLOv3Detector(WEIGHTS, CFG, conf_thr=0.5, nms_thr=0.4)

    # SORT 추적기 생성
    # max_age=5  : 최대 5프레임 동안 검출 안 되어도 tracker 유지
    # min_hits=1 : 1번만 잡혀도 바로 출력 허용
    # iou_thr=0.3: detection-tracker 매칭 기준
    tracker  = Sort(max_age=5, min_hits=1, iou_thr=0.3)

    # 비디오 열기
    cap = cv2.VideoCapture(VIDEO)

    # 비디오가 정상적으로 열리지 않으면 에러 출력 후 종료
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO}")
        return

    print("[INFO] 실행 중 ... 'q' 키로 종료")
    while True:
        # 프레임 1장 읽기
        ret, frame = cap.read()

        # 더 이상 읽을 프레임이 없으면 종료
        if not ret:
            print("[INFO] 영상 종료")
            break

        # 현재 프레임에서 객체 검출 수행
        dets = detector.detect(frame)

        # dets가 있으면 ndarray로 변환
        # 없으면 빈 (0,5) 배열 생성
        # 형식은 [x1, y1, x2, y2, conf]
        det_arr = (np.array(dets) if dets else np.empty((0, 5)))

        # 검출 결과를 SORT tracker에 넣어 추적 결과 갱신
        tracks = tracker.update(det_arr)

        # 추적 결과를 프레임 위에 시각화
        frame = draw_tracks(frame, tracks)

        # 화면에 출력
        cv2.imshow("SORT Tracker", frame)

        # 1ms 대기하면서 'q' 입력 감지
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 이 파일이 직접 실행될 때만 main() 호출
if __name__ == "__main__":
    main()
