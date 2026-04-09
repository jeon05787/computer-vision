## 01_SORT 알고리즘을활용한다중객체추적기구현

### 설명

* 이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중객체를 실시간으로 추적하는 프로그램을 구현합니다
* 이를 통해 객체 추적의 기본개념과 SORT 알고리즘의 적용방법을 학습할 수 있습니다

### 요구사항

* 객체 검출기 구현: YOLOv3와 같은 사전 훈련된 객체검출모델을 사용하여 각 프레임에서 객체를 검출합니다.
* mathworks.comSORT 추적기 초기화: 검출된 객체의 경계상자를 입력으로 받아 SORT 추적기를 초기화합니다.
* 객체추적: 각 프레임마다 검출된 객체와 기존추적객체를 연관시켜 추적을 유지합니다.
* 결과시각화: 추적된 각 객체에 고유ID를 부여하고, 해당ID와 경계상자를 비디오 프레임에 표시하여 실시간으로 출
력합니다
* 객체검출: OpenCV의 DNN 모듈을 사용하여 YOLOv3 모델을 로드하고, 각 프레임에서 객체를 검출할 수 있습니다.
* SORT 알고리즘: SORT 알고리즘은 칼만필터와 헝가리안알고리즘을 사용하여 객체의 상태를 예측하고, 데이터연
관을 수행합니다.
* 추적성능향상: 객체의 appearance 정보를 활용하는 Deep SORT와 같은 확장된 알고리즘을 사용하면 추적성능을
향상 시킬 수 있습니다.​

### 전체코드
```python
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
    xx1 = max(bb_test[0], bb_gt[0]);  yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2]);  yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1);           h = max(0., yy2 - yy1)
    inter = w * h
    area_a = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_b = (bb_gt[2]-bb_gt[0])     * (bb_gt[3]-bb_gt[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def bbox_to_z(bbox):
    w = bbox[2]-bbox[0];  h = bbox[3]-bbox[1]
    return np.array([[bbox[0]+w/2.], [bbox[1]+h/2.], [w*h], [w/float(h)]])

def z_to_bbox(x):
    x = x.flatten()
    w = float(np.sqrt(abs(x[2] * x[3])))
    h = float(x[2]) / w if w != 0 else 0
    return [float(x[0])-w/2., float(x[1])-h/2.,
            float(x[0])+w/2., float(x[1])+h/2.]

def associate(dets, trks, iou_thr=0.3):
    if len(trks) == 0:
        return [], list(range(len(dets))), []
    mat = np.array([[iou(d, t) for t in trks] for d in dets])
    ri, ci = linear_sum_assignment(-mat)
    matched, u_dets, u_trks = [], [], []
    for d, t in zip(ri, ci):
        if mat[d, t] >= iou_thr:
            matched.append((d, t))
        else:
            u_dets.append(d); u_trks.append(t)
    u_dets += [d for d in range(len(dets)) if d not in ri]
    u_trks += [t for t in range(len(trks)) if t not in ci]
    return matched, u_dets, u_trks

# ──────────────────────────────────────────────
# 2. 칼만 필터 기반 단일 객체 추적기
# ──────────────────────────────────────────────

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7); self.kf.F[:3, 4:7] = np.eye(3)
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10.; self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.;          self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.hits = self.hit_streak = self.time_since_update = 0

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0: self.kf.x[6] = 0.
        self.kf.predict()
        self.time_since_update += 1
        if self.time_since_update > 0: self.hit_streak = 0
        return z_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox))

    def get_state(self):
        return z_to_bbox(self.kf.x)

# ──────────────────────────────────────────────
# 3. SORT 추적기
# ──────────────────────────────────────────────

class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_thr=0.3):
        self.max_age = max_age; self.min_hits = min_hits
        self.iou_thr = iou_thr; self.trackers = []; self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1

        trk_preds, bad = [], []
        for i, t in enumerate(self.trackers):
            p = t.predict()
            if np.any(np.isnan(p)): bad.append(i)
            else: trk_preds.append(p)
        for i in reversed(bad): self.trackers.pop(i)

        det_boxes = dets[:, :4].tolist() if len(dets) else []
        matched, u_dets, _ = associate(det_boxes, trk_preds, self.iou_thr)

        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        for d in u_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4]))

        results = []
        for t in reversed(self.trackers):
            if (t.time_since_update < 1 and
                    (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                state = t.get_state()
                results.append([float(state[0]), float(state[1]),
                                 float(state[2]), float(state[3]),
                                 float(t.id + 1)])
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)
        return np.array(results) if results else np.empty((0, 5))

# ──────────────────────────────────────────────
# 4. YOLOv3 검출기
# ──────────────────────────────────────────────

class YOLOv3Detector:
    def __init__(self, weights, cfg, conf_thr=0.5, nms_thr=0.4):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i-1] for i in
                           self.net.getUnconnectedOutLayers().flatten()]
        self.conf_thr = conf_thr
        self.nms_thr  = nms_thr

    def detect(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)

        boxes, confs = [], []
        for out in outs:
            for det in out:
                conf = float(np.max(det[5:]))
                if conf < self.conf_thr: continue
                cx, cy, bw, bh = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                boxes.append([int(cx-bw/2), int(cy-bh/2), int(bw), int(bh)])
                confs.append(conf)

        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thr, self.nms_thr)
        res = []
        for i in (idx.flatten() if len(idx) else []):
            x, y, w, h = boxes[i]
            res.append([x, y, x+w, y+h, confs[i]])
        return res

# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────

COLORS = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
]

def draw_tracks(frame, tracks):
    for t in tracks:
        x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])
        c = COLORS[tid % len(COLORS)]
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
        label = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), c, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return frame

# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────

def main():
    BASE    = os.path.dirname(os.path.abspath(__file__))
    VIDEO   = os.path.join(BASE, "slow_traffic_small.mp4")
    WEIGHTS = os.path.join(BASE, "yolov3.weights")
    CFG     = os.path.join(BASE, "yolov3.cfg")

    detector = YOLOv3Detector(WEIGHTS, CFG, conf_thr=0.5, nms_thr=0.4)
    tracker  = Sort(max_age=5, min_hits=1, iou_thr=0.3)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO}")
        return

    print("[INFO] 실행 중 ... 'q' 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 영상 종료")
            break

        dets = detector.detect(frame)
        det_arr = (np.array(dets) if dets else np.empty((0, 5)))

        tracks = tracker.update(det_arr)

        frame = draw_tracks(frame, tracks)
        cv2.imshow("SORT Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```

### 주요코드
- **`코드`**:  
  

-  **`코드`**:  
  

-  **`코드`**:  

- **`코드`**:  

-  **`코드`**:  

### 실행결과

## 02_Mediapipe를 활용한 얼굴랜드 마크 추출 및 시각화

### 설명

* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개랜드마크를 추출하고, 이를 실시간영상에 시각화하는 프로그
램을 구현합니다.

### 요구사항

* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴랜드마크검출기를 초기화합니다.
* OpenCV를 사용하여 웹캠으로부터 실시간영상을 캡처합니다.
* 검출된 얼굴랜드마크를 실시간영상에 점으로 표시합니다.
* ESC 키를 누르면 프로그램이 종료되도록 설정합니다.
* Mediapipe의 solutions.face_mesh를 사용하여 얼굴 랜드마크검출기를 생성할 수 있습니다.
* 검출된 랜드마크좌표를 이용하여 OpenCV의 circle 함수를 사용해 각 랜드마크를 시각화할 수 있습니다
* 랜드마크 좌표는 정규화되어있으므로, 이미지크기에 맞게 변환이 필요합니다


### 전체코드
```python
import cv2
import numpy as np
import glob


```

### 주요코드
- **`코드`**:  
  

-  **`코드`**:  
  

-  **`코드`**:  

- **`코드`**:  

-  **`코드`**:  

### 실행결과

