<a id="top"></a>
# CV Notebook 6 — Object Detection: Single-Stage Methods (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §6](../CV_Revision_Guide.md#6-module6).

## What this notebook actually demonstrates

**Real-time** object detection for autonomous driving — cars, pedestrians, traffic lights. The two-stage detectors from Notebook 5 are too slow (~100 ms/frame). This notebook introduces the **single-stage family**:

| Detector | mAP-50 (COCO) | Inference time | Use case |
|---|---|---|---|
| Faster R-CNN | ~70% | ~98 ms | Max accuracy, offline |
| **YOLOv3 @ 416** | ~58% | **~29 ms** | Real-time |
| YOLOv4 | ~66% | ~15 ms (GPU) | Real-time + accurate |
| YOLOv5s | ~56% | ~6 ms | Mobile / edge |
| RetinaNet | ~61% | ~98 ms | Focal-loss baseline |

The notebook demonstrates **YOLOv5 via OpenCV DNN** (load ONNX → blob → forward → NMS) on both still images and video streams.

## 🪜 Mental anchors for this notebook

- **One network, one forward pass, all detections.** No region-proposal step.
- **Grid cells own objects** — each cell predicts B anchor-based boxes; the cell containing the object centroid is "responsible."
- **Confidence = `P(object) × IoU(pred, truth)`.** Combines "is something here" with "how good is my box."
- **Focal Loss = "focus gradient on hard examples."** Down-weights easy negatives → solves class imbalance.

## 🧠 Key cell-by-cell walkthrough

### 1. Why single-stage?
Two-stage detectors (R-CNN family) have separate proposal + classification stages → multiple forward passes. For 30+ FPS video, this is too slow. Single-stage: one network outputs all bboxes in one shot.

### 2. YOLO output structure
For an `S × S` grid with `B` anchors and `C` classes:
- **Output tensor:** `S × S × B × (5 + C)`
  - `5 = (tx, ty, tw, th, objectness)`
  - `C = class probabilities`
- YOLOv3: three scales (13×13, 26×26, 52×52) for small/medium/large objects.

### 3. Loading YOLOv5 via OpenCV DNN (the portable way)
```python
import cv2

modelWeights = "yolov5s.onnx"
net = cv2.dnn.readNet(modelWeights)

with open("coco.names.txt", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')   # 80 COCO classes
```

### 4. Inference (forward pass)
```python
INPUT_WIDTH, INPUT_HEIGHT = 640, 640

def yolo_forward(img, net):
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1/255, size=(INPUT_WIDTH, INPUT_HEIGHT),
        mean=[0,0,0], swapRB=True, crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs       # YOLOv5: shape (1, 25200, 85) for 80 classes
```

### 5. Post-processing + NMS
```python
OBJECT_SCORE_THRESHOLD = 0.5
CLASS_CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45

def post_process(img, outputs):
    rows = outputs[0][0]                      # (25200, 85)
    boxes, scores, class_ids = [], [], []
    h, w = img.shape[:2]
    x_factor, y_factor = w / INPUT_WIDTH, h / INPUT_HEIGHT
    
    for row in rows:
        confidence = row[4]
        if confidence < OBJECT_SCORE_THRESHOLD: continue
        class_scores = row[5:]
        class_id = np.argmax(class_scores)
        if class_scores[class_id] < CLASS_CONFIDENCE_THRESHOLD: continue
        
        cx, cy, bw, bh = row[0:4]
        x = int((cx - bw/2) * x_factor)
        y = int((cy - bh/2) * y_factor)
        ww, hh = int(bw * x_factor), int(bh * y_factor)
        boxes.append([x, y, ww, hh])
        scores.append(float(confidence))
        class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(
        boxes, scores,
        CLASS_CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
    )
    return boxes, scores, class_ids, indices
```

### 6. Drawing detections
```python
def draw_predictions(img, boxes, scores, class_ids, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img
```

### 7. Video inference loop
```python
stream = cv2.VideoCapture('road.mp4')
while True:
    ret, frame = stream.read()
    if not ret: break
    outputs = yolo_forward(frame, net)
    annotated = draw_predictions(frame, *post_process(frame, outputs))
    cv2.imshow('detection', annotated)
    if cv2.waitKey(1) == ord('q'): break
stream.release(); cv2.destroyAllWindows()
```

### 8. The Ultralytics shortcut (modern way)
```python
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
results = model('road.jpg')         # or video, webcam, batch directory
results[0].show()                   # auto-annotated
results[0].save('out.jpg')
```

## ⚙️ APIs introduced (specific to this notebook)

### OpenCV DNN (portable, no PyTorch dependency)
| Call | Purpose |
|---|---|
| `cv2.dnn.readNet(weights, config?)` | Load ONNX / Caffe / Darknet weights |
| `cv2.dnn.blobFromImage(img, scale, size, mean, swapRB, crop)` | Preprocess image → DNN input blob |
| `net.setInput(blob)` / `net.forward(layer_names)` | Run inference |
| `net.getUnconnectedOutLayersNames()` | Output layer names (auto-detected) |
| `cv2.dnn.NMSBoxes(boxes, scores, conf_t, nms_t)` | Non-Max Suppression |

### Ultralytics (modern PyTorch-based)
| Call | Purpose |
|---|---|
| `from ultralytics import YOLO; m = YOLO('yolov5s.pt')` | Load pretrained |
| `m('image.jpg')` | Inference (image, video, dir, webcam) |
| `m.train(data='data.yaml', epochs=N)` | Fine-tune on custom dataset |
| `m.export(format='onnx')` | Export to ONNX / CoreML / TFLite |

## ⚠️ Notebook-specific gotchas

1. **`blobFromImage` parameters matter.** `swapRB=True` because OpenCV is BGR, YOLO trained on RGB. Forget it → all colors wrong → no detections.
2. **`scalefactor=1/255`** for YOLOv5; some other models need different preprocessing.
3. **YOLOv5 output is one big tensor** `(1, 25200, 85)` — 25200 anchor predictions across 3 scales × 3 anchors × multiple grid sizes. Filter by objectness × class-confidence before NMS.
4. **`cv2.dnn.NMSBoxes` expects `(x, y, w, h)`** corner-format. Don't pass `(x1, y1, x2, y2)`.
5. **Per-class NMS is best** for crowded scenes. If you NMS globally, a car next to a truck might suppress the truck.
6. **YOLOv5 has 5 variants** — `n` (nano), `s` (small), `m`, `l`, `x` (extra-large). Speed vs accuracy: `n` is ~10× faster but ~10 mAP lower than `x`. Pick based on deployment target.
7. **CocoNames file ordering matters.** YOLO predicts class indices; you map them to names. Wrong file → all labels off by one.
8. **`net.forward()` returns a list of arrays** (one per output layer). For YOLOv5 ONNX there's typically one; for YOLOv3 there are three (one per scale).

## 🎯 Notebook-relevant questions

**Q1.** Why is YOLO faster than Faster R-CNN?
→ One forward pass over the entire image (dense prediction) vs RPN + classifier (two stages, two networks).

**Q2.** What does "objectness" mean in YOLO?
→ `P(object) × IoU(pred, truth)`. Combines "is there an object here?" with "how good is my bounding box?"

**Q3.** What is Focal Loss and what problem does it solve?
→ `−α(1−p)^γ log(p)`. Down-weights easy examples; addresses the extreme class imbalance in single-stage detectors (10⁴–10⁵ background locations per image).

**Q4.** Why does single-stage struggle more with class imbalance than two-stage?
→ Two-stage filters out most background with the RPN before the classifier sees anything. Single-stage classifies every anchor on every grid cell → millions of background examples per image.

**Q5.** Multi-scale detection — why use multiple feature map sizes?
→ Small objects activate shallow (high-res) layers; large objects activate deep (low-res) layers. Predict at multiple scales → detect everything.

## 🪞 Extra ladder — anchor-free vs anchor-based

**Basic** — anchor-based (YOLOv3, SSD): predefined templates per grid cell.

**Intermediate** — anchor-free (CenterNet, FCOS): predict center heatmap + bbox size directly. Fewer hyperparameters, simpler implementation.

**Advanced** — **DETR** (Detection Transformer, 2020): no anchors, no NMS. Uses a transformer + bipartite matching for end-to-end set prediction. Slower training but cleaner pipeline. **Deformable DETR** (2021) made it competitive on speed.

## What comes next

Single-stage detection gave bboxes. [Notebook 7 →](../7.Object%20segmentation/) goes finer: **per-pixel** classification (segmentation) — U-Net, FCN, Mask R-CNN.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
