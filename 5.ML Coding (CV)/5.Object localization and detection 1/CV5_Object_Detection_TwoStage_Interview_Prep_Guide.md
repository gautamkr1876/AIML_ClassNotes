<a id="top"></a>
# CV Notebook 5 — Object Detection: Two-Stage Methods (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §5](../CV_Revision_Guide.md#5-module5).

## What this notebook actually demonstrates

The full **classification → localization → detection** progression on a gun-detection dataset (PistolData, 3,703 images):
1. **Single-object localization** with a ResNet-101 backbone and dual heads (classification + bbox regression).
2. **IoU and NMS** implementations from scratch.
3. **R-CNN family evolution:** R-CNN → Fast R-CNN → **Faster R-CNN** (RPN replaces Selective Search).
4. **Anchor boxes**, RPN, ROI pooling — the building blocks of modern two-stage detectors.

Final localization model achieves **99.7% classification accuracy** and **0.74 train / 0.45 test mean IoU** on bounding boxes.

## 🪜 Mental anchors for this notebook

- **Two-stage = "propose then classify."** Stage 1 (RPN): "where might there be an object?" Stage 2 (head): "what's in each candidate and what's its precise box?"
- **IoU is the universal overlap metric.** `intersection / union`, range `[0, 1]`. ≥0.5 = match.
- **NMS = deduplication.** Many overlapping boxes → keep the highest-confidence one, drop the rest.
- **Anchors = "first guesses."** Network learns offsets from predefined templates instead of absolute coordinates.

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset: PistolData_merged
- 3,703 images (416 × 416 RGB), gun / no-gun binary labels.
- Annotation: YOLO-style normalized `[class, x_center, y_center, width, height]`.
- Distribution: 2,704 gun / 999 no-gun (imbalanced).
- Train/test split: 70/30 (2,592 / 1,111).

### 2. Bounding box format conversion
```python
def cv_coords(bbox):
    """Convert normalized (xc, yc, w, h) → (x1, y1, x2, y2) in pixels."""
    xc, yc, w, h = bbox
    x1 = int((xc - w/2) * IMG_W); y1 = int((yc - h/2) * IMG_H)
    x2 = int((xc + w/2) * IMG_W); y2 = int((yc + h/2) * IMG_H)
    return x1, y1, x2, y2
```

### 3. IoU from scratch
```python
def compute_iou(label_box, pred_box):
    x1, y1, x2, y2     = cv_coords(label_box)
    px1, py1, px2, py2 = cv_coords(pred_box)
    inter_w = max(0, min(x2, px2) - max(x1, px1))
    inter_h = max(0, min(y2, py2) - max(y1, py1))
    inter   = inter_w * inter_h
    union   = (x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter
    return inter / (union + 1e-3)
```

### 4. NMS from scratch
```python
def non_max_suppress(confidences, xy_min, xy_max, iou_thresh=0.4):
    boxes = list(zip(confidences, xy_min, xy_max))
    boxes.sort(key=lambda b: b[0], reverse=True)
    keep = []
    for i, (c, mn, mx) in enumerate(boxes):
        if c == 0: continue
        keep.append(i)
        for j in range(i+1, len(boxes)):
            if compute_iou(boxes[i][1:], boxes[j][1:]) >= iou_thresh:
                boxes[j] = (0, *boxes[j][1:])    # suppress
    return keep
```

### 5. Dual-head model — classification + bbox
```python
from tf.keras.applications import ResNet101
from tf.keras.layers import Input, Dense, Flatten
from tf.keras.models import Model

backbone = ResNet101(weights='imagenet', include_top=False,
                      input_tensor=Input(shape=(416, 416, 3)))
backbone.trainable = False

x = Flatten()(backbone.output)

# Classification head: "is there a gun?"
cls_branch  = Dense(128, activation='relu')(x)
cls_branch  = Dense(64,  activation='relu')(cls_branch)
cls_branch  = Dense(32,  activation='relu')(cls_branch)
class_out   = Dense(1, activation='sigmoid', name='class_output')(cls_branch)

# Regression head: where's the bbox?
bbox_branch = Dense(128, activation='relu')(x)
bbox_branch = Dense(64,  activation='relu')(bbox_branch)
bbox_branch = Dense(32,  activation='relu')(bbox_branch)
box_out     = Dense(4, activation='sigmoid', name='box_output')(bbox_branch)

model = Model(backbone.input, [box_out, class_out])
model.compile(
    optimizer='adam',
    loss={'box_output': 'mse', 'class_output': 'binary_crossentropy'},
    loss_weights={'box_output': 4.0, 'class_output': 1.0},  # heavier on bbox
    metrics={'class_output': 'accuracy'},
)
```

### 6. RPN sketch (Faster R-CNN component)
```python
def rpn_layer(base_layers, num_anchors=9):
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(base_layers)
    # Per-anchor objectness (sigmoid) and bbox offsets (linear)
    objectness = layers.Conv2D(num_anchors,     1, activation='sigmoid')(x)
    bbox_regr  = layers.Conv2D(num_anchors * 4, 1, activation='linear')(x)
    return [objectness, bbox_regr, base_layers]
```

### 7. ROI Pooling (custom Keras layer)
```python
# RoiPoolingConv applies max-pooling per proposal region → fixed-size feature
# Output: (num_rois, pool_h, pool_w, channels)
```

## ⚙️ APIs introduced (specific to this notebook)

| Concept | Implementation |
|---|---|
| Bbox formats | YOLO normalized (xc, yc, w, h) ↔ pixel (x1, y1, x2, y2) |
| IoU | `intersection / union` |
| NMS | Sort by confidence, iteratively suppress overlaps > threshold |
| Anchor boxes | Predefined templates (9 typical: 3 scales × 3 ratios) |
| Dual-head model | One backbone, two task-specific heads |
| Loss weighting | `loss_weights={'box': 4.0, 'class': 1.0}` to balance scales |
| RPN | Learnable proposer: 1 sigmoid (objectness) + 4 regressors (bbox) per anchor |
| ROI Pooling | Resize variable proposals → fixed grid (typically 7×7) |
| Faster R-CNN training | 4 simultaneous losses: RPN cls + RPN bbox + Fast R-CNN cls + Fast R-CNN bbox |

### Production-grade torchvision API
| Call | Purpose |
|---|---|
| `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` | Full Faster R-CNN with FPN |
| `model([img_tensor])` | Returns list of dicts: `boxes`, `labels`, `scores` |
| `torchvision.ops.nms(boxes, scores, iou_threshold)` | NMS |
| `torchvision.ops.box_iou(b1, b2)` | Pairwise IoU matrix |
| `torchvision.ops.roi_align(features, boxes, output_size)` | ROI Align (better than ROI Pool for masks) |

## ⚠️ Notebook-specific gotchas

1. **Loss weighting matters when scales differ.** `MSE` on normalized bboxes is `~0.01`; `BCE` on class is `~0.5`. Without `loss_weights={'box': 4.0}`, the bbox loss is invisible.
2. **Bbox format confusion** is the #1 detection bug. YOLO `(xc, yc, w, h)` normalized vs Pascal VOC `(x1, y1, x2, y2)` pixel — verify before any IoU computation.
3. **Anchor boxes must match your data.** Default COCO anchors are tuned for COCO sizes — K-means cluster your training bbox shapes to derive task-specific anchors.
4. **NMS is per-class, not global.** A person riding a bike legitimately overlaps — don't suppress across classes.
5. **Plain IoU has zero gradient when boxes don't overlap.** Modern detectors use **GIoU / DIoU / CIoU** loss instead, which has a non-zero gradient even for disjoint boxes.
6. **`sigmoid` on bbox outputs** constrains coordinates to `[0, 1]` — only valid for normalized formats. For pixel coordinates use `linear` and clamp at inference.

## 🎯 Notebook quiz cells (verbatim)

**Q1.** Why have separate heads for classification vs bbox regression?
→ Different tasks (binary classification vs 4-D regression) → different optimal layer counts and activations. Sharing the backbone is enough.

**Q2.** What is IoU and why use it?
→ `area(intersection) / area(union)`. Scale-invariant overlap metric — better than per-coordinate MSE because it captures all error types (position, size) uniformly.

**Q3.** What does NMS do?
→ Deduplicates overlapping predictions by keeping the highest-confidence box and suppressing any other box with IoU ≥ threshold.

**Q4.** Why is ROI Pooling necessary?
→ Region proposals have variable sizes; the downstream classifier head needs fixed-size inputs. ROI Pool resizes them to e.g. 7×7 via region-aware max-pooling.

**Q5.** RPN vs Selective Search?
→ Selective Search is a hand-crafted, fixed offline algorithm. RPN is a learnable mini-CNN that proposes regions end-to-end with the rest of the model.

## 🪞 Extra ladder — IoU evolution

**Basic** — `IoU = intersection / union`. Zero gradient when boxes don't overlap.

**Intermediate** — **GIoU** (Generalized IoU) adds a term penalizing the smallest enclosing box: `GIoU = IoU − (C \ (A ∪ B)) / C`. Always has gradient.

**Advanced** — **CIoU** (Complete IoU) further penalizes center-distance and aspect-ratio mismatch. YOLOv4+ uses it. Empirically gives faster convergence and ~1 mAP improvement over plain IoU loss.

## What comes next

This notebook covered **two-stage** detection (slow but accurate). [Notebook 6 →](../6.Object%20localization%20and%20detection%202/) covers **single-stage** detection (YOLO, SSD, RetinaNet) — one forward pass, real-time, ~3× faster.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
