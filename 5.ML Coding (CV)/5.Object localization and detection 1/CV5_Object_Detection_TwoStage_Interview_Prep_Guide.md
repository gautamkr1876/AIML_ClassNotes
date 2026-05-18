<a id="top"></a>
# CV Notebook 5 — Object Detection: Two-Stage Methods (Deep Dive)

> Per-notebook companion to the master guide. For the full cross-notebook cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §5](../CV_Revision_Guide.md#5-module5). **This deep-dive is standalone** — every concept it introduces is explained end-to-end here. You should never have to leave this file to understand a term.

## What this notebook actually demonstrates

The full **classification → localization → detection** progression on a gun-detection dataset (PistolData, 3,703 images):
1. **Single-object localization** with a ResNet-101 backbone and dual heads (classification + bbox regression).
2. **IoU and NMS** implementations from scratch.
3. **R-CNN family evolution:** R-CNN → Fast R-CNN → **Faster R-CNN** (RPN replaces Selective Search).
4. **Anchor boxes**, RPN, ROI pooling — the building blocks of modern two-stage detectors.

Final localization model achieves **99.7% classification accuracy** and **0.74 train / 0.45 test mean IoU** on bounding boxes.

## 🪜 Mental anchors for this notebook

- **Two-stage = "propose then classify"** — Stage 1 (RPN): "where might there be an object?" Stage 2 (head): "what's in each candidate and what's its precise box?" Slower than single-stage, but historically more accurate.
- **IoU is the universal overlap metric** — `intersection / union`, range `[0, 1]`. ≥ 0.5 = the canonical "match" threshold (PASCAL VOC). Drives every downstream metric (NMS, mAP, anchor-matching).
- **NMS = deduplication** — many overlapping boxes for the same object → sort by confidence, keep the best, drop anything that overlaps it too much, repeat. Run **per class** so a car and a pedestrian close together aren't merged.
- **Anchors = "first guesses"** — instead of predicting absolute bbox coordinates (hard, high-variance), the network predicts *small offsets* from a handful of predefined rectangle templates.
- **One backbone, two heads** — share the expensive CNN; let each task have its own cheap final layers. The notebook's dual-head model is the simplest example of this pattern.

## 📖 Concept walkthroughs

> Beginner-first introduction of every concept this notebook touches. Each entry is a full Concept Definition Template — mental model → what / why / how / where / related → code → gotcha — substituted with this notebook's actual shapes, numbers, and code.

### Object detection vs classification vs localization

> **🪜 Mental model:** *Find them all, draw boxes, label each.* Classification = one label for the image; localization = one label + one box; detection = many labels + many boxes.

**What it is.** Three related but increasingly hard CV tasks, distinguished by *how many* objects and *what info* you predict per image:
- **Classification** — "is there a gun in this image?" One label per image (in this notebook: gun / no-gun).
- **Localization** — "where is the (single) gun?" One label + one bounding box. This is what the notebook's dual-head ResNet-101 model actually does.
- **Object detection** — "where are *all* the guns and *all* other objects?" Many labels + many bboxes per image. The image can contain 0, 1, or many objects, and the model must figure out how many. Output: a list of `(class_id, confidence, x1, y1, x2, y2)` rows.

**Why it matters.** Almost every applied CV system beyond pure classification needs detection: self-driving cars find pedestrians and other vehicles, retail uses it for shelf monitoring, security uses it for intrusion alerts. The notebook *builds up* the progression — it implements localization but sketches the components (RPN, RoI Pooling) needed for full detection. Confusing the three tasks in an interview signals you haven't built one of them; they have different output structures, different losses, and very different evaluation metrics.

**How they differ.**
1. **Classification:** input = image; output = single class label (or softmax over classes). Loss = cross-entropy.
2. **Localization:** input = image; output = one class label + one bbox. Loss = cross-entropy for class + MSE / smooth-L1 for bbox. **The notebook's setup.**
3. **Detection:** input = image; output = a *variable-length list* of (class, bbox, score) tuples. Loss = sum of classification + bbox losses across all matched predictions, plus an "objectness" loss. Needs NMS to deduplicate.

The notebook's dual-head model predicts exactly one bbox per image, so it's **localization, not detection** — extending it to detection requires either an RPN (Faster R-CNN — sketched in cell 6) or a grid + anchor formulation (YOLO — Notebook 6).

**Where it's used.**
- Classification: ImageNet challenges, image-quality filtering, content moderation.
- Localization: medical imaging (single tumour), document layout (single table), the notebook's pistol-detection model.
- Detection: autonomous driving, retail shelf analytics, security cameras, drone monitoring.
- In this notebook: **cell 5** (dual-head localization model); detection is sketched but not trained.

**Related terms.**
- **Bounding box (bbox)** — the rectangle around an object (next entry).
- **Segmentation** (Module 7) — denser still: per-pixel labels.
- **mAP** — the detection metric; meaningless for single-image classification.
- **Instance segmentation** — detection + per-instance mask; what Mask R-CNN does.
- **Multi-task learning** — the dual-head pattern: one backbone, multiple task-specific heads.

```python
# Localization head — this notebook's setup
class_out = Dense(1, activation='sigmoid', name='class_output')(shared)   # is there a gun?
box_out   = Dense(4, activation='sigmoid', name='box_output')(shared)     # where is it?
# Detection would replace these with per-anchor predictions across a spatial grid (Notebook 6)
```

**Gotcha.** Don't conflate *localization* (one object per image) with *detection* (many objects per image). They use different loss shapes, different evaluation metrics, and different architectures. The notebook's "99.7% accuracy + 0.45 test IoU" is **localization** scoring — *not* mAP.

### Bounding box format — `(x, y, w, h)` vs `(x1, y1, x2, y2)`

> **🪜 Mental model:** *Two ways to draw the same rectangle.* Either give the **top-left + bottom-right corners** (`x1, y1, x2, y2`) or the **centre + size** (`xc, yc, w, h`). Same box, completely different numbers — mixing them silently produces nonsense.

**What it is.** A **bounding box** is the axis-aligned rectangle that encloses an object, described by four numbers. The two dominant conventions are:
- **Corner format `(x1, y1, x2, y2)`** — coordinates of the top-left and bottom-right corners. Used by PASCAL VOC, torchvision.
- **Centre format `(xc, yc, w, h)`** — centre point + width + height. Used by YOLO. Often **normalised** (each value divided by image width/height so all four numbers are in `[0, 1]` regardless of image size).

This notebook's annotations are in **YOLO normalised centre format**: `[class_id, xc, yc, w, h]`. The notebook converts to corner-pixel format via `cv_coords(bbox)` for OpenCV drawing and IoU computation.

**Why it matters.** Bbox-format bugs are the #1 silent failure in detection codebases. An IoU function written for `(x1, y1, x2, y2)` fed YOLO `(xc, yc, w, h)` will happily return numbers — they just won't mean anything (you'll be computing overlap between the wrong four points). Detection papers, datasets, and frameworks all assume *their* format; converting at every boundary is the only safe practice. Beginners lose hours to this; senior engineers verify format on the first print statement.

**How it works (conversions).**
- **Centre → Corner:** `x1 = xc − w/2; y1 = yc − h/2; x2 = xc + w/2; y2 = yc + h/2`.
- **Corner → Centre:** `xc = (x1 + x2)/2; yc = (y1 + y2)/2; w = x2 − x1; h = y2 − y1`.
- **Normalised → Pixel:** multiply x-coords by image width, y-coords by image height.
- **Pixel → Normalised:** divide by image width / height.

All four numbers describe the same rectangle in every format — you can round-trip without information loss.

**Where it's used.**
- Annotation tools: LabelImg saves PASCAL VOC corner-pixel; Roboflow can export any format.
- Model outputs: YOLO emits normalised centre; torchvision emits pixel corner.
- Every IoU / NMS / loss function (must match its expected format).
- Visualisation: OpenCV's `cv2.rectangle` wants pixel corner; matplotlib's `Rectangle` wants pixel `(x, y, w, h)`.
- This notebook: **cell 2** (`cv_coords` converts YOLO-normalised → pixel-corner before drawing or computing IoU).

**Related terms.**
- **IoU** — silently broken if box formats differ.
- **NMS** — same.
- **Anchor box** — defined in a specific format; offsets must match.
- **PASCAL VOC vs COCO format** — VOC uses pixel corner; **COCO uses pixel `(x, y, w, h)`** (not normalised, not corner — a third gotcha-prone variant).
- **Normalisation** — dividing all coords by image dimensions to be resolution-independent.

```python
# This notebook's conversion helper — normalised centre → pixel corner
def cv_coords(bbox):                  # bbox = (xc, yc, w, h), normalised [0, 1]
    xc, yc, w, h = bbox
    x1 = int((xc - w/2) * IMG_W); y1 = int((yc - h/2) * IMG_H)
    x2 = int((xc + w/2) * IMG_W); y2 = int((yc + h/2) * IMG_H)
    return x1, y1, x2, y2
```

**Gotcha.** **Always print one sample box** before any IoU / NMS / training step. The numbers tell you instantly which format you're in: all in `[0, 1]` → normalised; large values → pixel; `(x2, y2)` larger than `(x1, y1)` → corner; otherwise → centre.

### IoU (Intersection over Union)

> **🪜 Mental model:** *Overlap quality.* If two boxes mostly cover the same area, IoU is near 1; if they barely touch, IoU is near 0.

**What it is.** **IoU** (Intersection over Union, also called the **Jaccard index** for sets) is the standard scalar measure of how well two boxes overlap. Formula:

```
IoU = area(A ∩ B) / area(A ∪ B)
```

In words: *the area of the region where the two boxes overlap, divided by the total area covered by either box.* `A ∩ B` is the **intersection** (the rectangular overlap region — could be empty); `A ∪ B` is the **union** (the total area covered by either box, computed as `area(A) + area(B) − area(A ∩ B)` to avoid double-counting). Range `[0, 1]`: 0 = no overlap, 1 = identical boxes. The canonical "match" threshold from PASCAL VOC is **IoU ≥ 0.5**.

**Why it matters.** IoU is *the* atomic metric of detection — it powers NMS (deduplication), mAP (the evaluation metric), and anchor-matching during training (an anchor is assigned to a ground-truth box if their IoU is high enough). It's also scale-invariant — it doesn't care whether your boxes are 5×5 or 500×500 pixels. Computing IoU correctly is a near-guaranteed coding question in any CV detection round. The notebook implements it from scratch in **cell 3** for exactly this reason.

**How it works.**
1. Compute intersection box edges: `inter_x1 = max(a_x1, b_x1)`, `inter_y1 = max(a_y1, b_y1)`, `inter_x2 = min(a_x2, b_x2)`, `inter_y2 = min(a_y2, b_y2)`. (The overlap region's left edge is the rightmost of the two left edges, etc.)
2. Compute intersection area, clamping negative widths/heights to zero: `inter = max(0, inter_x2 − inter_x1) * max(0, inter_y2 − inter_y1)`. The `max(0, ...)` handles boxes that don't overlap at all.
3. Compute each box's area: `area_a = (a_x2 − a_x1) * (a_y2 − a_y1)`, same for `b`.
4. Compute union via inclusion-exclusion: `union = area_a + area_b − inter`.
5. Divide: `IoU = inter / (union + ε)`, where `ε` (e.g. `1e-3`) prevents division by zero for degenerate boxes.

**Where it's used.**
- This notebook in **cell 3** (`compute_iou` for a single pair) and inside `non_max_suppress` (cell 4).
- NMS: drop boxes with IoU > 0.5 to the kept one.
- mAP: a prediction matches a ground-truth box if IoU ≥ threshold (0.5, or 0.5:0.95 averaged).
- Anchor matching during training (an anchor is a "positive" for a ground-truth box if IoU ≥ 0.7).
- Loss functions: GIoU/DIoU/CIoU loss are differentiable variants used during training.

**Related terms.**
- **GIoU / DIoU / CIoU** — modern loss variants — plain IoU has zero gradient when boxes don't overlap.
- **mAP** — uses IoU as the matching criterion.
- **Jaccard index** — synonym from set theory.
- **NMS** — uses IoU to decide which boxes to suppress.
- **Anchor matching** — uses IoU to assign anchors to ground-truth boxes during training.

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

**Vectorised batch IoU** (used inside fast NMS):
```python
def iou_batch(box, others):                       # box: (4,), others: (N, 4) corner format
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area  = (box[2]-box[0])*(box[3]-box[1])
    areas = (others[:, 2]-others[:, 0])*(others[:, 3]-others[:, 1])
    return inter / (area + areas - inter + 1e-8)
```

**Gotcha.** Plain IoU is a fine *metric* but a poor *loss* — when two boxes don't overlap at all, IoU = 0 *and* its gradient is 0. The optimiser gets no signal. Modern detectors (YOLOv4+) use **CIoU loss** instead — always has a non-zero gradient, encodes centre-distance and aspect-ratio mismatch on top of overlap.

### NMS (Non-Maximum Suppression)

> **🪜 Mental model:** *Survivor of overlaps.* Sort boxes by confidence, keep the best one, throw out any others that overlap it too much, repeat.

**What it is.** **NMS** is a post-processing step that deduplicates a list of overlapping detections so each object gets a single bounding box in the output. Modern detectors typically fire 10–100 high-confidence boxes for one object (a YOLO grid emits dozens per car); without NMS the output is a forest of redundant boxes. NMS is the **last step** of every detector's inference pipeline (two-stage and single-stage alike).

The algorithm in plain English:
1. **Sort** all candidate boxes by confidence (highest first).
2. **Take** the highest-confidence box; add it to the "keep" list.
3. **Compute IoU** between the kept box and every remaining box.
4. **Drop** (suppress) any remaining box whose IoU with the kept box exceeds a threshold (typically 0.4–0.5).
5. **Repeat** from step 2 with the (now-smaller) remaining set, until empty.

**Why it matters.** Without NMS your output has duplicate boxes for every object, mAP tanks, and downstream consumers (trackers, alarms, UIs) go crazy. Every detection interview round asks you to write NMS pseudocode — it's a near-guaranteed coding question. The notebook implements NMS from scratch in **cell 4** for exactly this reason. Also: NMS must run **per class** — a car and a pedestrian standing close together legitimately overlap (cross-class), and a single global NMS would incorrectly merge them.

**How it works (mechanics).** Greedy iteration. Per class. Implementations (`torchvision.ops.nms`, `cv2.dnn.NMSBoxes`) are vectorised over boxes — they run on GPU even for 10K+ candidates and finish in milliseconds. The notebook's implementation is `O(N²)` (acceptable for ~100 boxes); production code uses the vectorised version.

The "suppress" step is just zeroing the confidence (or removing from a set) — there's no special data structure required.

**Where it's used.**
- This notebook in **cell 4** (`non_max_suppress` from scratch).
- The end of every detector's inference pipeline (Faster R-CNN, YOLO, SSD, RetinaNet).
- Inside the RPN (when generating proposals — keeps the top-`N` after NMS).
- `torchvision.ops.nms(boxes, scores, iou_threshold)` is the production call.
- `cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)` is OpenCV's variant.

**Related terms.**
- **IoU** — the metric NMS uses to decide overlap (previous entry).
- **Soft-NMS** — decays overlapping confidences instead of zeroing them; helps when objects legitimately overlap (dense crowds).
- **DIoU-NMS** — uses DIoU instead of IoU; better for elongated objects.
- **DETR's bipartite matching** — modern transformer-based detectors skip NMS entirely.
- **Per-class NMS** — the default in practice; run NMS separately per class so cross-class overlaps survive.

```python
def non_max_suppress(confidences, xy_min, xy_max, iou_thresh=0.4):
    boxes = list(zip(confidences, xy_min, xy_max))
    boxes.sort(key=lambda b: b[0], reverse=True)        # step 1
    keep = []
    for i, (c, mn, mx) in enumerate(boxes):
        if c == 0: continue                              # already suppressed
        keep.append(i)                                   # step 2
        for j in range(i+1, len(boxes)):                 # step 3
            if compute_iou(boxes[i][1:], boxes[j][1:]) >= iou_thresh:
                boxes[j] = (0, *boxes[j][1:])            # step 4: suppress
    return keep
```

**Gotcha.** **Run NMS per-class, not globally.** A car and a pedestrian standing close together can legitimately overlap and shouldn't be merged. Also: the notebook's `O(N²)` implementation is fine for educational code; swap to `torchvision.ops.nms` (CUDA, vectorised) in production where you may have 10K+ candidates.

### R-CNN lineage (R-CNN → Fast R-CNN → Faster R-CNN)

> **🪜 Mental model:** *Three generations, each fixing the previous one's slowest part.* R-CNN was correct but glacial; Fast R-CNN shared the CNN; Faster R-CNN replaced the proposal step with a learned mini-network.

**What it is.** The R-CNN family is the canonical two-stage detection lineage from 2014–2016. Each version inherits the "propose then classify" structure but fixes the bottleneck of the previous one:
- **R-CNN (Girshick 2014)** — Run **Selective Search** (a hand-crafted, CPU-only algorithm that groups pixels by colour/texture) → produces ~2 000 region proposals → push *each one* through a CNN separately → SVM classifier on top. Inference: **~47 s per image.** Painful.
- **Fast R-CNN (Girshick 2015)** — Run the CNN **once** on the whole image to get a shared feature map → use **RoI Pooling** to extract a per-proposal feature crop from the shared map → single multi-task head (classifier + bbox regressor). Inference: **~2 s per image**, ~10× speedup. Still uses Selective Search for proposals.
- **Faster R-CNN (Ren et al., 2016)** — Replace Selective Search with a learned **Region Proposal Network (RPN)** that shares features with the detector → fully end-to-end trainable. Inference: **~0.2 s per image (~5 fps).** Finally practical.

**Why it matters.** Faster R-CNN is the canonical two-stage detector still in heavy use today (and the foundation of Mask R-CNN, Cascade R-CNN, Panoptic FPN). Every CV interview probes the lineage — *"explain how R-CNN evolved to Faster R-CNN"* is a standard FAANG question. Knowing the bottleneck that each generation fixed is the senior-engineer answer; just listing the names is the junior one.

**How they differ (the bottleneck-fix progression).**

| Method | Proposals from | CNN runs | Speed | mAP (PASCAL VOC) | Trainable end-to-end? |
|---|---|---|---|---|---|
| **R-CNN** | Selective Search | once per proposal | ~47 s | 66.0 | No |
| **Fast R-CNN** | Selective Search | once per image (shared) | ~2 s | 70.0 | Partially |
| **Faster R-CNN** | RPN (learned) | once per image (shared) | ~0.2 s | 73.2 | Yes |

- R-CNN's bottleneck: thousands of CNN forward passes per image. *Fix: share the CNN (Fast R-CNN).*
- Fast R-CNN's bottleneck: Selective Search runs on CPU and isn't learnable. *Fix: learn proposals on GPU (Faster R-CNN).*
- Faster R-CNN's bottleneck (today): two-stage is still slower than single-stage. *Fix: drop the proposal stage entirely (YOLO — Notebook 6).*

**Where it's used.**
- `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` is the modern one-liner.
- Mask R-CNN (Module 7) extends Faster R-CNN with a mask head.
- Cascade R-CNN: stacks multiple Faster R-CNNs with increasing IoU thresholds for higher precision.
- This notebook **sketches** the Faster R-CNN components (cells 6, 7) but doesn't train end-to-end; the dual-head model in cell 5 is single-object localization, not full Faster R-CNN.

**Related terms.**
- **RPN (Region Proposal Network)** — Faster R-CNN's proposal stage (next entry).
- **RoI Pooling / RoI Align** — variable-size proposal → fixed-size feature.
- **Selective Search** — the hand-crafted predecessor that R-CNN/Fast R-CNN used.
- **Anchor box** — the reference shapes the RPN predicts offsets from.

```python
# The production one-liner — full Faster R-CNN with FPN
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
preds = model([img_tensor])           # [{'boxes': ..., 'labels': ..., 'scores': ...}]
```

**Gotcha.** "Fast R-CNN" and "Faster R-CNN" sound nearly identical but refer to different architectures. **Get the names right in an interview** — confusing them signals you read a blog post but not the papers. Fast R-CNN still uses Selective Search; Faster R-CNN replaced it with the RPN.

### RPN (Region Proposal Network)

> **🪜 Mental model:** *Mini-CNN that learns where to look.* A tiny network on top of the backbone outputs "objectness score + bbox refinement" at every position.

**What it is.** The **Region Proposal Network** is a small CNN inside Faster R-CNN that learns to generate region proposals end-to-end. It takes the shared feature map of the backbone (e.g., ResNet-50's conv5 output) as input, slides a **3×3 conv** across it for context, then attaches **two parallel 1×1 conv heads**: one predicting **`k` objectness scores** ("is there an object?") and one predicting **`k × 4` bbox refinement offsets** (`(dx, dy, dw, dh)` per anchor) per spatial position. `k = 9` is standard (3 scales × 3 aspect ratios). The proposals with highest objectness, post-NMS, are the candidates fed to the classifier head.

**Why it matters.** Replacing Selective Search (CPU-only, ~2 s per image, non-trainable) with the RPN (GPU, ~10 ms per image, end-to-end trainable) was the breakthrough that made Faster R-CNN possible. The RPN also lets the network *learn* what counts as an "object proposal" — Selective Search just groups pixels by colour, the RPN learns task-specific cues like edges and texture. Today it's a default component of every two-stage detector.

**How it works.**
1. Backbone produces a feature map of shape `(H', W', C)` — e.g. for a 416×416 input through ResNet-50's conv5, that's `(13, 13, 2048)`.
2. RPN slides a 3×3 conv with `512` channels across this feature map → `(13, 13, 512)`. This is the "context" conv.
3. Two parallel 1×1 convs branch off:
   - `objectness = Conv2D(num_anchors, 1, sigmoid)` → `(13, 13, 9)`: per-position, per-anchor "is object?" score.
   - `bbox_regr  = Conv2D(num_anchors * 4, 1, linear)` → `(13, 13, 36)`: per-position, per-anchor `(dx, dy, dw, dh)` offsets.
4. Decode each anchor: starting box `(anchor_x, anchor_y, anchor_w, anchor_h)` + offsets `(dx, dy, dw, dh)` → candidate proposal box.
5. Filter by objectness score (keep top-N, e.g. top 2 000).
6. NMS the candidates → ~300 final proposals.
7. Pass the 300 proposals to RoI Pool / RoI Align → second-stage classifier.

**Where it's used.** Inside every R-CNN-family detector (Faster R-CNN, Mask R-CNN, Cascade R-CNN). Internal — you rarely interact with it directly unless modifying architecture. This notebook implements just the RPN's conv layers in **cell 6** for educational purposes, without training.

**Related terms.**
- **Anchor box** — the reference shapes RPN predicts offsets for (next entry).
- **Selective Search** — the predecessor it replaced.
- **Objectness score** — RPN's "is there anything here?" output.
- **Region proposal** — the output of RPN, fed to stage 2.
- **Two-stage detection** — the family RPN belongs to.

```python
def rpn_layer(base_layers, num_anchors=9):
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(base_layers)
    objectness = layers.Conv2D(num_anchors,     1, activation='sigmoid')(x)   # "is object?" per anchor
    bbox_regr  = layers.Conv2D(num_anchors * 4, 1, activation='linear')(x)    # (dx, dy, dw, dh) per anchor
    return [objectness, bbox_regr, base_layers]
```

**Gotcha.** RPN's "network" is really just **three conv layers** glued on top of the backbone — the "network" in the name is tiny. The heavy lifting is the shared backbone. Beginners imagine RPN as a big subnetwork; the truth is "3×3 conv + two 1×1 convs."

### Anchor boxes

> **🪜 Mental model:** *Pre-shaped picture frames.* Instead of guessing a box from scratch, the network nudges one of a few standard rectangles into place.

**What it is.** **Anchor boxes** (also called "anchors" or "priors") are predefined rectangle shapes assigned to each spatial position of the prediction grid. Instead of having the network output absolute bbox coordinates from scratch (a high-variance regression problem with infinite possible answers), each position offers `k` anchor "templates," and the network outputs **small offsets and scale factors** to nudge the chosen anchor into place. For Faster R-CNN's RPN, `k = 9` — three scales (128², 256², 512² pixels) × three aspect ratios (1:1, 1:2, 2:1). For YOLOv3+, anchors are clustered from training data via K-means.

**Why it matters.** Anchors transform "predict an absolute bounding box" (hard) into "refine a starting guess" (easy). They also give the network a way to encode *aspect ratios* explicitly — a tall anchor for pedestrians, a wide one for cars, a square one for faces. This is why two-stage and most single-stage detectors learn faster and reach higher accuracy than anchor-free predecessors. For this notebook's dataset (long thin pistols), default COCO-derived anchors are suboptimal — K-means on training-set widths/heights would derive task-specific anchors. The notebook uses defaults; tuning anchors to data is a standard interview question.

**How they work.**
1. Pick `k` anchors per location, covering varied scales and aspect ratios. (Faster R-CNN: 9 hand-set. YOLOv3+: 9 K-means clustered.)
2. During training, each ground-truth box is matched to the anchor with highest IoU (typically requires IoU ≥ 0.7 to be "positive"); that anchor becomes "responsible" for predicting it.
3. The network outputs per anchor: `(tx, ty, tw, th)` offsets + objectness + class scores.
4. **Decode at inference:** `bx = anchor_x + tx · anchor_w`; `by = anchor_y + ty · anchor_h`; `bw = anchor_w · exp(tw)`; `bh = anchor_h · exp(th)`. The exponential lets the network shrink *or* grow the anchor symmetrically in log space.
5. Filter by objectness, NMS, return final boxes.

**Where they're used.**
- Faster R-CNN's RPN — this notebook's **cell 6** uses 9 default anchors per location.
- YOLOv2–v5, SSD, RetinaNet — every modern anchor-based detector.
- Modern **anchor-free** detectors (CenterNet, FCOS, YOLOX, DETR) skip them and predict bboxes directly — but they're newer and not universal yet.

**Related terms.**
- **Anchor-free detection** — predict bbox directly; removes anchor hyperparameters but typically slower to converge.
- **K-means anchor selection** — cluster training bbox sizes to derive task-specific anchors (next module's notebook uses this).
- **IoU matching** — how anchors get assigned to ground truths during training.
- **Default boxes** — SSD's term for the same idea.
- **Priors** — alternative name (RetinaNet paper uses this).

```python
# Per-cell anchors: 3 scales × 3 aspect ratios = 9 (Faster R-CNN default)
scales = [128, 256, 512]
ratios = [(1, 1), (1, 2), (2, 1)]
anchors = [(s * w, s * h) for s in scales for (w, h) in ratios]   # 9 (w, h) pairs
```

**Gotcha.** **Default anchors (COCO-derived) may not fit your data.** For datasets with unusual aspect ratios (long thin pistols, tall pedestrians, tiny drones), K-means cluster your training bbox `(w, h)` pairs and use the resulting centres as anchors. The Ultralytics YOLO codebase does this automatically; older Faster R-CNN code does not — you have to do it by hand.

### mAP (mean Average Precision)

> **🪜 Mental model:** *The detector's report card.* Combines precision and recall across confidence thresholds and IoU thresholds into one number.

**What it is.** **mAP** (mean Average Precision) is the standard evaluation metric for object detection. It's a single scalar that aggregates how well the detector ranks its predictions and how well its boxes overlap the ground truth, across all classes. Computing it is multi-step but conceptually simple — every step builds on something you already know (IoU, precision, recall).

To compute **AP for one class:**
1. Sort all predictions of that class by confidence (highest first).
2. Walk down the sorted list. For each prediction, label it:
   - **TP (true positive)** if it matches some ground-truth box with IoU ≥ threshold AND that ground-truth hasn't already been "claimed" by a higher-confidence prediction.
   - **FP (false positive)** otherwise.
3. After each step, compute running **precision** = TP / (TP + FP) and **recall** = TP / (total GT for the class).
4. Plot the **precision-recall curve** (recall on x-axis, precision on y-axis).
5. **AP** = the area under this curve.

To compute **mAP**:
6. Average AP across all classes → **mAP**.

**Standard variants:**
- **mAP@0.5** (PASCAL VOC) — uses one IoU threshold of 0.5 to decide what counts as a match. Easier number, ~70–80 on COCO.
- **mAP@0.5:0.95** (COCO) — averages AP over IoU thresholds `[0.5, 0.55, 0.60, …, 0.95]` (10 values). Harder, more discriminating: ~35–55 on COCO.

**Why it matters.** mAP is *the* number everyone compares detectors on. Every detection paper reports it; every interview question on "how do you evaluate a detector" expects "mAP" as the first word. The two-threshold variants (VOC vs COCO) catch beginners — a model that scores 80 on VOC might score only 50 on COCO; **always state the threshold.**

**How it works at a glance.**
- **Precision** = TP / (TP + FP) = *"of the boxes I called positive, how many were right?"* Penalises false alarms.
- **Recall** = TP / (TP + FN) = *"of the ground-truth boxes I should have found, how many did I?"* Penalises misses.
- Sweeping the confidence threshold traces the precision-recall curve from high-confidence (high precision, low recall) to low-confidence (low precision, high recall).
- AP is the integral (area) of precision over recall — captures the whole tradeoff in one number.

**Where it's used.**
- The COCO leaderboard scoring formula.
- Every published detector's results table.
- Your own validation metric during training a real detector.
- **NOT in this notebook** — the notebook reports classification accuracy + mean IoU because it's doing single-object localization (one bbox per image). Real detection (Notebook 6's YOLO) requires mAP.

**Related terms.**
- **Precision / Recall** — components of AP.
- **AP (Average Precision)** — per-class number, averaged across to give mAP.
- **IoU threshold** — what counts as a match (0.5 loose, 0.75 tight, 0.5:0.95 averaged).
- **F1 score** — alternative; less common for detection (a single point on the PR curve).
- **PR curve** — what AP is the area under.

```python
# Standard library:
from torchmetrics.detection import MeanAveragePrecision
metric = MeanAveragePrecision(iou_type='bbox')
metric.update(preds, targets)         # both lists of dicts with 'boxes', 'scores', 'labels'
print(metric.compute())                # {'map': ..., 'map_50': ..., 'map_75': ..., ...}
```

**Gotcha.** **PASCAL VOC mAP@0.5 and COCO mAP@0.5:0.95 are different numbers** — a model scoring 80 on the first might score 50 on the second. Always state the threshold in interviews and papers. Also: the notebook's "mean IoU" (~0.45 on test) is *not* mAP — it's the average IoU across the localization predictions and isn't comparable to detector benchmarks.

### RoI Pool vs RoI Align

> **🪜 Mental model:** *Snap-to-grid vs interpolate.* Both turn a variable-size proposal region into a fixed-size feature patch (typically 7×7). RoI Pool **rounds** proposal coords to the nearest feature-map pixel; RoI Align uses **bilinear interpolation** at exact coords — no rounding, sharper features, required for pixel-accurate masks.

**What they are.** Two layers that solve the same problem — *"the classifier head needs a fixed-size input (e.g., 7×7×C), but proposals come in arbitrary sizes"* — by extracting a fixed-size feature crop from the shared backbone feature map for each proposal.

- **RoI Pool** (Fast R-CNN, 2015) — quantises proposal coordinates to integer grid cells on the feature map (rounds non-integer values to the nearest pixel), then max-pools each cell into one output value. Fast, but the rounding loses pixel-level precision.
- **RoI Align** (Mask R-CNN, 2017) — keeps proposal coords as floats and uses **bilinear interpolation** to sample the feature map at exact (non-integer) positions. No rounding. Each output cell averages 4 interpolated samples.

**Why it matters.** Without one of these, you couldn't share the backbone across proposals — you'd have to run the CNN separately per proposal, which is exactly R-CNN's slowness. RoI Align in particular boosted Mask R-CNN's segmentation mAP by ~3 points over plain RoI Pool. For classification-only Faster R-CNN, the difference is small; for pixel-precise tasks (masks, keypoints), it's essential.

**How they work.**
1. Map each proposal's coordinates from image space onto feature-map space (divide by the cumulative stride, e.g., 16 for ResNet-50's conv5 output — image is 16× smaller in feature space).
2. Divide the proposal region into `output_h × output_w = 7 × 7 = 49` bins.
3. **RoI Pool:** for each bin, snap edges to integer feature-map pixels, take max over those pixels. (`max` because pool, but `avg` is also used.)
4. **RoI Align:** for each bin, place 4 sample points at exact (float) positions, bilinearly interpolate the feature map at each, average them.
5. Output: a `(7, 7, C)` tensor per proposal; stack into a batch.

**Where they're used.**
- Inside Faster R-CNN (RoI Pool originally; modern versions use RoI Align).
- Inside Mask R-CNN, Cascade R-CNN, Panoptic FPN — always RoI Align.
- `torchvision.ops.roi_pool` and `torchvision.ops.roi_align` for direct use.
- This notebook implements a custom `RoiPoolingConv` Keras layer in **cell 7**.

**Related terms.**
- **Quantisation** — what RoI Pool does and RoI Align avoids.
- **Bilinear interpolation** — RoI Align's sampling trick.
- **Spatial Pyramid Pooling (SPP)** — earlier method from SPP-Net; RoI Pool is a single-scale special case.
- **Feature map stride** — the integer factor (e.g., 16, 32) by which the backbone downsamples; required to map proposal coords to feature-map coords.

```python
# torchvision built-in — preferred
import torchvision.ops as ops
crops = ops.roi_align(features, proposals, output_size=(7, 7), spatial_scale=1/16)
# crops.shape == (num_proposals, channels, 7, 7)
```

**Gotcha.** RoI Pool's quantisation is fine for classification but **breaks pixel-accurate masks** — that's *why* Mask R-CNN had to invent RoI Align. In an interview: if they ask *"why doesn't Mask R-CNN just use RoI Pool?"*, the answer is *"quantisation loses sub-pixel alignment, and masks need that."*

[🔝 Back to top](#top)

## 🧠 Cell-by-cell walkthrough

### 1. Dataset: PistolData_merged
- 3,703 images (416 × 416 RGB), gun / no-gun binary labels.
- Annotation: YOLO-style normalized `[class, x_center, y_center, width, height]`.
- Distribution: 2,704 gun / 999 no-gun (imbalanced).
- Train/test split: 70/30 (2,592 / 1,111).

### 2. Bounding box format conversion

Converts YOLO normalised centre format → pixel corner format so OpenCV can draw the box and IoU can compute over actual pixel areas. The `int(...)` cast rounds to pixel grid.

```python
def cv_coords(bbox):
    """Convert normalized (xc, yc, w, h) → (x1, y1, x2, y2) in pixels."""
    xc, yc, w, h = bbox
    x1 = int((xc - w/2) * IMG_W); y1 = int((yc - h/2) * IMG_H)
    x2 = int((xc + w/2) * IMG_W); y2 = int((yc + h/2) * IMG_H)
    return x1, y1, x2, y2
```

### 3. IoU from scratch

The `max(0, ...)` clamps zero-overlap cases to zero intersection; the `+ 1e-3` denominator avoids divide-by-zero on degenerate boxes.

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

Greedy loop — sort by confidence, keep the top, zero out anything overlapping it above threshold, repeat. The `c == 0` check skips already-suppressed boxes. `O(N²)` because of the nested loop; fine for educational code, replace with `torchvision.ops.nms` in production.

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

One backbone (ResNet-101, frozen ImageNet weights), two heads. `loss_weights={'box_output': 4.0}` makes bbox MSE (small numbers, ~0.01) and class BCE (~0.5) numerically comparable — without it the bbox gradient is invisible.

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

Demonstrates the RPN's structure: a 3×3 conv for context, then two parallel 1×1 conv heads — `k` objectness scores and `k × 4` bbox offsets per spatial position. Not connected to a full Faster R-CNN here; standalone for teaching.

```python
def rpn_layer(base_layers, num_anchors=9):
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(base_layers)
    # Per-anchor objectness (sigmoid) and bbox offsets (linear)
    objectness = layers.Conv2D(num_anchors,     1, activation='sigmoid')(x)
    bbox_regr  = layers.Conv2D(num_anchors * 4, 1, activation='linear')(x)
    return [objectness, bbox_regr, base_layers]
```

### 7. ROI Pooling (custom Keras layer)

Takes the shared feature map + a list of proposal regions; for each region, crops the feature map and max-pools to a fixed `(pool_h, pool_w)` grid. Output is a stack of fixed-size feature patches the classifier head can batch over.

```python
# RoiPoolingConv applies max-pooling per proposal region → fixed-size feature
# Output: (num_rois, pool_h, pool_w, channels)
```

[🔝 Back to top](#top)

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

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **Loss weighting matters when scales differ.** `MSE` on normalized bboxes is `~0.01`; `BCE` on class is `~0.5`. Without `loss_weights={'box': 4.0}`, the bbox loss is invisible to the optimiser.
2. **Bbox format confusion** is the #1 detection bug. YOLO `(xc, yc, w, h)` normalized vs PASCAL VOC `(x1, y1, x2, y2)` pixel — verify before any IoU computation by printing a sample box.
3. **Anchor boxes must match your data.** Default COCO anchors are tuned for COCO sizes — K-means cluster your training bbox shapes to derive task-specific anchors.
4. **NMS is per-class, not global.** A person riding a bike legitimately overlaps — don't suppress across classes.
5. **Plain IoU has zero gradient when boxes don't overlap.** Modern detectors use **GIoU / DIoU / CIoU** loss instead, which has a non-zero gradient even for disjoint boxes.
6. **`sigmoid` on bbox outputs** constrains coordinates to `[0, 1]` — only valid for normalized formats. For pixel coordinates use `linear` and clamp at inference.
7. **The notebook's NMS is `O(N²)`** — fine for a few hundred boxes, swap for `torchvision.ops.nms` in production where you may have 10k+ candidates.
8. **The notebook doesn't compute full mAP** — it reports accuracy + mean IoU because there's only one object per image. Don't confuse "mean IoU" (notebook metric) with "mAP" (the standard detection metric).
9. **RoI Pool ≠ RoI Align.** Pool quantises; Align interpolates. Use Align for any pixel-precise task (masks, keypoints).
10. **"Fast R-CNN" and "Faster R-CNN" are different architectures.** Confusing the names in an interview signals you read a blog, not the papers.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

**Q1.** What's the difference between classification, localization, and detection?
→ Classification = one label per image. Localization = one label + one bbox (the notebook's task). Detection = many labels + many bboxes per image, with NMS to deduplicate and mAP to evaluate. Different losses, different output shapes, different metrics — don't conflate. *(common FAANG CV question, original)*

**Q2.** Why have separate heads for classification vs bbox regression?
→ Different tasks (binary classification vs 4-D regression) need different optimal final-layer activations and counts. Sharing the backbone is enough — it's the expensive part and it learns visual features useful to both heads. This is the simplest example of **multi-task learning**. *(common FAANG multi-task learning question)*

**Q3.** What is IoU and why use it?
→ `area(intersection) / area(union)`. Scale-invariant overlap metric — better than per-coordinate MSE because it captures all error types (position, size, aspect-ratio mismatch) in a single bounded number in `[0, 1]`. Drives NMS, mAP, and anchor-matching. *(adapted from `andrewekhalel/MLQuestions`)*

**Q4.** What does NMS do?
→ Deduplicates overlapping predictions: sort by confidence, keep the highest, drop any box with IoU ≥ threshold to the kept one, repeat. Run **per-class** so legitimate cross-class overlaps (car next to pedestrian) survive. *(adapted from `andrewekhalel/MLQuestions`)*

**Q5.** Why is RoI Pooling necessary?
→ Region proposals come in variable sizes; the downstream classifier head needs fixed-size inputs to batch. RoI Pool resizes each proposal's feature crop to e.g. 7×7 via region-aware max-pooling, so all proposals become a uniform `(7, 7, C)` tensor. *(adapted from `alexeygrigorev/data-science-interviews`)*

**Q6.** RPN vs Selective Search?
→ Selective Search is a hand-crafted, CPU-only, ~2 s/image algorithm that groups pixels by colour/texture into ~2 000 proposals. RPN is a learnable 3×3-conv-plus-two-1×1-conv mini-network on top of the shared backbone, runs in ~10 ms on GPU, and is trained end-to-end with the rest of the detector. *(adapted from `andrewekhalel/MLQuestions`)*

**Q7.** What's the difference between RoI Pool and RoI Align — and when does it matter?
→ RoI Pool **quantises** proposal coords to integer feature-map pixels (rounding); RoI Align uses **bilinear interpolation** at exact float positions. The rounding error is harmless for classification but **breaks pixel-accurate masks** — that's why Mask R-CNN had to invent RoI Align (~3 mAP boost on segmentation). *(common FAANG detection-internals question, adapted from `chiphuyen/ml-interviews-book`)*

**Q8.** Why do modern detectors use GIoU / DIoU / CIoU loss instead of plain IoU loss?
→ Plain IoU is zero (and has zero gradient) when boxes don't overlap — the optimiser gets no signal to move them closer. GIoU adds a term penalising the enclosing box (always non-zero gradient); DIoU adds centre-distance; CIoU adds aspect-ratio mismatch. YOLOv4+ uses CIoU and gains ~1 mAP over plain IoU loss. *(common FAANG follow-up, adapted from `andrewekhalel/MLQuestions`)*

**Q9.** Walk me through the R-CNN → Fast R-CNN → Faster R-CNN evolution.
→ **R-CNN** (2014): Selective Search proposes ~2 000 regions, CNN runs **per region** → 47 s/image. **Fast R-CNN** (2015): Selective Search proposals + CNN runs **once** on the whole image, RoI Pool extracts per-proposal features → 2 s/image. **Faster R-CNN** (2016): Replace Selective Search with a learned RPN that shares features with the detector → 0.2 s/image, fully end-to-end. Each generation fixed the previous one's bottleneck. *(common FAANG architecture question)*

**Q10.** What is mAP and what's the difference between mAP@0.5 and mAP@0.5:0.95?
→ mAP = mean Average Precision across classes. For each class: sort predictions by confidence, walk down assigning TP/FP based on IoU ≥ threshold, plot precision-recall, integrate. Average across classes. **mAP@0.5** uses one IoU threshold (PASCAL VOC); **mAP@0.5:0.95** averages over 10 thresholds 0.5–0.95 (COCO, stricter). A model scoring 80 on the first might score 50 on the second. *(adapted from `chiphuyen/ml-interviews-book`)*

**Q11.** Faster R-CNN trains 4 losses simultaneously — what are they and why?
→ (1) RPN objectness (BCE on anchor-is-object), (2) RPN bbox regression (smooth L1 on anchor offsets), (3) detection-head classification (cross-entropy over classes), (4) detection-head bbox refinement (smooth L1). All four are summed with weights; multi-task learning shares the backbone and forces it to learn features useful for everything. *(common FAANG architecture question)*

**Q12.** If you were detecting long thin pistols (like in this notebook) and got poor IoU, what would you change about the anchor boxes?
→ Default 9 anchors (3 scales × 3 aspect ratios) are COCO-tuned and assume balanced aspect ratios. For long thin objects, K-means cluster the training-set `(width, height)` pairs and use the resulting cluster centres as anchors — gives the network templates that already match your object shape, so it only has to learn small refinements. *(original, design-judgment)*

[🔝 Back to top](#top)

## 🪞 Extra ladder — IoU evolution

**Basic** — `IoU = intersection / union`. Zero gradient when boxes don't overlap.

**Intermediate** — **GIoU** (Generalized IoU) adds a term penalizing the smallest enclosing box: `GIoU = IoU − (C \ (A ∪ B)) / C`. Always has gradient.

**Advanced** — **CIoU** (Complete IoU) further penalizes centre-distance and aspect-ratio mismatch. YOLOv4+ uses it. Empirically gives faster convergence and ~1 mAP improvement over plain IoU loss.

[🔝 Back to top](#top)

## What comes next

This notebook covered **two-stage** detection (slow but accurate). [Notebook 6 →](../6.Object%20localization%20and%20detection%202/) covers **single-stage** detection (YOLO, SSD, RetinaNet) — one forward pass, real-time, ~3× faster.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
