<a id="top"></a>
# CV Notebook 6 — Object Detection: Single-Stage Methods (Deep Dive)

> Per-notebook companion to the master guide. For the full cross-notebook cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §6](../CV_Revision_Guide.md#6-module6). **This deep-dive is standalone** — every concept it introduces is explained end-to-end here. You should never have to leave this file to understand a term.

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

- **One network, one forward pass, all detections** — no region-proposal step. Single-stage detectors predict every box, every class, everywhere, in one shot. That's where the speed comes from.
- **Grid cells own objects** — each cell predicts B anchor-based boxes; the cell containing the object centroid is "responsible" for it during training. Mental picture: a chessboard laid over the image, each square has its own little detector.
- **Confidence = `P(object) × IoU(pred, truth)`** — combines "is something here" with "how good is my box." Cheap pre-NMS filter.
- **Focal Loss = "focus gradient on hard examples"** — down-weights easy negatives so the network can't get away with predicting "no object" everywhere. The whole reason single-stage caught up to two-stage on accuracy.
- **Class imbalance is the hidden boss of single-stage** — millions of background anchors vs a handful of object anchors per image. Solving this (focal loss) was the central problem of the field for 2017–2019.

## 📖 Concept walkthroughs

> Beginner-first introduction of every concept this notebook touches. Each entry is a full Concept Definition Template — mental model → what / why / how / where / related → code → gotcha — substituted with this notebook's actual shapes, numbers, and code.

### One-stage vs two-stage detection

> **🪜 Mental model:** *Fast scan vs careful look.* One-stage = look once, predict everything (fast). Two-stage = propose, then classify (slower, more accurate — historically).

**What they are.** Two architectural families of object detectors, distinguished by whether they generate region proposals before classifying:
- **Two-stage** (Notebook 5) — propose ~300 candidate regions with an RPN, then classify and refine each one. Accuracy-first. Examples: Faster R-CNN, Mask R-CNN, Cascade R-CNN.
- **Single-stage** (this notebook) — divide the image into a grid; predict `(objectness, class probabilities, bbox offsets)` for every grid cell + anchor in a **single forward pass**. Speed-first. Examples: **YOLO** (v1–v8), **SSD**, **RetinaNet**, **YOLOX**.

**Why it matters.** The choice is the classic speed-vs-accuracy tradeoff in detection. Self-driving cars and real-time video want single-stage (30+ FPS); medical imaging or security review want two-stage (max accuracy, latency irrelevant). This notebook's whole motivation is *"two-stage is too slow → switch to YOLO"* — Faster R-CNN takes ~98 ms/frame, YOLOv5s takes ~6 ms (~15× faster). For 30 FPS autonomous-driving video, two-stage is just impossibly slow. It's asked in nearly every CV interview: *"which detector would you pick for X?"*

**How they differ.**

| | Two-stage | Single-stage |
|---|---|---|
| Speed | 5–10 FPS | 30–150 FPS |
| Accuracy (mAP@0.5:0.95 on COCO) | ~40–50 | ~35–50 (closing the gap) |
| Architecture | RPN + classifier head | Single dense prediction head |
| Anchor count | ~9 per location × 300 proposals = ~2 700 evaluations | thousands per image (e.g., 25 200 for YOLOv5 @ 640) |
| Hardest part | RPN training, NMS | **class imbalance** (focal loss invented to solve it) |
| Best at | High-precision, single-class detail | Real-time multi-object scenes |

**Where they're used.**
- **Two-stage:** offline detection in medical imaging, retail-audit pipelines, security review (where you can afford 100ms/frame for higher accuracy).
- **Single-stage:** real-time autonomous driving (Waymo, Cruise), video analytics, mobile / edge devices, drone obstacle detection.
- This notebook demonstrates the single-stage path: load a pretrained YOLOv5 ONNX file via OpenCV DNN, run inference per frame on a road video, post-process and draw the boxes.

**Related terms.**
- **YOLO** — the canonical single-stage detector (next entry).
- **Faster R-CNN** — the canonical two-stage detector (Notebook 5).
- **DETR** — newer transformer-based detector; neither classical one-stage nor two-stage (uses bipartite matching, skips NMS).
- **Focal loss** — the trick that closed most of the accuracy gap for single-stage.
- **Anchor-free** — recent trend (CenterNet, FCOS, YOLOX); single-stage without predefined anchors.

```python
# Single-stage one-liner (Ultralytics)
from ultralytics import YOLO
results = YOLO('yolov5s.pt')('road.jpg')   # one forward pass, all detections
# vs two-stage one-liner (torchvision)
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
preds = model([img])                        # RPN + classifier, two passes internally
```

**Gotcha.** Modern YOLOs (v5+, v8) have closed the accuracy gap. Don't assume "single-stage = inaccurate" — YOLOv8 ≥ Faster R-CNN on COCO with 5× the speed. The historical "two-stage is more accurate" was true 2017–2019 and is mostly false today.

### YOLO grid + output decoding (the notebook's signature mechanic)

> **🪜 Mental model:** *Chessboard of mini-detectors.* The image is divided into an `S × S` grid; each cell is responsible for any object whose centre falls inside it, and predicts `B` bounding boxes (one per anchor) as **offsets from the cell** (centre) and **scale multipliers on the anchor** (size).

**What it is.** **YOLO** ("You Only Look Once") produces a single output tensor of shape `S × S × B × (5 + C)` per scale:
- `S × S` — spatial grid (e.g., 13×13 for YOLOv3 at 416×416 input, downsample 32×; or 80×80, 40×40, 20×20 for YOLOv5 at 640).
- `B` — number of anchors per cell (typically 3 per scale in YOLOv3+).
- `5 + C` — per anchor: `(tx, ty, tw, th, objectness, c_1, c_2, …, c_C)`.

`tx, ty` are **offsets within the cell** (sigmoid → in `[0, 1]`); `tw, th` are **log-scale relative to the anchor**; `objectness` is "is something here"; `c_1..c_C` are per-class probabilities. These are *raw outputs*; you must **decode** them into absolute pixel boxes before you can draw anything.

YOLOv3+ predicts at **three scales** simultaneously — coarse grid (13×13) for big objects, medium (26×26), fine (52×52) for small. That's where YOLOv5's `(1, 25 200, 85)` output comes from for COCO: `25 200 = 3 scales × 3 anchors × (80² + 40² + 20²)` (for 640×640 input), and `85 = 5 + 80 COCO classes`.

**Why it matters.** This grid-based formulation is *the* defining trick of single-stage detection and the most-asked YOLO interview question (*"walk me through how YOLO predicts bounding boxes"*). Without understanding the decoding math, post-processing your own YOLO outputs is impossible — you'll silently get boxes that look reasonable but are systematically shifted or rescaled. This notebook implements the decoding by hand in `post_process` because OpenCV DNN doesn't bundle it.

**How decoding works (the math, with word-by-word translation).**
```
bx = (sigmoid(tx) + cx) / S * img_w      # box centre x in pixels
by = (sigmoid(ty) + cy) / S * img_h      # box centre y in pixels
bw = anchor_w * exp(tw)                  # box width in pixels (anchor × log-scale)
bh = anchor_h * exp(th)                  # box height in pixels
conf = sigmoid(objectness) * sigmoid(class_prob)   # final confidence used for NMS
```

In words, line by line:
1. *"`sigmoid(tx)` keeps the centre's x-offset inside the cell (always in `[0, 1]`); add the cell's grid column index `cx`; divide by grid size `S` to get a fraction of the image; multiply by image width to get pixels."*
2. Same logic in y.
3. *"Width = anchor's preset width times an exponential scale factor."* The `exp` lets the network shrink (`tw < 0`) *or* grow (`tw > 0`) the anchor symmetrically in log space — important so the loss doesn't favour growing over shrinking.
4. Same logic in height.
5. *"Final confidence = the sigmoid of the objectness score times the sigmoid of the predicted class probability."* Threshold this before NMS.

**Where it's used.**
- Inside every YOLO post-processing function.
- Bundled inside `ultralytics.YOLO` (you never see it), but you write it by hand when running YOLOv5 via **OpenCV DNN** or **ONNX Runtime** — exactly what this notebook does in **cell 5** (`post_process`).
- Hand-coded YOLO decoders in C++ / Rust / mobile inference engines (TFLite, CoreML).

**Related terms.**
- **Anchor box** — supplies the `(anchor_w, anchor_h)` that `exp(tw)` multiplies (next entry).
- **Grid cell** — the `(cx, cy)` index.
- **Stride** — `img_w / S`; the pixel size of one grid cell. For 416×416 input with S=13, stride = 32.
- **Sigmoid / exp** — the activations applied to raw outputs; sigmoid for *bounded* outputs (centre, objectness, class probs), exp for *positive* outputs (width, height).
- **Objectness × class score** — the final per-class confidence (next entry).

```python
# Decoding one anchor's raw output to a pixel bbox
def decode(tx, ty, tw, th, cx, cy, S, anchor_w, anchor_h, img_w, img_h):
    bx = (sigmoid(tx) + cx) / S * img_w
    by = (sigmoid(ty) + cy) / S * img_h
    bw = anchor_w * np.exp(tw)
    bh = anchor_h * np.exp(th)
    return bx, by, bw, bh                # centre + size, in pixels
```

**Gotcha.** **Output format differs between YOLO versions.** v3 uses `(tx, ty)` sigmoid + `(tw, th)` exp; **v5 uses a modified scaling** (`sigmoid(t) * 2 - 0.5` for centres, `(sigmoid(t) * 2)²` for sizes) to expand the predictable range and prevent gradient saturation; v8 is partially anchor-free with different math. **Always match the decoding code to the model version**, or your boxes are silently off.

### Anchor boxes (notebook context: K-means selection)

> **🪜 Mental model:** *Pre-shaped picture frames.* Instead of guessing a box from scratch, the network nudges one of a few standard rectangles into place.

**What it is.** **Anchor boxes** (also called "anchors" or "priors") are predefined rectangle shapes assigned to each grid cell. Instead of having the network output absolute bbox coordinates from scratch (a high-variance regression problem), each cell offers `k` anchor "templates" and the network outputs **offsets and log-scale factors** to nudge them into place. For YOLOv3+, three anchors per scale, three scales → 9 anchors total per cell. **For YOLOv3+, the anchors are not hand-set — they're K-means clustered from training-data bbox sizes.**

**Why it matters.** Anchors transform the bbox-regression problem from *"predict absolute coordinates"* (hard, high-variance) into *"refine a starting guess"* (easy). They also let the network encode *aspect ratios* explicitly — a tall anchor for pedestrians, a wide one for cars. **The notebook-specific twist:** YOLOv3+ derives anchors via K-means on the training set's `(width, height)` pairs, so the anchors are tuned to *your* dataset. For unusual aspect ratios (tall thin pedestrians, wide vehicles, tiny drones), default COCO anchors miss objects entirely — but K-means-derived anchors fit. Ultralytics YOLO runs K-means automatically when you train from scratch on `data.yaml`.

**How they work (with K-means selection).**
1. Collect all ground-truth bbox sizes from the training set as `(w, h)` pairs.
2. Run K-means with `k = num_anchors` (typically 9 for YOLOv3+) on those pairs. The resulting cluster centres become the anchor templates.
3. At each grid cell, the network outputs `(tx, ty, tw, th)` per anchor — offsets from the cell + log-scale factors.
4. During training, each ground-truth box is matched to the anchor with highest IoU (typically requires IoU ≥ 0.5); that anchor becomes "responsible" for predicting it.
5. At inference, decode: `bx = (sigmoid(tx) + cx) / S * img_w`; `bw = anchor_w * exp(tw)` (etc.).
6. Filter by `objectness × class_prob`, per-class NMS, return final boxes.

**Where they're used.**
- This notebook implicitly — YOLOv5 was trained with K-means-derived anchors; the `yolov5s.onnx` weights bake them in.
- Every modern anchor-based detector: YOLOv2–v5, SSD, RetinaNet.
- **Anchor-free** detectors (CenterNet, FCOS, YOLOX, DETR) skip them — they predict bbox sizes directly and have no anchor hyperparameter to tune.

**Related terms.**
- **Anchor-free detection** — predict bbox directly; cleaner pipeline, slower convergence.
- **K-means anchor selection** — the algorithm for picking anchors from training data.
- **IoU matching** — how anchors get assigned to ground truths during training.
- **Default boxes** — SSD's term for the same idea.
- **Priors** — RetinaNet's term.

```python
# K-means anchor selection (sketch)
from sklearn.cluster import KMeans
wh = np.array([(b[2]-b[0], b[3]-b[1]) for b in gt_boxes])  # widths, heights
anchors = KMeans(n_clusters=9).fit(wh).cluster_centers_     # 9 anchor templates
```

**Gotcha.** **Default anchors (COCO-derived) may not fit your data.** For datasets with unusual aspect ratios (long thin objects, tiny objects), K-means cluster your bboxes and use the resulting sizes. The notebook uses COCO-pretrained anchors because the demo classes (cars, pedestrians, traffic lights) overlap with COCO — for a custom dataset, anchor recalibration is one of the first things to check when accuracy is low.

### Objectness score

> **🪜 Mental model:** *"Is there anything here?"* — a single number per bbox prediction saying how confident the network is that the box contains *some* object.

**What it is.** The **objectness score** is one of the numbers predicted per anchor in single-stage detectors. It's the probability that the box contains an object — **independent of which object**. Formally it's defined as `P(object) × IoU(predicted_box, ground_truth)` — high only when both *"there is an object"* and *"my box is well-aligned"* hold. The final per-class confidence used at inference is `objectness × class_prob`.

In the notebook's `post_process` (cell 5):
```python
confidence = row[4]           # the objectness score for this anchor
class_scores = row[5:]        # 80 COCO class probabilities for this anchor
```

**Why it matters.** Objectness lets the detector separate *"this looks like a thing"* from *"this thing is a car."* It enables a cheap early filter: drop any prediction with `objectness < 0.5` *before* multiplying with class probabilities — this slashes the candidate set from 25 200 (YOLOv5 raw output) to a few hundred before NMS. The notebook applies *two* thresholds in series: first `objectness < 0.5` (cheap), then `class_scores[class_id] < 0.45` (more expensive, class-specific). Without this two-stage filter, NMS over 25 200 boxes would be unacceptably slow even on GPU.

**How it works.**
- Trained as a binary cross-entropy target: 1 for anchors matched to a ground-truth box, 0 for unmatched anchors.
- The IoU-encoded definition (`P(object) × IoU`) makes the network *self-aware* — it knows when its bbox is misaligned, not just whether something's there.
- At inference, `sigmoid(objectness_raw)` is applied to keep the value in `[0, 1]`.

**Where it's used.**
- YOLOv1–v8 (with minor variations across versions).
- This notebook in **cell 5** as `confidence = row[4]`, gated by `OBJECT_SCORE_THRESHOLD = 0.5`.
- SSD doesn't use a separate objectness; it folds "background" into a class.
- RetinaNet uses focal loss on per-anchor classification without an explicit objectness.

**Related terms.**
- **Confidence threshold** — the cutoff applied to `objectness × class_prob` before NMS.
- **Focal loss** — used in RetinaNet to handle the same class imbalance objectness was created for.
- **Background class** — SSD's alternative; one of the class-prob outputs is "no object."
- **Class score** — the *which-object* probability; multiplied with objectness for final confidence.

```python
# From this notebook's post_process
for row in rows:                              # rows shape (25200, 85)
    confidence = row[4]                       # objectness
    if confidence < OBJECT_SCORE_THRESHOLD: continue        # cheap early filter
    class_scores = row[5:]
    class_id = np.argmax(class_scores)
    if class_scores[class_id] < CLASS_CONFIDENCE_THRESHOLD: continue
```

**Gotcha.** Don't confuse objectness with class confidence. `objectness=0.9, class_prob=0.1` means *"I'm sure something's here, but I'm not sure it's a car."* Always **multiply** before thresholding — the network's certainty about *what* and certainty about *whether* are independent signals.

### Class imbalance in single-stage detection (the motivation for focal loss)

> **🪜 Mental model:** *A million haystacks, three needles.* Each image produces tens of thousands of anchor predictions, and 99%+ of them are "no object here." Vanilla cross-entropy drowns in the easy negatives — the rare positive gradients are inaudible in the noise.

**What it is.** **Class imbalance** in single-stage detection refers to the extreme ratio between *background* anchor predictions and *foreground* (object) anchor predictions per image. A YOLOv3 model at 416×416 emits ~10 000 anchor predictions; YOLOv5 at 640 emits **25 200**. An image with 3 cars has ~3–9 anchors that should fire positive and the rest (~25 190) that should fire negative. That's roughly a **1 000:1 negative-to-positive ratio**, far worse than typical classification problems (which are usually 1:1 to 10:1).

**Why it matters — what goes wrong without a fix.** Cross-entropy loss summed over 25 200 anchors is dominated by the ~25 190 easy negatives. Each easy negative contributes a *tiny* loss (it's correctly classified at low predicted-object probability), but tiny × 25 000 ≫ a few × the per-positive loss. The gradient ends up pushing the network to predict *"no object"* everywhere — that *minimises* the loss while doing the wrong thing. Early single-stage detectors (YOLOv1, YOLOv2, SSD) all hit this wall and lagged two-stage detectors by ~10 mAP. **Two-stage detectors don't have this problem** because the RPN filters background *before* the classifier sees anything (it sees ~300 proposals, not 25 200 anchors). This was the central tension of the field for 2017–2019, and the most-asked focal-loss interview question is *not* "what's the formula" but *"why was it needed."*

**How it manifests in the notebook.** The notebook's `OBJECT_SCORE_THRESHOLD = 0.5` and `CLASS_CONFIDENCE_THRESHOLD = 0.45` are **inference-time** heuristics to filter out background — they don't help training. During training, the same imbalance would dominate the loss; YOLOv5 internally addresses it via a combination of: (a) IoU-based positive-anchor assignment (only the few high-IoU anchors are "positive"), (b) weighted BCE on objectness, (c) loss-component balancing. Ultralytics handles all this for you when you call `model.train(...)`.

**Where it shows up.**
- Every single-stage detector (YOLO, SSD, RetinaNet).
- Implicitly handled by `torchvision.ops.sigmoid_focal_loss`.
- Asked about in every CV interview (*"why was focal loss invented?"*).
- System-design rounds: *"design a real-time defect detector for an assembly line"* — defects are rare, so class imbalance compounds the single-stage anchor imbalance.

**Related terms.**
- **Focal loss** — the fix; down-weights easy negatives in the loss (next entry).
- **Hard-negative mining** — older alternative; manually sample hard background anchors during training. SSD uses this.
- **RPN** — two-stage's natural defence against the same problem (filters background before classification).
- **Cross-entropy loss** — what focal loss extends.
- **OHEM (Online Hard Example Mining)** — yet another fix; pick the top-`k` hardest examples per batch and only backprop through those.

```python
# Focal loss in PyTorch (the standard fix for single-stage class imbalance)
from torchvision.ops import sigmoid_focal_loss
loss = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean')
```

**Gotcha.** Focal loss only helps **when imbalance is severe**. For balanced classes (cats vs dogs), it actively *hurts* — suppressing the legitimate easy-example gradient loses useful training signal. Apply focal loss to the anchor-objectness output of a single-stage detector; don't slap it on every multi-class classification problem.

### Focal loss

> **🪜 Mental model:** *Pay attention to the hard ones.* Down-weight easy examples in the loss so the gradient focuses on the hard, misclassified ones.

**What it is.** **Focal loss** (Lin et al., RetinaNet, 2017) is a variant of cross-entropy designed for severe class imbalance — specifically the millions-of-background-boxes-vs-few-objects problem in single-stage detection. The formula:

```
FL(p) = −α · (1 − p)^γ · log(p)
```

Word-by-word translation:
- `p` — the predicted probability of the true class (what the network thinks the right answer is).
- `−log(p)` — **standard cross-entropy**: large when `p` is small (model is wrong/uncertain), small when `p` is close to 1 (model is right and confident).
- `(1 − p)^γ` — **modulating factor**: small when `p` is high (easy example, near-zero contribution to loss), large when `p` is low (hard example, full contribution). With `γ=2` and `p=0.9`, this factor is `0.01` → 100× down-weighting of the easy case.
- `α` — **class-balancing weight** (typically 0.25). Down-weights the over-represented class globally.
- `γ` — **focusing parameter** (typically 2). Higher = more aggressive down-weighting of easy examples.

So *focal loss = standard cross-entropy multiplied by a factor that shrinks the loss for easy examples and keeps it (nearly) full for hard ones.*

**Why it matters.** Without focal loss, single-stage detectors with thousands of anchors are dominated by easy negatives — the network just learns to predict "no object" everywhere. Focal loss was the key innovation that let **RetinaNet match two-stage accuracy with single-stage speed** in 2017, winning ICCV's best-paper award. Modern YOLO variants (v4+) use focal-style losses or related rebalancing. It's the most-asked focal-loss interview question — and as noted above, the right answer leads with *"why,"* not *"what's the formula."*

**How it works.** When `p` (predicted probability of the true class) is high — say 0.9 — `(1 − p)^γ = 0.01` (with γ=2), so the loss contribution is 100× smaller than vanilla cross-entropy. Easy examples (mostly backgrounds) contribute negligible gradient; hard examples (genuine objects, ambiguous cases) dominate. The optimiser now spends most of its capacity on the hard examples — exactly where it should.

**Where it's used.**
- **RetinaNet** — the paper that introduced it.
- Newer YOLO variants (v4 and beyond) — focal-style anchor classification losses.
- **EfficientDet**, FCOS, YOLOX — most modern single-stage detectors.
- Available as `torchvision.ops.sigmoid_focal_loss` in PyTorch.
- **NOT in this notebook directly** — YOLOv5's pretrained weights are *the product* of training with focal-style losses, and inference here only consumes those weights. But every conversation about single-stage detection eventually reaches focal loss.

**Related terms.**
- **Cross-entropy** — the standard classification loss focal loss extends.
- **Class imbalance** — the problem focal loss solves (previous entry).
- **Hard-negative mining** — older alternative; manually pick hard negatives during training.
- **Label smoothing** — different regularisation technique; not a replacement for focal loss.
- **α, γ** — focal loss's two hyperparameters.

```python
from torchvision.ops import sigmoid_focal_loss
# logits: (N,) raw predictions; targets: (N,) 0/1 labels
loss = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean')
```

**Gotcha.** Focal loss only helps when imbalance is severe. For balanced classes (cats vs dogs), plain cross-entropy is better — focal loss suppresses the legitimate easy-example gradient signal and slows convergence. The typical hyperparameters `α=0.25, γ=2.0` are RetinaNet defaults; tuning them is often necessary on new datasets.

### SSD (Single-Shot Detector)

> **🪜 Mental model:** *Multi-scale YOLO.* Predict bboxes at multiple feature-map scales — small objects from shallow maps (high resolution), large objects from deep maps (low resolution).

**What it is.** **SSD** (Liu et al., 2016) is a single-stage detector with the same *"one forward pass"* idea as YOLO but with **multi-scale prediction**: instead of predicting at a single output grid, SSD attaches detection heads to *multiple* feature-map scales inside the backbone. Shallow (high-res) maps predict small objects; deep (low-res) maps predict large objects. SSD uses anchor boxes (called "default boxes" in the paper) and runs ~30–60 FPS — slightly slower than YOLO, slightly different speed/accuracy tradeoff.

**Why it matters.** YOLOv1 struggled with small objects because its grid was coarse (7×7). SSD's multi-scale outputs handled small objects much better and pushed single-stage detection forward in 2016. Later YOLOs (v3+) adopted the same multi-scale idea via FPN (Feature Pyramid Network). SSD is still in heavy use as a *lighter alternative to YOLO* on mobile — **SSD-MobileNet** is the workhorse mobile detector for Coral Edge TPU, Jetson Nano, and older iOS deployments. The notebook references SSD in the speed/accuracy comparison table as a deployment-time alternative.

**How it works.**
1. Backbone (typically VGG-16 or MobileNet) produces feature maps at several resolutions — e.g., 38×38, 19×19, 10×10, 5×5, 3×3, 1×1.
2. At each scale, a small detection head predicts `(objectness, class probs, bbox offsets)` for `k` default boxes per location.
3. Default boxes are hand-set with varied aspect ratios.
4. All predictions across all scales are combined → NMS → final boxes.

The total anchor count is on par with YOLO (~8 000 for SSD300), but spread across more scales.

**Where it's used.**
- Mobile / edge devices when YOLO is too heavy. SSD-MobileNet TFLite for Android.
- `torchvision.models.detection.ssd300_vgg16(pretrained=True)` for server inference.
- Google Coral Edge TPU runs SSD-MobileNet natively.
- **NOT in this notebook** — referenced only in the speed/accuracy table; this notebook runs YOLOv5 only.

**Related terms.**
- **FPN (Feature Pyramid Network)** — modern way to add multi-scale prediction to any detector; pioneered by SSD's idea, formalised by RetinaNet.
- **YOLOv3+** — adopted multi-scale prediction inspired by SSD.
- **RetinaNet** — newer single-stage; multi-scale + focal loss.
- **MobileNet** — lightweight backbone often paired with SSD for mobile.
- **Default boxes** — SSD's term for anchor boxes.

```python
import torchvision
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()
preds = model([img_tensor])
# Mobile equivalent: tflite_runtime.Interpreter(model_path='ssd_mobilenet.tflite')
```

**Gotcha.** SSD's multi-scale design helps small objects only modestly — the shallow feature maps are *semantically weak* (they've seen few conv layers, so they encode edges but not "this is a tire"). **FPN's top-down pathway** is what truly solved small-object detection: it injects semantic info from deeper layers back into shallow ones.

[🔝 Back to top](#top)

## 🧠 Cell-by-cell walkthrough

### 1. Why single-stage?
Two-stage detectors (R-CNN family) have separate proposal + classification stages → multiple forward passes. For 30+ FPS video, this is too slow. Single-stage: one network outputs all bboxes in one shot.

### 2. YOLO output structure
For an `S × S` grid with `B` anchors and `C` classes:
- **Output tensor:** `S × S × B × (5 + C)`
  - `5 = (tx, ty, tw, th, objectness)`
  - `C = class probabilities`
- YOLOv3: three scales (13×13, 26×26, 52×52) for small/medium/large objects.

### 3. Loading YOLOv5 via OpenCV DNN (the portable way)

`cv2.dnn.readNet` consumes the ONNX file produced by `model.export(format='onnx')`. The advantage: no PyTorch dependency at runtime, runs on any platform OpenCV supports.

```python
import cv2

modelWeights = "yolov5s.onnx"
net = cv2.dnn.readNet(modelWeights)

with open("coco.names.txt", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')   # 80 COCO classes
```

### 4. Inference (forward pass)

`blobFromImage` is OpenCV's preprocessing helper: scales pixel values (`1/255`), resizes to the network's expected input size, optionally swaps BGR→RGB (`swapRB=True` is mandatory for YOLO because OpenCV reads BGR but YOLO trained on RGB), and packs into a 4-D NCHW blob.

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

Two-stage filter to slash 25 200 predictions to a handful: (1) objectness threshold (cheap), then (2) per-prediction class-confidence threshold. Surviving boxes are decoded to image pixels (centre → corner conversion, plus `x_factor` / `y_factor` to map from 640-input space to the original image size), then NMS deduplicates.

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

`indices.flatten()` because `cv2.dnn.NMSBoxes` returns a 2-D-ish structure that flattens to a 1-D list of survivor indices.

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

Same `yolo_forward + post_process + draw` chain, just looped over `VideoCapture` frames. On a GPU at ~6 ms/frame this comfortably hits 30 FPS.

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

Everything in cells 3–7 collapses into 3 lines. Use OpenCV DNN when you need a PyTorch-free deployment; use Ultralytics for training, fine-tuning, and any setup where PyTorch is available.

```python
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
results = model('road.jpg')         # or video, webcam, batch directory
results[0].show()                   # auto-annotated
results[0].save('out.jpg')
```

[🔝 Back to top](#top)

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

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **`blobFromImage` parameters matter.** `swapRB=True` because OpenCV is BGR, YOLO trained on RGB. Forget it → all colors wrong → no detections.
2. **`scalefactor=1/255`** for YOLOv5; some other models need different preprocessing.
3. **YOLOv5 output is one big tensor** `(1, 25200, 85)` — 25200 anchor predictions across 3 scales × 3 anchors × multiple grid sizes. Filter by objectness × class-confidence before NMS.
4. **`cv2.dnn.NMSBoxes` expects `(x, y, w, h)`** corner-format. Don't pass `(x1, y1, x2, y2)`.
5. **Per-class NMS is best** for crowded scenes. If you NMS globally, a car next to a truck might suppress the truck.
6. **YOLOv5 has 5 variants** — `n` (nano), `s` (small), `m`, `l`, `x` (extra-large). Speed vs accuracy: `n` is ~10× faster but ~10 mAP lower than `x`. Pick based on deployment target.
7. **CocoNames file ordering matters.** YOLO predicts class indices; you map them to names. Wrong file → all labels off by one.
8. **`net.forward()` returns a list of arrays** (one per output layer). For YOLOv5 ONNX there's typically one; for YOLOv3 there are three (one per scale).
9. **Output-decoding math differs between YOLO versions** — v3, v5, v8 use slightly different formulas. Match the decoding code to the model version or boxes are silently miscalibrated.
10. **Default anchors may not fit your data** — for unusual aspect ratios, retrain (or fine-tune) so K-means picks anchors matched to your distribution.
11. **Focal loss only helps when imbalance is severe** — for balanced classes it actively hurts by suppressing easy-example gradient.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

**Q1.** Why is YOLO faster than Faster R-CNN?
→ One forward pass over the entire image with dense prediction at every grid cell, vs Faster R-CNN's RPN + classifier (two networks, two passes internally). No proposal-generation step, no per-proposal RoI Pool. *(adapted from `andrewekhalel/MLQuestions`)*

**Q2.** What does "objectness" mean in YOLO?
→ `P(object) × IoU(pred, truth)`. Combines "is there an object here?" with "how good is my bounding box?" Multiplied with class probability to get final confidence; thresholded twice (objectness cut + class cut) to filter the candidate set cheaply before NMS. *(common FAANG YOLO question)*

**Q3.** What is Focal Loss and what problem does it solve?
→ `−α(1−p)^γ log(p)`. The `(1−p)^γ` factor down-weights easy examples (high `p`) so the gradient focuses on hard, misclassified ones. Solves the extreme class imbalance of single-stage detection: 10⁴–10⁵ background anchors per image vs a handful of foreground anchors — without focal loss, the network just learns "predict no object." *(adapted from `chiphuyen/ml-interviews-book`, detection chapter)*

**Q4.** Why does single-stage struggle more with class imbalance than two-stage?
→ Two-stage filters background out via the RPN *before* the classifier sees anything (it sees ~300 high-objectness proposals). Single-stage classifies every anchor on every grid cell at every scale → ~10 000–25 000 background anchors per image dominate the loss. The 1 000:1 negative-to-positive ratio is what focal loss was invented to fix. *(common FAANG follow-up)*

**Q5.** What's the difference between one-stage and two-stage detection?
→ Two-stage: RPN proposes regions → classifier head refines each (~98 ms/frame, ~70 mAP-50 on COCO). One-stage: dense prediction across a grid in one forward pass (~6–30 ms/frame, ~56–66 mAP-50). Pick one-stage for real-time / mobile; two-stage for offline maximum-accuracy work. Modern YOLOs have closed most of the accuracy gap. *(common FAANG architecture question)*

**Q6.** Multi-scale detection — why use multiple feature map sizes?
→ Small objects activate only the shallow (high-res) layers because they don't survive deep-layer downsampling; large objects span many deep-layer receptive fields. Predicting at multiple scales (YOLOv3's 13/26/52 grids, SSD's 6 scales) means small objects come from the fine grid and large objects from the coarse grid. *(adapted from `alexeygrigorev/data-science-interviews`)*

**Q7.** Walk me through decoding a YOLO output tensor to a pixel bbox.
→ For each of the `S × S × B` anchors: `bx = (sigmoid(tx) + cx) / S * img_w`, `by = (sigmoid(ty) + cy) / S * img_h`, `bw = anchor_w * exp(tw)`, `bh = anchor_h * exp(th)`. Then `conf = sigmoid(obj) × sigmoid(class_prob)`, threshold, and NMS. *(common YOLO whiteboard question)*

**Q8.** How are YOLO's anchor sizes chosen?
→ K-means clustered on the training-set ground-truth `(width, height)` pairs. The cluster centres become the anchor templates. Tunes anchors to the data distribution — default COCO anchors miss objects on datasets with unusual aspect ratios (drones, faces, vehicles). *(common interview design question)*

**Q9.** What does the `(1, 25200, 85)` shape of YOLOv5's output mean?
→ Batch size 1, 25 200 anchor predictions (3 scales × 3 anchors × the sum of grid² over scales for 640×640 input), and 85 = 4 (bbox: cx, cy, w, h) + 1 (objectness) + 80 (COCO class probabilities). You filter by objectness × class prob, decode, and run NMS to get the few-dozen final boxes. *(common YOLOv5 deployment question)*

**Q10.** When would you pick SSD over YOLO?
→ When deploying to a fixed-function edge device (Coral Edge TPU, older mobile chips) where SSD-MobileNet variants have hand-optimised inference kernels and a mature TFLite toolchain. YOLOv5n is competitive on most edge targets now, but the SSD-MobileNet ecosystem still has the best support on older devices. *(common FAANG deployment question, original)*

**Q11.** Why does the notebook filter by objectness *before* class confidence?
→ Cost. Objectness is a single scalar per anchor; class confidence requires `argmax` over 80 entries. Filtering by objectness first cuts the candidate set from 25 200 to maybe a few hundred *before* doing the expensive class step — total work drops by ~100×. *(original, implementation-level)*

**Q12.** If you have to design a real-time defect detector for an assembly line where defects are 0.01% of frames, what changes?
→ The class imbalance compounds the single-stage anchor imbalance — focal loss alone won't be enough. Add: (a) heavy data augmentation specifically on defect crops, (b) class-balanced sampling during training (oversample defect frames), (c) two-stage architecture (RPN filters first) if latency permits, (d) anomaly-detection fallback (one-class SVM on YOLO embeddings) for unseen defect types. *(original, senior design-judgment)*

[🔝 Back to top](#top)

## 🪞 Extra ladder — anchor-free vs anchor-based

**Basic** — anchor-based (YOLOv3, SSD): predefined templates per grid cell.

**Intermediate** — anchor-free (CenterNet, FCOS): predict center heatmap + bbox size directly. Fewer hyperparameters, simpler implementation.

**Advanced** — **DETR** (Detection Transformer, 2020): no anchors, no NMS. Uses a transformer + bipartite matching for end-to-end set prediction. Slower training but cleaner pipeline. **Deformable DETR** (2021) made it competitive on speed.

[🔝 Back to top](#top)

## What comes next

Single-stage detection gave bboxes. [Notebook 7 →](../7.Object%20segmentation/) goes finer: **per-pixel** classification (segmentation) — U-Net, FCN, Mask R-CNN.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
