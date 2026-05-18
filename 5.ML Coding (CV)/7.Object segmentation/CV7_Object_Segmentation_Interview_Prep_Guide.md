<a id="top"></a>
# CV Notebook 7 — Object Segmentation (Deep Dive)

> Per-notebook companion to the master guide. For the cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §7](../CV_Revision_Guide.md#7-module7). This deep dive is **standalone** — every concept below carries its own full Concept Definition Template entry (mental model + what + why + how + where + related + code + gotcha), substituted with the notebook's actual shapes, numbers, and code. You should never need to click through to the master to understand a term.

## What this notebook actually demonstrates

- **Google-Pixel portrait-mode** use case — classify every pixel as person / background, then Gaussian-blur the background pixels only.
- **Dataset:** 18,698 portrait images, 128×128 RGB, with binary person-vs-background masks.
- **Two architectures compared:** FCN-8 (VGG-16 encoder + late skip-additions) vs **U-Net** (dense skip-concats at every level). Both reach high Dice, but U-Net gives visibly sharper boundaries — the *spatial-detail* win you can't see in a single number.
- **Custom loss & metric:** implements **Dice coefficient** and **Dice loss** from scratch (instead of plain cross-entropy) because portrait masks are imbalanced.
- **Inference demo:** uses `np.argmax` on the per-pixel softmax to get a class mask, then `cv2.GaussianBlur` + `np.where` to composite the portrait-blur effect.
- **Conceptual extensions:** introduces **Mask R-CNN** (instance segmentation) and **DeepLab v3** (atrous / dilated convs) without coding them up — they're the "what to use in production" pointers.

## 🪜 Mental anchors for this notebook

- **Segmentation = classification *per pixel*.** Output is the same `(H, W)` shape as the input; each cell carries a class label. Headline difference from detection (one box per object) or classification (one label per image).
- **Encoder shrinks → bottleneck → decoder grows.** A "U" of feature maps. Encoder is good at "what's in the picture"; decoder restores "where exactly" by upsampling.
- **Skip connections are the *where* cable.** Without them, the decoder has only the bottleneck's blurry summary; with them, it gets a fresh copy of the encoder's high-resolution feature map at every level — so boundaries stay crisp.
- **Dice = F1 score for masks.** Reward whole-region overlap rather than averaging across millions of pixels — robust to the 95% background pixels that drown plain cross-entropy.
- **`Conv2DTranspose` is *learnable* upsampling; `UpSampling2D` is *fixed* upsampling.** The notebook's U-Net uses the cheaper `UpSampling2D + Conv2D` pair to avoid checkerboard artefacts; FCN uses `Conv2DTranspose`.

[🔝 Back to top](#top)

## 📖 Concept walkthroughs

Beginner-first introductions to every concept this notebook teaches. Each concept entry is a **full Concept Definition Template** — mental model + what + why + how + where + related + code + gotcha — with the notebook's actual shapes (`128×128×3` images, U-Net with depth 4, Dice loss) substituted in. The deep-dive is standalone; you should never need to click out to understand a term.

### Segmentation vs detection vs classification

> **🪜 Mental model:** *Per-image label vs per-object box vs per-pixel label.* The output shape is the cleanest way to remember which task is which.

**What it is.** These are the three core image-understanding tasks in computer vision, distinguished by *granularity* of the prediction. **Classification** assigns one label to the whole image (e.g., "this is a portrait"). **Object detection** finds each object instance and draws an axis-aligned **bounding box** (a rectangle) around it, attaching a class label and confidence. **Semantic segmentation** goes finer still — it assigns a class label to *every single pixel*. This notebook does semantic segmentation: input is a `(128, 128, 3)` image, output is a `(128, 128, 2)` map of per-pixel `[background, person]` probabilities.

**Why it matters.** The portrait-blur task needs *every person pixel* labelled separately from background — a bounding box would blur the person's shoulders along with the actual background. Picking the right task is the first design decision in any CV project, and interviewers love to test whether candidates can argue *why* one is appropriate over another. Pixel-level output is the entire point when the downstream operation (blur, composite, masking) is itself pixel-level.

**How it works (the output-shape lens).**

| Task | Output shape | Example | Notebook reference |
|---|---|---|---|
| **Classification** | `(num_classes,)` — one vector per image | `[0.97, 0.03]` ("portrait, not landscape") | Module 1 (the MLP baseline) |
| **Detection** | `(num_objects, 5)` — per-object `(class, x, y, w, h)` | `[("person", 30, 12, 80, 110)]` | Module 6 (YOLO / R-CNN) |
| **Semantic segmentation** | `(H, W, num_classes)` — per-pixel class distribution | `(128, 128, 2)` softmax map | **This notebook** (U-Net / FCN) |
| **Instance segmentation** | List of `(class, score, mask)` triples | One mask per detected person | Mask R-CNN (mentioned only) |

The labels you need at training time mirror the output shape: one label per image for classification, a list of boxes per image for detection, a `(H, W)` mask per image for segmentation.

**Where it's used.**
- **Classification** — "is this image a cat or a dog?", quality screening, content moderation.
- **Detection** — autonomous driving (boxes around cars), retail (count items on a shelf).
- **Semantic segmentation** — portrait blur (this notebook), road / sidewalk / sky in self-driving, medical organ delineation.
- **Instance segmentation** — counting individual cells, AR object pinning, Zoom multi-person background blur.

**Related terms.**
- **Localisation** — classification + a single bounding box (one object only); midpoint between classification and detection.
- **Panoptic segmentation** — semantic + instance combined; every pixel gets both a class and (where applicable) an instance ID.
- **Pixel-wise classification** — synonym for semantic segmentation; emphasises that it's just classification done independently at each pixel position.
- **Dense prediction** — umbrella term for any task that outputs one value per pixel (segmentation, depth estimation, optical flow).

```python
# In this notebook, the output shape is what makes it segmentation:
out = layers.Conv2D(num_classes=2, kernel_size=1, activation='softmax')(d1)
# out.shape == (None, 128, 128, 2)   ← per-pixel class probabilities
```

**Gotcha.** "Image segmentation" alone is ambiguous — interviewers expect you to clarify *semantic vs instance vs panoptic*. Defaulting to one without naming the others is a red flag.

### Semantic vs instance segmentation

> **🪜 Mental model:** *Class-mask vs labelled-instances.* Semantic = "these are cat-pixels"; instance = "these are cat #1's pixels, these are cat #2's."

**What it is.** Both tasks output a per-pixel mask. The difference is whether the mask *distinguishes individual objects of the same class*. **Semantic segmentation** assigns one of `K` class labels to every pixel — all "person" pixels share the same label whether there's one person or ten. **Instance segmentation** assigns a class **and** a unique instance ID, so each individual person gets its own colour in the output mask. Panoptic segmentation is the union — every pixel gets a class, and "thing" classes (people, cars) also get an instance ID while "stuff" classes (road, sky) don't.

**Why it matters.** The choice depends on whether your downstream task needs to *count or track individual objects*. The portrait-blur problem (this notebook) doesn't — there's at most one person per image, and even with multiple people you'd blur the background the same way. So the cheaper semantic model (U-Net) is enough. If the use case were "Zoom-style multi-speaker focus where the active speaker stays sharp and the rest get blurred," you'd need instance segmentation (Mask R-CNN) to distinguish person #1 from person #2.

**How it works.**
- Semantic models output `(H, W, num_classes)` and you typically `argmax` along the last axis to get a `(H, W)` class index map. No notion of "this pixel belongs to cat #3."
- Instance models output a **list** of detected objects, each with a class, score, and a per-object mask (usually small, e.g., 28×28, pasted back into image coordinates via a bounding box).
- Training data differs too — semantic needs `(H, W)` masks with integer class labels; instance needs per-instance masks (each labelled separately).

**Where it's used.**
- **Semantic** — road segmentation, portrait blur (this notebook), medical organ delineation, satellite land-use.
- **Instance** — cell counting (microscopy), retail shelf audit, AR object pinning, robotic grasping.
- **Panoptic** — full-scene parsing for autonomous driving (every pixel gets a class + optional instance).

**Related terms.**
- **Panoptic segmentation** — every pixel gets a class, "thing" classes also get instance IDs.
- **Thing vs stuff** — "things" are countable objects (people, cars); "stuff" is amorphous (road, sky).
- **Mask R-CNN** — the standard instance-segmentation architecture; combines detection + per-ROI mask head.
- **U-Net / FCN / DeepLab** — the standard semantic-segmentation architectures.

```python
# Semantic output → one class index per pixel
preds = model.predict(X_test[:1])         # (1, 128, 128, 2)
mask  = np.argmax(preds, axis=-1)         # (1, 128, 128) — one label per pixel
# Instance output (Mask R-CNN-style) would be a list of dicts, each carrying a mask + box
```

**Gotcha.** "Segmentation" with no qualifier is ambiguous — always name the variant. Recruiters use this question to filter candidates who learned the buzzword but not the distinctions.

### U-Net architecture (encoder-decoder + skip connections)

> **🪜 Mental model:** *Information flows down the encoder, hits the bottleneck, flows back up the decoder, with side-doors at every level.* The shape on the page literally looks like a "U" — hence the name.

**What it is.** **U-Net** is a symmetric encoder-decoder neural network designed for semantic segmentation. The **encoder** (left half of the U) repeatedly applies `Conv → Conv → MaxPool` blocks that halve the spatial size and double the channel count — same shape as a classification CNN. The **decoder** (right half of the U) mirrors the encoder: it repeatedly applies `Upsample → Concatenate(encoder skip) → Conv → Conv` blocks that double the spatial size and halve the channels. At the bottom of the U sits the **bottleneck** — the smallest, deepest feature map. The "U" is closed by **skip connections** that wire each encoder block's output directly across to the matching decoder block.

**Why it matters.** Plain classification CNNs throw spatial information away through pooling — by the time you've reduced 128×128 to 8×8, you can say *what* is in the image but not *where exactly*. Segmentation needs both. U-Net is the canonical solution: the bottleneck gives the *what* (deep semantic features), the skip connections give the *where* (high-resolution spatial detail), and the decoder fuses them. Originally introduced for biomedical microscopy in 2015, U-Net is now the backbone of every modern segmentation network and even the denoiser inside Stable Diffusion. **For semantic segmentation interviews, U-Net is the default architecture you must be able to draw from memory.**

**How it works (concrete trace through this notebook).** Input shape: `(128, 128, 3)`.
1. **Encoder block 1:** `Conv2D(64) → Conv2D(64)` → save output `c1` (shape `128×128×64`) → `MaxPool` → `p1` (`64×64×64`).
2. **Encoder block 2:** `Conv2D(128) → Conv2D(128)` → save `c2` (`64×64×128`) → `MaxPool` → `p2` (`32×32×128`).
3. **Encoder block 3:** save `c3` (`32×32×256`) → pool → `p3` (`16×16×256`).
4. **Encoder block 4:** save `c4` (`16×16×512`) → pool → `p4` (`8×8×512`).
5. **Bottleneck:** `Conv2D(1024) → Conv2D(1024)` → `b` (`8×8×1024`).
6. **Decoder block 4:** `UpSampling2D` (`16×16×1024`) → `Concatenate([·, c4])` (`16×16×1536`) → `Conv2D(512) → Conv2D(512)`.
7. **Decoder blocks 3, 2, 1:** mirror — upsample, concat the matching encoder feature, two convs.
8. **Head:** `Conv2D(num_classes=2, kernel_size=1, activation='softmax')` → `(128, 128, 2)`.

Channel counts go `64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 2` — doubling then halving, mirroring the depth.

**Where it's used.**
- **This notebook's portrait-blur model** — vanilla U-Net trained from scratch.
- **Medical imaging** — organ, tumour, lesion segmentation (the field where U-Net was born).
- **Stable Diffusion's denoiser** — same U-shape, conditioned on text embeddings; segmentation's architecture quietly powers modern generative AI.
- **Self-driving lane segmentation, satellite imagery analysis, microscopy.**

**Related terms.**
- **Encoder-decoder** — the umbrella architectural pattern U-Net belongs to.
- **Bottleneck** — the smallest, deepest feature map; the network's "summary."
- **Skip connection** — the side-door wires that hand the decoder the encoder's high-resolution features.
- **FCN-8** — earlier (2015) fully-convolutional alternative; only two late skips with addition instead of concat → coarser boundaries.
- **DeepLab v3** — modern competitor using dilated/atrous convolutions instead of an explicit decoder; better for high-res inputs.

```python
# Encoder block 1 + matching decoder block 1 (from this notebook)
c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)   # save c1!
p1 = layers.MaxPooling2D()(c1)

# ... encoder + bottleneck ...

u1 = layers.UpSampling2D()(d2)                  # decoder side
u1 = layers.Concatenate()([u1, c1])             # ← the skip
d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(d1)
out = layers.Conv2D(2, 1, activation='softmax')(d1)
```

**Gotcha.** Input dimensions must be a multiple of `2^depth` (here `2^4 = 16`) — otherwise the encoder's pooled outputs and the decoder's upsampled features won't have matching `(H, W)` at concat time, and you'll get a shape-mismatch exception. Pad to a multiple of 16 (or 32 for deeper nets) before training.

### Why skip connections matter (full template — the beginner-killer)

> **🪜 Mental model:** *Without skip wires, the decoder is given a blurry thumbnail and asked to paint a sharp portrait.* Skip connections hand the decoder a fresh, high-resolution copy of the encoder's feature map at every level so it can recover the edges.

**What it is.** A **skip connection** in U-Net is a direct path from an encoder block's output to the matching decoder block's input. Instead of relying only on the upsampled bottleneck features (which have been compressed `16×16×` smaller spatially and then stretched back), the decoder *also* receives the encoder's same-resolution feature map at that level. The two feature maps are stacked along the channel axis (`Concatenate`) before the next conv block runs. So at decoder level k, the conv sees both "where edges were before downsampling" and "what semantic category each region is."

**Why it matters.** Downsampling (max-pool or stride-2 conv) is *lossy* — every pool step throws away three out of every four pixels' worth of spatial information. After four pools, a 128×128 image has shrunk to 8×8 — that's only 64 spatial positions describing the entire image. The decoder *cannot* paint sharp object boundaries from 64 positions alone, no matter how many channels it has. Skip connections re-inject the original high-resolution detail. **This is the single reason U-Net beats FCN on boundaries** — FCN's skips happen at only two late levels (Pool3 and Pool4), so most of the detail is lost by the time the decoder gets help.

**How it works (concrete trace through this notebook's U-Net).**
1. Encoder level 1 produces `c1` of shape `(128, 128, 64)`. We *save* `c1`.
2. After downsampling through levels 2, 3, 4 and the bottleneck, the decoder upsamples back to shape `(128, 128, 64)` (call it `u1`).
3. We concatenate: `Concatenate()([u1, c1])` → shape `(128, 128, 128)` — *the channel count doubles.*
4. A `Conv2D(64, 3)` block fuses the two streams into refined features.
5. Repeat at every decoder level (4 levels in this notebook).
6. **Important:** the encoder feature maps `c1, c2, c3, c4` must be *kept in GPU memory throughout the forward pass* — that's why deep U-Nets at high resolution explode VRAM.

**Where it's used (beyond this notebook).** Every modern U-shaped network: U-Net (concat), V-Net (3D U-Net), nnU-Net (medical), the denoising U-Net inside Stable Diffusion. Also conceptually identical to ResNet's residual connection — same idea, but ResNet *adds* `(out = F(x) + x)` instead of concatenating, which keeps the channel count constant.

**Related terms.**
- **Residual connection** — ResNet's variant; uses addition instead of concatenation. Same purpose (preserve information), different combinator.
- **Dense connection** — DenseNet pattern; every layer skips to every later layer.
- **FCN-style skip** — FCN-8 uses `Add` at only two late levels — that's why its edges are coarser than U-Net's.
- **Bottleneck** — the smallest, deepest feature map at the bottom of the U. Without skips, this is *all* the decoder has to work with.

```python
# Pattern from this notebook's U-Net decoder, level 1
u1 = layers.UpSampling2D()(d2)             # (128, 128, 64) upsampled
u1 = layers.Concatenate()([u1, c1])        # ← skip; (128, 128, 64+64=128)
d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(d1)
```

**Gotcha.** The encoder and decoder feature maps must have the *same `(H, W)`* at each skip — off-by-one errors here are the #1 cause of "shape mismatch" exceptions when building U-Nets on non-power-of-2 input sizes. Pad or crop to make them match before the `Concatenate`.

### Upsampling (nearest / bilinear) vs transposed convolution

> **🪜 Mental model:** *Stretch the picture back up.* Either by simple interpolation (fixed, no parameters) or by a learnable "reverse-conv" that can choose what to write into the new pixels.

**What it is.** Both operations *increase* the spatial size of a feature map — the inverse of pooling. **Upsampling** (`tf.keras.layers.UpSampling2D`) is a pure interpolation: it has *no learnable parameters*. With `interpolation='nearest'`, each output pixel just copies its nearest input pixel; with `'bilinear'`, it's a weighted average of the four nearest input pixels. **Transposed convolution** (`tf.keras.layers.Conv2DTranspose`, sometimes called "deconvolution" — a misnomer) is a *learnable* upsampling: it applies a learnable kernel that scatters each input value into a larger output region according to the trained weights.

**Why it matters.** The decoder of any segmentation network has to undo the encoder's downsampling. The choice of *how* matters: learnable upsampling can in theory produce better outputs (the network learns what details to add back in), but in practice transposed convolutions often produce visible **checkerboard artefacts** — uneven contributions across output pixels when `kernel_size` isn't a multiple of `stride`. Fixed interpolation followed by a standard `Conv2D` gives the network just as much expressivity (the `Conv2D` is learnable) without the artefacts.

**How it works.**
- **`UpSampling2D(size=2, interpolation='nearest')`** — each input pixel is replicated into a 2×2 block. Zero parameters, no learning. Followed by a `Conv2D(filters, 3)` if you want the network to refine the upsampled feature map.
- **`UpSampling2D(size=2, interpolation='bilinear')`** — same shape change but the new pixels are smooth interpolations of the four nearest neighbours.
- **`Conv2DTranspose(filters, kernel=4, strides=2, padding='same')`** — applies a learnable 4×4 kernel; each input element scatters into a 4×4 patch of the output, with strides controlling the spacing. The patches overlap, and at every output position the contributions are summed — which is where the checkerboard pattern comes from when `kernel % stride ≠ 0`.

**Where it's used in this notebook.**
- **FCN-8** uses `Conv2DTranspose(num_classes, kernel=4, strides=2)` and `Conv2DTranspose(num_classes, kernel=16, strides=8)` in its decoder — because the original 2015 FCN paper used these layers.
- **U-Net** in this notebook uses `UpSampling2D(size=2)` followed by `Conv2D(filters, 3)` at every decoder block — the modern best-practice choice, free of checkerboard artefacts.

**Related terms.**
- **Checkerboard artefact** — uneven output values that appear in transposed-conv outputs when `kernel_size % stride ≠ 0`; visible as a grid pattern in generated/upsampled images.
- **Pixel shuffle (sub-pixel conv)** — a third alternative: rearrange channel values into spatial positions. Used in super-resolution networks (ESPCN).
- **Dilated/atrous convolution** — a *different* way to expand receptive field without losing resolution; the heart of DeepLab.
- **Nearest vs bilinear** — nearest is cheaper and preserves hard edges; bilinear is smoother but can blur boundaries.

```python
# This notebook's U-Net decoder (cheap, no artefacts)
u1 = layers.UpSampling2D(size=2)(d2)
u1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)

# This notebook's FCN-8 decoder (learnable, but watch for checkerboard)
x  = layers.Conv2DTranspose(num_classes, 4, strides=2, padding='same')(x)
```

**Gotcha.** If you must use `Conv2DTranspose`, pick `kernel_size = 2 * strides` (e.g., `kernel=4, stride=2`) — it dramatically reduces the checkerboard artefact. The notebook's FCN follows this rule.

### Dice loss — translating the formula (full template)

> **🪜 Mental model:** *F1 score for masks.* Two masks overlap perfectly → Dice = 1. No overlap → Dice = 0. As a *loss*, we minimise `1 − Dice`.

**What it is.** **Dice loss** is a segmentation-specific loss based on the **Dice coefficient** (a.k.a. F1 for sets):

`Dice(predicted_mask, ground_truth_mask) = 2 · |intersection| / (|predicted_mask| + |ground_truth_mask|)`

In words, *twice the number of pixels both masks agree are foreground, divided by the total foreground pixels in either mask.* Translating each symbol:
- `|intersection|` = number of pixels that are foreground in **both** masks (true positives).
- `|predicted_mask|` = number of pixels the model predicted as foreground.
- `|ground_truth_mask|` = number of pixels actually foreground in the label.
- The factor of `2` is what makes Dice equivalent to F1 (`2·precision·recall / (precision+recall)`).

The loss is `1 − Dice`. With a smoothing constant to avoid division by zero on empty masks:
```python
dice = (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
dice_loss = 1 - dice
```

**Why it matters (the BCE-failure story).** Per-pixel **binary cross-entropy** (BCE) treats every pixel independently and averages over the whole image. On a portrait mask, roughly **70% of pixels are background**. The model can score ~95% accuracy and very low BCE simply by predicting "background everywhere" — the foreground pixels' contribution is drowned out in the mean. Dice doesn't average — it measures the *overlap of the foreground region as a whole*. A "predict background everywhere" model has zero foreground pixels, so the numerator (`2 · |intersection|`) is zero, and Dice is zero. **BCE rewards the easy majority; Dice rewards getting the rare minority right.**

**How it works (one training step).**
1. Forward pass produces `p` = predicted per-pixel foreground probability, shape `(H, W)`. Values in `[0, 1]`.
2. Ground truth `y` is `0/1` per pixel.
3. Compute `intersection = sum(p * y)` — soft intersection (sum of foreground-probability mass that overlaps with true foreground).
4. Compute `dice = (2*intersection + smooth) / (sum(p) + sum(y) + smooth)`. The smooth (`= 1` is standard) prevents `0/0` when both masks are empty.
5. Loss = `1 − dice`. Backprop.

The gradient encourages the predicted-mask region to *match the ground-truth region globally* — not pixel-by-pixel.

**Where it's used.** Medical imaging (organ, tumour, lesion segmentation — all heavily imbalanced). Portrait masks (this notebook). Any binary mask where foreground is a small fraction of the image. **Combined with BCE** (`loss = BCE + Dice`) is the de-facto production combo — BCE gives stable early-training gradients, Dice fine-tunes the boundary.

**Related terms.**
- **Dice coefficient** — the metric (1 = perfect, 0 = none).
- **IoU / Jaccard loss** — sibling overlap loss; `IoU = TP / (TP + FP + FN)`. Smaller gradient near the optimum than Dice.
- **Focal loss** — alternative imbalance fix; down-weights easy examples.
- **BCE + Dice combo** — the practical default.

```python
def dice_coefficient(y_true, y_pred, smooth=1):
    y_t = tf.keras.backend.flatten(y_true)
    y_p = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(y_t * y_p)
    return (2 * inter + smooth) / (
        tf.keras.backend.sum(y_t) + tf.keras.backend.sum(y_p) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
```

**Gotcha.** Pure Dice has *unstable* gradients early in training (when predictions are near-random, the soft intersection is tiny and the gradient explodes). That's why production uses BCE + Dice combined — BCE provides the stable warm-up signal.

### IoU for segmentation (mask-vs-mask Intersection over Union)

> **🪜 Mental model:** *Same IoU as detection, but on pixel sets instead of bbox corners.* The set you compare are the foreground pixels of each mask.

**What it is.** **Intersection over Union (IoU)**, also called the **Jaccard index**, is the ratio `|A ∩ B| / |A ∪ B|` — the size of the overlap between two sets divided by the size of their union. For segmentation, the "sets" are the foreground pixels of the predicted mask and the ground-truth mask. IoU equals 1 when the two masks are identical and 0 when they don't overlap at all. **Mean IoU (mIoU)** is the per-class IoU averaged across all classes — the canonical leaderboard metric on PASCAL VOC, Cityscapes, and ADE20K.

**Why it matters.** Pixel accuracy is misleading on imbalanced masks — a model that predicts "background everywhere" scores 95%+ on portrait data while finding zero person pixels. IoU doesn't have this loophole: a "background-everywhere" predictor has zero intersection with any non-empty ground-truth foreground, so IoU is 0. IoU also penalises both over- and under-segmentation symmetrically — predicting too many person pixels grows the union (and the false positive count); predicting too few shrinks the intersection. **For any imbalanced segmentation problem, mIoU is the metric you report.**

**How it works.** Translating the formula into pixel-counting words:
- `|A ∩ B|` = **True Positives (TP)** = pixels predicted as foreground that are actually foreground.
- `|A ∪ B|` = TP + FP + FN, where FP = pixels predicted foreground but actually background, FN = pixels actually foreground but predicted background.
- So `IoU = TP / (TP + FP + FN)`.
- For multi-class, compute IoU for each class separately, then average → mIoU.

The relationship to Dice (this notebook's training metric): `Dice = 2·IoU / (1 + IoU)`. Dice is always ≥ IoU; both are 1 at perfect overlap and 0 at no overlap.

**Where it's used.**
- **Training metric** alongside Dice (the notebook reports Dice; many production pipelines report both).
- **Detection** — IoU between predicted and ground-truth bounding boxes is how detections are scored as "match" or "no match" at a given threshold (e.g., IoU ≥ 0.5).
- **Non-Maximum Suppression (NMS)** in detection uses IoU to merge overlapping bounding boxes.
- **Public benchmarks** — PASCAL VOC (mIoU), Cityscapes (mIoU), ADE20K (mIoU).

**Related terms.**
- **Dice coefficient** — closely related metric: `2·TP / (2·TP + FP + FN)`. Equivalent ordering to IoU but with different scale.
- **Jaccard index** — exact synonym for IoU; sometimes used in older statistics literature.
- **Mean IoU (mIoU)** — per-class IoU averaged across classes.
- **Pixel accuracy** — the trap metric IoU exists to replace.
- **Box IoU** — same formula applied to detection bounding boxes (rectangles, not pixel sets).

```python
# Vectorised IoU for binary masks
def iou(y_true, y_pred, smooth=1e-6):
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_t * y_p)
    union = tf.reduce_sum(y_t) + tf.reduce_sum(y_p) - inter
    return (inter + smooth) / (union + smooth)
```

**Gotcha.** Class index `0` is usually the background, and including it in mIoU inflates the score because background is huge and easy. Some benchmarks exclude background; always check the convention before reporting.

### Mask R-CNN — instance segmentation in one model

> **🪜 Mental model:** *Faster R-CNN plus a mask painter.* Detect each object with a bounding box, then paint a per-object mask inside that box.

**What it is.** **Mask R-CNN** (He et al., 2017) is the canonical *instance segmentation* model. It extends Faster R-CNN (a two-stage detector that produces class-labelled bounding boxes) by adding a small **mask head** — a tiny fully-convolutional network (FCN) that, for each region-of-interest (ROI) the detector outputs, produces a binary 28×28 mask of the object inside that box. So one forward pass yields, for every detected object: a bounding box, a class label, a confidence score, **and** a per-instance mask. Importantly, it also replaces Faster R-CNN's `RoIPool` with **RoIAlign** — a bilinear-interpolation crop that avoids the quantisation errors that would otherwise blur fine mask boundaries.

**Why it matters.** Semantic segmentation (this notebook's U-Net) cannot count or separate instances — all "person" pixels share one label, you can't tell person #1 from person #2. Mask R-CNN solves this in a single end-to-end model: detection + per-object masks in one pass. It's the standard baseline for any task that needs to *count, track, or per-object-process* individuals (cell biology, retail shelf audits, AR object pinning, robotic grasping). The notebook only mentions it because the portrait dataset has one person per image, so semantic is sufficient — but interviewers expect candidates to know when to reach for it.

**How it works.**
1. **Backbone** (e.g., ResNet-50 + FPN) extracts multi-scale feature maps from the input image.
2. **Region Proposal Network (RPN)** scans the feature maps and proposes ~1,000 candidate object boxes.
3. **RoIAlign** crops a fixed-size feature map (e.g., `7×7×256`) from each proposal, using bilinear interpolation so no spatial information is rounded away.
4. **Detection head** classifies each proposal (`K+1` classes including background) and refines the box coordinates.
5. **Mask head** — a small FCN (4 convs + transposed conv) produces a `28×28` binary mask *per class*; at inference, select the mask for the predicted class.
6. **Loss** = classification loss + bbox regression loss + per-pixel BCE on the mask head (only for the ground-truth class).

**Where it's used.**
- **Cell biology** — count and segment individual cells in microscopy.
- **Retail / inventory** — count individual products on a shelf.
- **Autonomous driving** — separate individual cars, pedestrians, cyclists.
- **AR effects** — segment each face / hand / object separately for per-object processing.

**Related terms.**
- **Faster R-CNN** — the detector Mask R-CNN extends; same architecture minus the mask head.
- **RoI Align vs RoI Pool** — RoIAlign uses bilinear interpolation (preserves sub-pixel detail); RoIPool quantises (loses detail). Critical for mask quality.
- **FPN (Feature Pyramid Network)** — the multi-scale backbone Mask R-CNN typically uses.
- **U-Net** — the *semantic* counterpart; one mask per class, no instance distinction.
- **YOLACT / SOLOv2** — one-stage instance-segmentation alternatives; faster but slightly less accurate.

```python
# torchvision provides a pretrained Mask R-CNN — usable in a single line
from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(weights='DEFAULT').eval()
output = model([img_tensor])   # list of dicts: 'boxes', 'labels', 'scores', 'masks'
```

**Gotcha.** Mask R-CNN's masks are produced at low resolution (28×28 per ROI) and upsampled into the box at display time — fine for human-eye inspection but coarse for sub-pixel applications. Don't expect U-Net-level boundary precision from a vanilla Mask R-CNN.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset prep — masks to one-hot
```python
images = np.load('img_uint8.npy')         # (18698, 128, 128, 3)
masks  = np.load('msk_uint8.npy')         # (18698, 128, 128, 1) 0/255
masks[masks > 0] = 1                       # binarize
Y = tf.keras.utils.to_categorical(masks)  # (18698, 128, 128, 2) one-hot
```
Plain English: masks are loaded as 0/255 grayscale; we squash them to binary 0/1, then one-hot encode so each pixel has a `(2,)` distribution over `[background, person]`. This matches the U-Net's final `softmax` output shape `(128, 128, 2)`.

### 2. Dice coefficient metric (the segmentation metric)
```python
def dice_coefficient(y_true, y_pred, smooth=1):
    y_t = tf.keras.backend.flatten(y_true)
    y_p = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(y_t * y_p)
    return (2 * inter + smooth) / (
        tf.keras.backend.sum(y_t) + tf.keras.backend.sum(y_p) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
```
Both tensors are flattened before computing the intersection so the calculation works uniformly across batch, height, and width. `smooth=1` avoids division-by-zero on empty masks.

### 3. FCN-8 (Fully Convolutional Network)
```python
# VGG-16 encoder (frozen)
vgg = tf.keras.applications.VGG16(
    weights='imagenet', include_top=False,
    input_tensor=tf.keras.Input(shape=(128,128,3)),
)
vgg.trainable = False

# Decoder: 4× transpose to add Pool4, 4×4 to add Pool3, 8× final
x = layers.Conv2D(4096, 7, padding='same')(vgg.output)
x = layers.Conv2D(4096, 1, padding='same')(x)
x = layers.Conv2DTranspose(num_classes, 4, strides=2, padding='same')(x)
x = layers.Add()([x, pool4_features])    # skip add (NOT concat in FCN)
x = layers.Conv2DTranspose(num_classes, 4, strides=2, padding='same')(x)
x = layers.Add()([x, pool3_features])
out = layers.Conv2DTranspose(num_classes, 16, strides=8, padding='same',
                              activation='softmax')(x)
fcn8 = Model(vgg.input, out)
```
**Result:** Strong Dice but blurry boundaries — skips happen too late (only Pool3 and Pool4) and use `Add` (which doesn't preserve the encoder feature map's full information the way `Concatenate` does).

### 4. U-Net (the better architecture for portraits)
```python
def unet(input_shape=(128,128,3), num_classes=2):
    inp = layers.Input(input_shape)
    
    # ── Encoder (downsample, double channels per level) ──
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D()(c1)                    # 64×64
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D()(c2)                    # 32×32
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D()(c3)                    # 16×16
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    p4 = layers.MaxPooling2D()(c4)                    # 8×8
    
    # ── Bottleneck ──
    b = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
    b = layers.Conv2D(1024, 3, padding='same', activation='relu')(b)
    
    # ── Decoder (upsample, concatenate skip, halve channels) ──
    u4 = layers.UpSampling2D()(b)                                   # 16×16
    u4 = layers.Concatenate()([u4, c4])                              # skip!
    d4 = layers.Conv2D(512, 3, padding='same', activation='relu')(u4)
    d4 = layers.Conv2D(512, 3, padding='same', activation='relu')(d4)
    
    u3 = layers.UpSampling2D()(d4); u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv2D(256, 3, padding='same', activation='relu')(u3)
    d3 = layers.Conv2D(256, 3, padding='same', activation='relu')(d3)
    
    u2 = layers.UpSampling2D()(d3); u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(128, 3, padding='same', activation='relu')(u2)
    d2 = layers.Conv2D(128, 3, padding='same', activation='relu')(d2)
    
    u1 = layers.UpSampling2D()(d2); u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(d1)
    
    out = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    return Model(inp, out)
```
Notice the symmetric `c1↔u1, c2↔u2, …` skips and the doubling-then-halving channel count. The final `1×1` Conv is the per-pixel classifier head.

### 5. Training
```python
model = unet()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[dice_coefficient])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=16)
```
The loss is CE here, but the *reported metric* is Dice — a common pattern. (Production-grade: combine `BCE + Dice` into a single loss.)

### 6. Portrait blur effect — putting the mask to use
```python
preds = model.predict(X_test[:5])         # (5, 128, 128, 2) softmax
masks = np.argmax(preds, axis=-1)         # (5, 128, 128) class indices

for i in range(5):
    img       = X_test[i]
    blurred   = cv2.GaussianBlur(img, (21, 21), 0)
    portrait  = np.where(masks[i][..., None] == 1, img, blurred)
    plt.imshow(portrait); plt.show()
```
The `argmax` converts soft probabilities to hard class indices. `np.where(mask, img, blurred)` is a vectorised "if person pixel keep, else use blurred" — exactly the portrait-mode effect.

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Concept | Implementation |
|---|---|
| `layers.Conv2DTranspose(filters, kernel, strides, padding)` | Learnable upsampling |
| `layers.UpSampling2D(size=2, interpolation='nearest'|'bilinear')` | Fixed upsampling (no params) |
| `layers.Concatenate()` | Stack feature maps along channel axis (skip connections) |
| `layers.Add()` | Element-wise add (FCN-style skips, ResNet residual) |
| Dice coefficient | `2|A∩B| / (|A| + |B|)` |
| Dice loss | `1 - dice_coefficient` |

### Production-grade segmentation
| Call | Purpose |
|---|---|
| `segmentation_models_pytorch.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=2)` | U-Net with pretrained encoder |
| `torchvision.models.segmentation.fcn_resnet50(pretrained=True)` | FCN |
| `torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)` | DeepLab v3 |
| `torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)` | Mask R-CNN (instance segmentation) |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **`Conv2DTranspose` produces checkerboard artifacts** when `kernel_size % strides ≠ 0`. U-Net in this notebook avoids them by using `UpSampling2D + Conv2D` instead.
2. **Pixel accuracy is misleading on imbalanced masks.** A "predict background everywhere" model scores 95%+ on portraits where only ~30% of pixels are person. Use **mean IoU** or **Dice** as the metric.
3. **Dice loss + BCE/CE is the standard combo.** Pure Dice has unstable gradients early in training; pure CE gets dominated by majority class. The two together are the de-facto pattern for medical/portrait segmentation.
4. **Class index 0 is background in most pretrained segmentation models.** Verify before computing IoU.
5. **Mask R-CNN outputs masks at low resolution** (28×28 per ROI). You must upsample them to bounding-box size at display time.
6. **`to_categorical(masks)` requires integer-valued masks** in `[0, num_classes)`. If your masks are 0/255 binary, divide by 255 first.
7. **Concatenation doubles channel count at every skip.** A deep U-Net at high resolution explodes VRAM because every encoder feature map is held in memory until its matching decoder block runs.
8. **Encoder/decoder shape mismatch on non-power-of-2 inputs.** If input is 130×130, four pools yield 8×8 (with rounding), but four upsamples yield 128×128 — `Concatenate` will fail. Pad inputs to a multiple of `2^depth`.

[🔝 Back to top](#top)

## 🎯 Notebook quiz cells

**Q1.** Why is Dice loss better than cross-entropy for segmentation? *(adapted from `andrewekhalel/MLQuestions`, segmentation section)*
→ Cross-entropy treats pixels independently and gets dominated by the majority class. Dice measures *global mask overlap* → robust to imbalance. A "predict background everywhere" model scores ~95% pixel accuracy but Dice = 0.

**Q2.** Transposed convolution vs UpSampling2D — when which?
→ Transposed conv has learnable parameters → can adapt, but checkerboard artefacts if `kernel_size % stride ≠ 0`. UpSampling2D is fixed (cheap, no artefacts). This notebook's U-Net uses UpSampling for efficiency; FCN uses Conv2DTranspose because that's what the original FCN paper specified.

**Q3.** How do skip connections improve segmentation? *(adapted from `chiphuyen/ml-interviews-book`, vision chapter)*
→ Encoder destroys spatial detail (4 max-pools × 2× = 16× spatial compression). Skip connections inject fine-grained spatial info from the encoder directly into the decoder at matching resolutions → sharp edges. Without skips, the decoder paints from a blurry thumbnail.

**Q4.** Mask R-CNN vs U-Net? *(common FAANG CV question)*
→ Mask R-CNN is **instance-aware** (RPN finds objects, then segments each one separately). U-Net is **semantic** only (one mask, no instance distinction). For "count and segment each car separately" → Mask R-CNN. For "find the road" → U-Net.

**Q5.** Why does FCN produce coarser edges than U-Net?
→ FCN's decoder has only a few upsampling stages with late skip *additions* at Pool3 and Pool4 → information lost by the time we upsample. U-Net's symmetric structure with skip *concatenation* at every level preserves much more detail; the decoder always sees a fresh copy of the encoder's same-resolution features.

**Q6.** What's the difference between box-IoU and mask-IoU? *(adapted from `alexeygrigorev/data-science-interviews`)*
→ Box-IoU compares two axis-aligned rectangles via their overlap area / union area. Mask-IoU compares two pixel sets — `|pixels in both masks| / |pixels in either mask|`. Mask-IoU is strictly tighter (rewards exact outline, not just rough location).

**Q7.** Why one-hot encode the masks for this notebook?
→ The U-Net's final layer is `softmax` over `num_classes` channels — it outputs a `(H, W, 2)` per-pixel distribution. `categorical_crossentropy` expects the labels in the same shape, hence `to_categorical`.

**Q8.** When would you choose DeepLab over U-Net? *(adapted from `chiphuyen/ml-interviews-book`)*
→ When you need to keep spatial resolution high without explosively increasing parameter count. DeepLab uses **atrous (dilated) convolutions** to expand the receptive field without pooling — better for very-high-res inputs where U-Net's memory blows up.

**Q9.** Classification, detection, segmentation — pick the right one for "blur everything except the face in a portrait photo." *(adapted from `chiphuyen/ml-interviews-book`, vision chapter)*
→ Semantic segmentation. Classification gives one label per image (useless for masking). Detection gives a bounding box (would blur the shoulders / hair edges along with the background). Only per-pixel labelling lets the blur respect the actual person outline. This is exactly this notebook's task.

**Q10.** Semantic vs instance segmentation — when does the distinction matter in practice? *(adapted from `alexeygrigorev/data-science-interviews`, vision section)*
→ The distinction matters whenever you need to **count, track, or per-object-process** individuals of the same class. Counting cells in a microscope image → instance (semantic would give one big blob). Detecting "is there a road in this image?" → semantic. Self-driving car tracking each pedestrian individually → instance. Portrait blur with one person per image → semantic suffices (this notebook).

[🔝 Back to top](#top)

## 🪞 Extra ladder — segmentation evaluation

**Basic** — pixel accuracy: `(correct pixels) / (total pixels)`. Misleading on imbalance.

**Intermediate** — **mean IoU** across classes: `mean_c [ IoU_c ]`. The standard PASCAL VOC metric.

**Advanced** — **boundary metrics** like the Boundary F-measure for edge-quality evaluation. Or **panoptic quality** (PQ) for panoptic segmentation: combines instance recognition with semantic accuracy.

[🔝 Back to top](#top)

## What comes next

This notebook covered semantic + instance segmentation. [Notebook 8 →](../8.Siamese%20network/) shifts gears entirely: **metric learning** with Siamese networks for **one-shot verification** (faces, signatures).

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
