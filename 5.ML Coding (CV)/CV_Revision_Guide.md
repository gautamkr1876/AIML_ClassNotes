<a id="top"></a>
# Computer Vision — Master Revision Guide

> **Consolidated, interview-ready revision notes for all 9 CV notebooks** (CNN fundamentals → Tackling overfitting → Transfer learning → Image similarity & embeddings → Object detection (2-stage + 1-stage) → Segmentation → Siamese networks → GANs). Every concept, every API, every gotcha — in scannable form, with mental models, basic→advanced ladders, real sourced interview questions, and a 100-question revision drill at the end.

**Companion guides (Data Foundation, for prerequisites):**
- 🐍 [`Data_Foundation_Revision_Guide.md`](../Data%20Foundation/Data_Foundation_Revision_Guide.md) — NumPy
- 🐼 [`Pandas_Revision_Guide.md`](../Data%20Foundation/Pandas_Revision_Guide.md) — Pandas
- 📊 [`Amazon_Sachin_EDA_Revision_Guide.md`](../Data%20Foundation/Amazon_Sachin_EDA_Revision_Guide.md) — applied pandas + probability

**How to use:**
- **Pre-interview:** read the [🚀 Topic finder](#topic-finder) → skim a module's cheat sheet → drill the Q&A.
- **Just before a coding round:** run the [§14 Drill](#14-drill).
- **For per-notebook depth:** see each `CV_*_Interview_Prep_Guide.md` inside the corresponding lecture folder.

**External practice (use after you've drilled this guide):**
- 🎯 [`andrewekhalel/MLQuestions`](https://github.com/andrewekhalel/MLQuestions) — CV-heavy.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — DL/CV theory with answers.
- 🎯 [`chiphuyen/ml-interviews-book`](https://huyenchip.com/ml-interviews-book/) — DL chapter + system design.
- 🎯 [`Sroy20/machine-learning-interview-questions`](https://github.com/Sroy20/machine-learning-interview-questions) — DL question pool.
- 🎯 **Papers With Code** — for state-of-the-art on each task (detection, segmentation, GAN).

---

<a id="topic-finder"></a>
## 🚀 Topic finder

| Need to revise… | Go to |
|---|---|
| Image as tensor, channels, MLP vs CNN, locality/stationarity/compositionality | [Module 1](#1-module1) |
| Conv2D math, padding (same/valid), stride, output-shape formula, pooling | [Module 1](#1-module1) |
| Overfitting symptoms, train/val curves, dropout, BatchNorm, weight decay (L1/L2) | [Module 2](#2-module2) |
| Augmentation pipeline, early stopping, LR scheduling, GlobalAveragePooling | [Module 2](#2-module2) |
| Pretrained models, feature extraction vs fine-tuning, freezing, classifier head | [Module 3](#3-module3) |
| ResNet / VGG / Inception architectures, ImageNet, top-k accuracy | [Module 3](#3-module3) |
| Embeddings, penultimate-layer features, cosine vs L2, normalization | [Module 4](#4-module4) |
| NN search (brute / Annoy / FAISS), PCA, t-SNE, reverse image search | [Module 4](#4-module4) |
| Classification vs localization vs detection, bbox formats, IoU, NMS | [Module 5](#5-module5) |
| Two-stage detection: R-CNN → Fast R-CNN → Faster R-CNN, RPN, ROI pooling | [Module 5](#5-module5) |
| Single-stage detection: YOLO/SSD/RetinaNet, grid + anchors, focal loss | [Module 6](#6-module6) |
| Real-time CV, speed-vs-accuracy trade-off, video inference | [Module 6](#6-module6) |
| Semantic vs instance vs panoptic segmentation, encoder-decoder, U-Net | [Module 7](#7-module7) |
| Transposed conv vs UpSampling, Dice loss, IoU metric, Mask R-CNN | [Module 7](#7-module7) |
| Siamese architecture, contrastive vs triplet loss, hard-negative mining | [Module 8](#8-module8) |
| One-shot / verification, embedding metric learning, signature use case | [Module 8](#8-module8) |
| Generator / Discriminator, minimax game, BCE loss, DCGAN guidelines | [Module 9](#9-module9) |
| Mode collapse, FID/IS, training instability, GAN variants | [Module 9](#9-module9) |
| All terms at once | [§10 Glossary](#10-terms) |
| Every API at once | [§11 API cheat sheet](#11-apis) |
| Common gotchas | [§12 Gotchas](#12-gotchas) |
| Advanced cross-module Q&A | [§13 Advanced Q&A](#13-advanced) |
| 🌐 Sourced interview questions (real, paraphrased) | [Sourced bank](#sourced-bank) |
| Speed-run revision drill | [§14 Drill](#14-drill) |
| Best practices | [§15 Best practices](#15-bestpractices) |
| Notebook mapping | [§16 Mapping](#16-mapping) |

---

## 📑 Table of contents

1. [Module 1 — Intro to CV & CNN Fundamentals](#1-module1)
2. [Module 2 — Tackling Overfitting in CNNs](#2-module2)
3. [Module 3 — Transfer Learning](#3-module3)
4. [Module 4 — Image Similarity & Embeddings](#4-module4)
5. [Module 5 — Object Detection: Two-Stage](#5-module5)
6. [Module 6 — Object Detection: Single-Stage](#6-module6)
7. [Module 7 — Object Segmentation](#7-module7)
8. [Module 8 — Siamese Networks](#8-module8)
9. [Module 9 — GANs for Image Generation](#9-module9)
10. [📚 Terms glossary](#10-terms)
11. [⚙️ API cheat sheet](#11-apis)
12. [⚠️ Gotchas & traps](#12-gotchas)
13. [🎯 Advanced interview Q&A](#13-advanced)
14. [🌐 Sourced interview questions](#sourced-bank)
15. [🔁 100-question revision drill](#14-drill)
16. [✅ Best practices](#15-bestpractices)
17. [📦 Notebook mapping](#16-mapping)

---

<a id="1-module1"></a>
## 1. Module 1 — Intro to CV & CNN Fundamentals

> Notebook 1 — CV motivation, image as tensor, why MLPs fail on images, CNN building blocks (Conv2D, padding, stride, pooling, activation), forward pass math, training loop on the Clothing-Small dataset. Achieved **50% test acc with a single Conv layer vs 36% with a 50M-param MLP** — the canonical "specialization beats brute force" lesson.

### 🪜 Mental model

**A CNN slides a small set of learned filters over the image.** The filters share weights across positions (you don't learn "find an edge at pixel (12,34)" — you learn "find an edge" and apply it everywhere). This single design choice gives you three superpowers at once:
- **Locality** — pixels near each other are correlated; convolve over neighborhoods.
- **Stationarity** — the same pattern can appear anywhere; reuse one filter across positions (parameter sharing).
- **Compositionality** — stack layers; early ones find edges, deeper ones combine edges into parts and objects.

When you hear "CNN" think *sliding filter + share weights + stack*.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Image as tensor | `(H, W, C)` — height × width × channels (1 for grayscale, 3 for RGB) |
| Pixel range | 0–255; almost always rescale to `[0, 1]` (`Rescaling(1./255)`) |
| Why MLPs fail on images | Flattening destroys spatial structure; param count explodes (128×128×3 → 49,152 inputs) |
| Conv2D | Sliding learnable filter producing a feature map per filter |
| Kernel size | Odd (3×3, 5×5) so the center is a single pixel |
| Padding | `'same'` preserves output size with zeros around edges; `'valid'` shrinks output |
| Stride | Pixels the filter moves per step; larger stride → smaller output, fewer ops |
| Output-shape formula | `O = (N + 2P − F) / S + 1` |
| Pooling | No-learnable downsampling — `MaxPooling2D` (strongest activation) or `AveragePooling2D` |
| Activation | ReLU after every Conv (introduces non-linearity), softmax at output |
| Parameter count (Conv2D) | `(kH × kW × Cin + 1) × Cout` |

### 🪞 Basic → Intermediate → Advanced — Conv2D output shape

**Basic** — apply a single Conv2D to a 28×28 grayscale image with default padding.
```python
layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
# output: (26, 26, 16) — valid padding shrinks by (kernel - 1)
```

**Intermediate** — control output shape via padding and stride explicitly.
```python
layers.Conv2D(32, 3, strides=2, padding='same')      # 28×28 → 14×14×32
layers.Conv2D(32, 3, strides=1, padding='valid')     # 28×28 → 26×26×32
```

**Advanced** — predict the exact output without running the model. For `N=128, F=5, P=2, S=2`:
`O = (128 + 4 − 5) / 2 + 1 = 64.5 → 64` (integer division). Mismatches here are the #1 cause of silent shape bugs in deep architectures. Always verify with `model.summary()` before training.

### 🪞 Basic → Intermediate → Advanced — pooling

**Basic** — `MaxPooling2D` halves spatial dims with default `pool_size=2, strides=2`.
```python
layers.MaxPooling2D()           # (28,28,C) → (14,14,C)
```

**Intermediate** — pooling has **no learnable parameters** — it's a fixed downsampling. Use it to (1) reduce spatial dims, (2) introduce small translation invariance, (3) keep memory bounded.

**Advanced** — replacing pooling with **strided convolutions** is often better in modern architectures (ResNet, EfficientNet). Strided convs are *learnable* downsamplers — they can adapt to data, unlike fixed max/avg pool. Use `MaxPooling2D` when you want fewer parameters; use strided `Conv2D` when you want learning power.

### ⚙️ Top APIs

```python
# Loading + preprocessing
tf.keras.utils.image_dataset_from_directory(path, image_size=(128,128), batch_size=32)
layers.Resizing(128, 128); layers.Rescaling(1./255)

# CNN building blocks
layers.Conv2D(filters, kernel_size, strides=1, padding='same'|'valid', activation='relu')
layers.MaxPooling2D(pool_size=2, strides=2)
layers.AveragePooling2D()
layers.Flatten()
layers.Dense(units, activation='relu'|'softmax')

# Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

### 🧩 Code patterns

```python
# 1. Build a simple CNN from scratch
model = keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# 2. Preprocess on the fly
preprocess = keras.Sequential([layers.Resizing(128,128), layers.Rescaling(1./255)])
train_ds = train_raw.map(lambda x, y: (preprocess(x), y))

# 3. Output-shape math (always run this before training)
model.summary()

# 4. Inference + accuracy
y_pred = model.predict(test_ds)
preds = tf.argmax(y_pred, axis=1)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_true, preds)
```

### 🎯 Q&A — Module 1

> Mix of original drills + questions adapted from `alexeygrigorev/data-science-interviews` and `andrewekhalel/MLQuestions`.

1. **Why is a CNN better than an MLP for images?** *(common opener)* CNNs exploit locality (sparse connections), stationarity (weight sharing), and compositionality (stacking) — three things MLPs throw away when they flatten. Plus they have 100× fewer parameters.
2. **Output-shape formula?** `O = (N + 2P − F) / S + 1`. Drill this — it appears in almost every CV interview.
3. **Why use odd kernel sizes (3, 5, 7)?** Odd kernels have a single integer center pixel; even kernels center at `(0.5, 0.5)` and create off-by-half artifacts when stacked.
4. **Padding `'same'` vs `'valid'`?** `'same'` zero-pads to preserve output size. `'valid'` doesn't pad and the output shrinks by `(F − 1)` per layer.
5. **Does pooling have learnable parameters?** *(from `alexeygrigorev`)* **No** — it's a fixed reduction. That's part of why it's cheap.
6. **Why ReLU and not sigmoid for hidden layers?** *(from `alexeygrigorev`)* ReLU doesn't saturate for positive inputs → no vanishing gradient. Sigmoid's derivative is ≤ 0.25 and goes to 0 for |x| large → gradients die in deep nets.
7. **Translation invariance — what does the CNN actually guarantee?** *(common trap)* CNNs are *translation-equivariant* (a shifted input → a correspondingly shifted feature map). Full translation **invariance** comes from pooling + global pooling at the end.
8. **Image flattened to MLP — how many params for `(128,128,3) → Dense(1024)`?** `128 × 128 × 3 × 1024 + 1024 ≈ 50M`. This is why MLPs blow up on images.

[🔝 Back to top](#top)

---

<a id="2-module2"></a>
## 2. Module 2 — Tackling Overfitting in CNNs

> Notebook 2 — when the baseline CNN hits 99% train / 59% val (huge gap), apply the regularization toolkit: **dropout, BatchNorm, L2 weight decay, data augmentation, early stopping, LR scheduling, GlobalAveragePooling**. Result: 51% → **78% test accuracy** with the train/val gap closing from 40% to ~7%.

### 🪜 Mental model

**Overfitting = the model memorizes the training set.** Three classes of remedy, in order of impact:
1. **More data** (or augmentation = virtual data multiplication).
2. **Regularize the loss** (L1/L2 penalty, dropout = ensemble trick, BatchNorm = noise injection).
3. **Stop training before it gets bad** (early stopping, LR decay).

If train acc ≫ val acc → overfit → add regularization. If both are low → underfit → bigger model. The gap is your diagnostic signal.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Overfit symptom | Train ↑, val plateaus or drops |
| Underfit symptom | Both train and val low / not improving |
| Dropout | Randomly zeros activations during training only; default `rate=0.5` for dense, 0.1–0.25 for conv |
| Dropout intuition | Acts as an implicit ensemble of sub-networks |
| BatchNormalization | Normalize layer inputs per batch → faster, more stable training |
| BN at inference | Uses running averages computed over training |
| L2 (weight decay) | Adds `λ Σ w²` to loss → shrinks weights toward 0 (but not to 0) |
| L1 | Adds `λ Σ |w|` → drives weights *exactly* to 0 (feature selection) |
| Data augmentation | Random transforms (flip, rotate, crop, color jitter) applied train-time only |
| EarlyStopping | Stop when val loss hasn't improved for `patience` epochs |
| ReduceLROnPlateau | Multiply LR by `factor` when val loss stalls |
| GlobalAveragePooling2D | Replaces `Flatten + Dense` — averages each feature map to one value; massively cuts params |

### 🪞 Basic → Intermediate → Advanced — dropout

**Basic** — drop 50% of dense activations during training.
```python
layers.Dense(256, activation='relu'), layers.Dropout(0.5)
```

**Intermediate** — placement matters: typically **after** the activation, **before** the next dense layer. Use lighter rates (0.1–0.25) inside conv blocks.

**Advanced** — dropout is **disabled at inference** (the `training=False` mode rescales activations automatically). Forgetting this — e.g. by calling `model(x, training=True)` at eval — is a classic stealth bug that makes your model "regress" on day-2 evaluation.

### 🪞 Basic → Intermediate → Advanced — augmentation

**Basic** — random horizontal flip and rotation.
```python
augment = keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(0.1)])
```

**Intermediate** — chain augmentations into the input pipeline (train only).
```python
train_ds = train_ds.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
```

**Advanced** — augmentation must **not** be applied at val/test time (else metrics are noise). Modern practice: build augmentation as a `Sequential` model and wrap it in a layer that respects `training` mode. Also: heavy augmentation on tiny datasets can hurt (you're inventing pixels that don't exist).

### ⚙️ Top APIs

```python
# Regularization layers
layers.Dropout(rate)
layers.BatchNormalization(momentum=0.99)
regularizers.l2(weight)            # used as kernel_regularizer=

# Augmentation layers
layers.RandomFlip('horizontal'|'vertical'|'horizontal_and_vertical')
layers.RandomRotation(factor)
layers.RandomCrop(h, w)
layers.RandomBrightness(factor)
layers.RandomContrast(factor)
layers.RandomTranslation(h, w)

# Callbacks
keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5)

# Architecture trick to cut params
layers.GlobalAveragePooling2D()    # replaces Flatten + huge Dense
```

### 🧩 Code patterns

```python
# 1. Conv block with BN + Dropout
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    return x

# 2. Final classifier with dropout
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(num_classes, activation='softmax')(x)

# 3. Train with callbacks
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)

# 4. Augmentation pipeline applied only at train
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
val_ds   = val_ds  .map(lambda x, y: (augment(x, training=False), y))   # no-op
```

### 🎯 Q&A — Module 2

> Mix of original + questions adapted from `alexeygrigorev/data-science-interviews` and `Sroy20/machine-learning-interview-questions`.

1. **What is dropout?** *(from `alexeygrigorev`)* Stochastically zero a fraction of activations during training. Acts as implicit ensembling — every forward pass is a different sub-network.
2. **Is dropout active at inference?** **No.** Activations are rescaled by `1/(1−p)` so expected magnitude matches training.
3. **What does BatchNorm normalize?** *(from `alexeygrigorev`)* Per-batch, per-channel: it computes mean and variance across the batch (and spatial dims for conv layers), normalizes to zero mean / unit variance, then applies a learnable affine `γ·x + β`.
4. **BN at inference — what's different?** Uses **running** mean/variance computed during training (an EMA), not the current batch.
5. **Place BatchNorm where: before or after activation?** Convention varies; the original paper says before activation (`Conv → BN → ReLU`). Modern practice often does `Conv → ReLU → BN`. Both work; pick one and stick to it.
6. **L1 vs L2 regularization?** L1 = `λ Σ |w|` → produces **sparse** weights (feature selection). L2 = `λ Σ w²` → shrinks all weights smoothly. L2 is the default for CNNs.
7. **Why does augmentation help?** *(from `alexeygrigorev`)* It's free virtual data — the model sees variations of the same content, forcing it to learn invariance instead of memorizing pixels.
8. **Should you augment the validation set?** **No.** Augmentation distorts the eval signal. The val set must reflect deployment distribution.
9. **Why `GlobalAveragePooling2D` over `Flatten + Dense`?** Saves **massive** parameter count. Replacing a `Flatten → Dense(256)` after a `(16,16,256)` feature map cuts ~17M params to 65k.

[🔝 Back to top](#top)

---

<a id="3-module3"></a>
## 3. Module 3 — Transfer Learning

> Notebook 3 — recognize 10 famous landmarks with only **737 training images**. From-scratch VGG16 → 12% test acc; VGG16 with ImageNet weights + frozen conv + new classifier head → **79% test acc** in 5 epochs. The canonical "reuse what's already learned" lesson.

### 🪜 Mental model

**Low layers learn generic features, high layers learn task-specific ones.** Edges and textures in layer 1 generalize to *any* image dataset; "this looks like a wing of a 747" in layer 25 is ImageNet-specific. So the recipe is: **keep the generic stuff, retrain the specific stuff.** Two strategies:
- **Feature extraction** — freeze the whole pretrained backbone; train only a fresh classifier on top.
- **Fine-tuning** — unfreeze the last few layers and train them with a *small* LR alongside the new head.

Always start with feature extraction (faster, less risk of catastrophic forgetting); fine-tune later if you need more accuracy.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Transfer learning | Reuse a model trained on a large dataset (ImageNet) as the starting point |
| Pretrained backbone | The convolutional portion (no classifier head) |
| `include_top=False` | Drop the original 1000-class classifier |
| Feature extraction | Freeze backbone, train new classifier only |
| Fine-tuning | Unfreeze top few backbone layers; train with small LR |
| Freezing | `model.trainable = False` or per-layer `layer.trainable = False` |
| ImageNet input size | Typically `(224, 224, 3)` — match the pretrained model's expectation |
| Preprocessing | Each pretrained family has its own (`vgg16.preprocess_input`, etc.) |
| Top-k accuracy | Score 1 if true label is in the top-k predictions |
| AlexNet | 2012 ILSVRC winner, 60M params, started the DL revolution in CV |
| VGG16 / VGG19 | 138M / 144M params, 3×3 conv stack |
| ResNet | Residual connections enable very deep nets (50–152 layers) |
| Inception | Multi-scale conv blocks, parameter-efficient (~11M for v1) |
| EfficientNet / MobileNet | Lightweight backbones for mobile / edge |

### 🪞 Basic → Intermediate → Advanced — transfer learning

**Basic** — load a pretrained backbone with no classifier.
```python
base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

**Intermediate** — feature extraction: freeze everything, add a classifier.
```python
base.trainable = False
model = keras.Sequential([base, layers.Flatten(), layers.Dense(num_classes, activation='softmax')])
```

**Advanced** — fine-tune the top blocks after the head has trained. Use a **10× smaller LR** on the unfrozen layers to avoid destroying ImageNet features. Always run feature extraction *first*; only fine-tune *after* the head has converged.
```python
base.trainable = True
for layer in base.layers[:-4]: layer.trainable = False   # unfreeze last 4
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy')
```

### ⚙️ Top APIs

```python
# Keras applications (each has matching preprocess_input)
tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
tf.keras.applications.ResNet50(...)
tf.keras.applications.InceptionV3(...)
tf.keras.applications.EfficientNetB0(...)
tf.keras.applications.MobileNetV2(...)

# Preprocessing (model-specific)
from tf.keras.applications.vgg16 import preprocess_input

# Freezing
base.trainable = False
for layer in base.layers[:N]: layer.trainable = False
```

### 🧩 Code patterns

```python
# 1. Feature extraction pipeline
base = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                    input_shape=(224,224,3))
base.trainable = False
model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax'),
])

# 2. Model-specific preprocessing
inputs = keras.Input(shape=(224,224,3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)  # NOT 1/255
x = base(x, training=False)                                # BN stays in inference mode

# 3. Fine-tune top blocks (after head training)
base.trainable = True
for layer in base.layers[:-4]: layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy')
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

### 🎯 Q&A — Module 3

> Mix of original + questions adapted from `alexeygrigorev/data-science-interviews` and `andrewekhalel/MLQuestions`.

1. **What is transfer learning?** *(from `alexeygrigorev`)* Reusing the representations learned by a model trained on one large dataset as the starting point for a different but related task — works because low-level features (edges, textures) are universal.
2. **When does transfer learning fail?** When the source and target domains differ drastically (e.g., ImageNet → medical X-rays). The deeper you go, the more ImageNet-specific the features become.
3. **Why freeze the backbone?** *(common opener)* Three reasons: (1) preserve ImageNet features, (2) avoid overfitting when target dataset is small, (3) much faster training (no backprop through backbone).
4. **Feature extraction vs fine-tuning — order?** *(from `andrewekhalel`)* Always do feature extraction first (head converges fast); fine-tune top layers afterwards with 10× smaller LR.
5. **Why a smaller LR when fine-tuning?** Large LR + pretrained weights = catastrophic forgetting. Small LR (e.g. `1e-5`) lets the backbone adapt gently without erasing its prior.
6. **`include_top=False` does what?** Removes the original 1000-class classifier so you can attach your own head.
7. **Why `(224,224,3)` for VGG?** That's the size VGG was trained on; the pretrained weights expect this exact input. Resize your data to match (or use `EfficientNet` variants for different sizes).
8. **What's top-5 accuracy?** Score 1 if the true label is in the top 5 predictions. Useful when classes have ambiguous boundaries (a husky vs a malamute).

[🔝 Back to top](#top)

---

<a id="4-module4"></a>
## 4. Module 4 — Image Similarity & Embeddings

> Notebook 4 — reverse image search on Caltech-101. Use a pretrained **ResNet-50** (no top) to extract **2048-dim embeddings** from the penultimate layer; L2-normalize; nearest-neighbour search by L2 or cosine distance. Scale from brute-force (60 ms/query) to **Annoy** approximate-NN (~42 µs/query — a 1,400× speedup) with virtually no accuracy loss after PCA to 150 dims.

### 🪜 Mental model

**An embedding is a coordinate in feature space.** Similar images live near each other; dissimilar images live far apart. The pretrained CNN gave you a great coordinate system *for free* — the penultimate layer is rich with object-level semantics. Once you have coordinates, similarity is just **nearest neighbours**: a non-learning, geometric operation that's blazingly fast with the right data structure.

Cosine vs Euclidean: after **L2-normalization**, the two are monotonic-equivalent. Use cosine when magnitudes matter less (e.g., bag-of-words); use L2 when geometric distance has a physical meaning.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Embedding | Dense feature vector representing the input (image, word, etc.) |
| Penultimate layer | Layer just before the classifier — output is the embedding |
| Embedding dim | ResNet-50: 2048; VGG-16: 4096; MobileNet-v2: 1280 |
| L2 normalization | Divide by `‖x‖₂` to put on unit sphere; makes cosine and L2 equivalent |
| Cosine similarity | `cos(θ) = (x·y) / (‖x‖ ‖y‖)` — measures angle, ignores magnitude |
| Euclidean (L2) distance | `‖x − y‖₂` — straight-line distance in feature space |
| Brute-force NN | Compare query against every item; O(N·D) per query |
| Tree-based ANN | Annoy, KD-trees — sub-linear in N, small accuracy loss |
| Hash-based ANN | LSH — even faster but more accuracy loss |
| FAISS | Facebook's NN library — GPU-accelerated, billion-scale |
| PCA | Linear dim-reduction; preserves variance |
| t-SNE / UMAP | Non-linear; for *visualization only*, not retrieval |

### 🪞 Basic → Intermediate → Advanced — embeddings & NN search

**Basic** — extract a single embedding.
```python
backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')
emb = backbone.predict(preprocess(img))   # shape (1, 2048)
emb = emb / np.linalg.norm(emb)           # L2 normalize
```

**Intermediate** — batch-extract, then brute-force NN search.
```python
features = backbone.predict(image_ds)             # (N, 2048)
features = features / np.linalg.norm(features, axis=1, keepdims=True)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(features)
distances, indices = nn.kneighbors(query[None])
```

**Advanced** — production-scale with Annoy + PCA. Compress 2048 → 150 dims (accuracy drops <0.5%), build an Annoy index for sub-millisecond search.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=150).fit(features)
features_small = pca.transform(features)

from annoy import AnnoyIndex
idx = AnnoyIndex(150, metric='angular')      # angular = cosine-equivalent on normalized vectors
for i, v in enumerate(features_small): idx.add_item(i, v)
idx.build(40)                                # 40 trees
idx.get_nns_by_vector(query_pca, 5, include_distances=True)
```

### ⚙️ Top APIs

```python
# Embedding extraction
tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
tf.keras.applications.resnet50.preprocess_input

# NN search
sklearn.neighbors.NearestNeighbors(n_neighbors=, metric='euclidean'|'cosine')
annoy.AnnoyIndex(dim, 'angular'|'euclidean')
# faiss.IndexFlatL2(dim) / faiss.IndexIVFFlat(...)

# Dim reduction
sklearn.decomposition.PCA(n_components=)
sklearn.manifold.TSNE(n_components=2)
```

### 🧩 Code patterns

```python
# 1. Extract + normalize
def embed(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    e = model.predict(x).flatten()
    return e / np.linalg.norm(e)

# 2. Top-K NN with sklearn
nn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(features)
dists, idxs = nn.kneighbors(query[None])

# 3. PCA + Annoy production stack
pca = PCA(n_components=150).fit(features)
small = pca.transform(features)
idx = AnnoyIndex(150, 'angular')
for i, v in enumerate(small): idx.add_item(i, v)
idx.build(40)
idx.save('items.ann')

# 4. t-SNE for visualization
emb_2d = TSNE(n_components=2).fit_transform(features[:4000])
plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels[:4000], cmap='tab10')
```

### 🎯 Q&A — Module 4

> Mix of original + questions adapted from `andrewekhalel/MLQuestions` (CBIR / content-based image retrieval).

1. **What is an embedding?** A dense vector representation of an input where similar inputs land near each other in vector space.
2. **Why use penultimate-layer activations, not logits?** Logits are forced into class probabilities — they lose all the rich nuance. The penultimate layer carries `n`-dimensional semantics (texture, shape, parts).
3. **Cosine vs Euclidean — which when?** After L2 normalization both are equivalent (cosine = `1 − ‖x−y‖²/2`). Use cosine when magnitude is irrelevant (text, sparse features); use L2 when geometric distance matters.
4. **Why L2-normalize embeddings?** Makes the comparison scale-invariant and puts everything on the unit sphere, which stabilizes downstream metrics.
5. **Brute-force NN doesn't scale to 100M images — what's the production trick?** *(common system-design question)* Approximate nearest neighbours (Annoy, FAISS, ScaNN) — tree- or hash-based indices that trade ~0.1% recall for 1000–10000× speedup.
6. **What does PCA do to embeddings?** Linearly projects them onto the top-`k` principal components. Often you keep ~150 of 2048 dims with ≥95% variance — search becomes 13× faster with negligible accuracy loss.
7. **t-SNE for retrieval?** **No** — t-SNE is for *visualization* only. The 2D coordinates don't preserve neighborhood relations in a metric you can search.
8. **Why does a pretrained classifier give good embeddings even though it wasn't trained for similarity?** *(common interview question)* The classifier learned features that *separate* classes — and features that separate classes are exactly the features that cluster similar inputs.

[🔝 Back to top](#top)

---

<a id="5-module5"></a>
## 5. Module 5 — Object Detection: Two-Stage

> Notebook 5 — beyond "what's in the image?" to "**where** is it?" Bounding-box regression, IoU, NMS, anchor boxes, the R-CNN → Fast R-CNN → Faster R-CNN evolution, RPN, ROI pooling. Demonstrated on a gun-detection dataset (PistolData) with a ResNet101 backbone + dual heads (classification + bbox regression).

### 🪜 Mental model

**Two-stage detection = "propose, then classify."**
- **Stage 1 (Region Proposal Network, RPN):** "Where *might* something interesting be?" Output: a few thousand candidate boxes.
- **Stage 2 (Classifier head):** "What's in each candidate, and what's its exact bbox?"

Compare with single-stage methods (Module 6) which do both in one pass. Two-stage is **slower but more accurate** — used when latency isn't critical (medical, security, offline processing).

**Anchors** are a key trick: instead of regressing absolute bbox coordinates (hard), regress **offsets** from predefined anchor boxes (much easier — the anchor gives a starting guess).

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Classification | "What's in this image?" — one label |
| Localization | "What's in it AND where?" — one bbox |
| Detection | "What and where, for ALL objects" — many bboxes + labels |
| Bbox formats | `(x1, y1, x2, y2)` corners ↔ `(xc, yc, w, h)` center+size ↔ normalized vs pixels |
| IoU | `area(intersection) / area(union)` — overlap quality, in `[0, 1]` |
| NMS | Iteratively keep highest-confidence box, drop overlaps above IoU threshold |
| Anchor box | Predefined reference box; network predicts *offsets* from it |
| Anchor count (Faster R-CNN) | Typically 9 per location (3 scales × 3 ratios) |
| RPN | Learned region proposer (replaces Selective Search) |
| ROI Pooling | Resize variable-size proposals to a fixed grid (7×7) for the classifier |
| mAP | Mean Average Precision across IoU thresholds — the detection metric |
| Two-stage family | R-CNN → SPP-Net → Fast R-CNN → Faster R-CNN → Mask R-CNN |

### 🪞 Basic → Intermediate → Advanced — IoU

**Basic** — IoU between two boxes.
```python
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-8)
```

**Intermediate** — batch IoU between many predictions and many ground truths (broadcast-vectorized) — used inside NMS.

**Advanced** — generalize to **GIoU / DIoU / CIoU** losses for bbox regression. Plain IoU has zero gradient when boxes don't overlap (`intersection = 0`); GIoU adds a term that penalizes the enclosing box, so gradients exist everywhere. Modern detectors (YOLOv4+, RetinaNet) use these.

### 🪞 Basic → Intermediate → Advanced — NMS

**Basic** — keep the top-confidence box, discard overlaps above threshold.

**Intermediate** — per-class NMS: run NMS independently within each class, so a car and a pedestrian close together aren't merged.

**Advanced** — **Soft-NMS** decays the confidence of overlapping boxes instead of zeroing them, helping when objects overlap legitimately (a crowd of people). Even more advanced: **learned NMS** (a tiny MLP decides which boxes to keep) and **non-NMS detectors** (DETR uses bipartite matching instead).

### ⚙️ Top APIs

```python
# torchvision
torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
torchvision.ops.nms(boxes, scores, iou_threshold)
torchvision.ops.box_iou(boxes1, boxes2)
torchvision.ops.roi_pool / roi_align

# OpenCV utilities (during preprocessing/visualization)
cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.dnn.NMSBoxes(boxes, scores, score_thresh, iou_thresh)
```

### 🧩 Code patterns

```python
# 1. IoU
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-8)

# 2. NMS skeleton
def nms(boxes, scores, iou_thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        rest = [j for j in order[1:] if iou(boxes[i], boxes[j]) < iou_thr]
        order = np.array(rest)
    return keep

# 3. Dual-head model (classification + bbox)
backbone = ResNet101(weights='imagenet', include_top=False, input_tensor=Input((416,416,3)))
feat = layers.Flatten()(backbone.output)
cls  = layers.Dense(1, activation='sigmoid', name='class_output')(layers.Dense(32, 'relu')(feat))
bbox = layers.Dense(4, activation='sigmoid', name='box_output')  (layers.Dense(32, 'relu')(feat))
model = Model(backbone.input, [bbox, cls])
model.compile(loss={'box_output': 'mse', 'class_output': 'binary_crossentropy'},
              loss_weights={'box_output': 4.0, 'class_output': 1.0})

# 4. Faster R-CNN inference (PyTorch)
import torchvision; m = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
preds = m([img_tensor])             # list of dicts: boxes, labels, scores
keep = torchvision.ops.nms(preds[0]['boxes'], preds[0]['scores'], iou_threshold=0.5)
```

### 🎯 Q&A — Module 5

> Mix of original + questions adapted from `andrewekhalel/MLQuestions` and `alexeygrigorev/data-science-interviews`.

1. **Classification vs localization vs detection — define each.** Classification: label only. Localization: label + one bbox. Detection: many labels + many bboxes.
2. **What is IoU?** Overlap = intersection / union of two boxes. Standard threshold for "match" in detection is 0.5.
3. **Why do we need NMS?** *(from `andrewekhalel`)* Detectors fire many overlapping boxes for one object; NMS deduplicates by keeping the highest-confidence and dropping anything that overlaps it too much.
4. **What does an anchor box do?** Provides a reference shape. The network predicts *offsets* from it instead of absolute coordinates — easier optimization, faster convergence.
5. **What's an RPN, and what does it replace?** Region Proposal Network — a learnable mini-CNN that proposes regions of interest. Replaces hand-crafted Selective Search (used in original R-CNN), making the whole pipeline end-to-end trainable.
6. **R-CNN vs Fast R-CNN vs Faster R-CNN — one-line each.**
   - R-CNN: 2000 selective-search regions, each pushed through CNN separately → slow.
   - Fast R-CNN: run CNN once on whole image, share features across regions via ROI pooling.
   - Faster R-CNN: replace Selective Search with RPN → fully end-to-end.
7. **Why ROI Pooling?** Region proposals have different sizes; the classifier head needs a fixed-size input. ROI pooling resizes them to 7×7 (or similar) via region-aware max pooling.
8. **What metric do you use to evaluate a detector?** *(common interview question)* **mAP** — mean Average Precision, typically reported at IoU ≥ 0.5 (PASCAL VOC) or averaged over `[0.5, 0.55, …, 0.95]` (COCO).

[🔝 Back to top](#top)

---

<a id="6-module6"></a>
## 6. Module 6 — Object Detection: Single-Stage

> Notebook 6 — for **real-time** detection (self-driving cars, video, robotics), two-stage is too slow. **Single-stage detectors (YOLO, SSD, RetinaNet)** predict all boxes in one forward pass over the image. Trade-off: 2–3× faster, ~5% lower mAP than the best two-stage methods. Demonstrated YOLOv5 loaded via OpenCV DNN on driving data.

### 🪜 Mental model

**One network, one forward pass, all detections.** YOLO divides the image into a `S × S` grid; each cell predicts `(bbox, objectness, class probs)` for `B` anchors. No separate proposal step — the network learns to look at every position simultaneously.

Three things make single-stage hard:
1. **Class imbalance** — 99% of grid cells are background. Without a fix, the model just learns "predict 'no object'." Solution: **Focal Loss** (RetinaNet) — down-weight easy negatives.
2. **Multi-scale** — small objects need fine-grained feature maps; large objects need coarse ones. Solution: predict at multiple scales (YOLOv3 uses 3 scales).
3. **Anchor design** — too few anchors miss objects; too many slow inference. Solution: K-means cluster anchor shapes from your training data.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Single-stage detection | One forward pass → all bboxes + classes |
| YOLO | "You Only Look Once" — grid-based dense prediction |
| YOLO output shape | `S × S × B × (5 + C)` per scale (5 = x, y, w, h, objectness) |
| SSD | Single-Shot Detector — multi-scale dense prediction with VGG backbone |
| Anchor (in YOLO) | Predefined box shape per cell; network predicts offsets |
| Focal Loss | `−α(1−p)^γ log(p)` — down-weights easy examples |
| RetinaNet | First single-stage method to match two-stage accuracy (via Focal Loss) |
| FPN | Feature Pyramid Network — multi-scale features for small/large objects |
| Confidence | `P(object) × IoU(pred, truth)` — combined objectness × overlap |
| Speed (YOLOv3 @ 416) | ~29 ms / frame (30+ FPS on commodity GPU) |
| ONNX | Open Neural Network Exchange — portable format (run via OpenCV DNN, ONNX Runtime) |
| Ultralytics | The modern YOLOv5+ Python package |

### 🪞 Basic → Intermediate → Advanced — YOLO output decoding

**Basic** — for a grid cell, network outputs `(tx, ty, tw, th, obj, c1...cC)`.
- `(tx, ty)`: offset within the cell (sigmoid → ∈ [0,1])
- `(tw, th)`: log-scale relative to anchor
- `obj`: probability there's an object
- `c1..cC`: per-class probabilities

**Intermediate** — recover absolute bbox in image coordinates:
```python
bx = (tx + cell_x) / grid_w * img_w
by = (ty + cell_y) / grid_h * img_h
bw = anchor_w * exp(tw)
bh = anchor_h * exp(th)
```

**Advanced** — apply confidence threshold (drop boxes with `obj × class_prob < τ`), then **per-class NMS**. Final mAP is sensitive to both thresholds; tune on a held-out set.

### ⚙️ Top APIs

```python
# Ultralytics YOLOv5+
from ultralytics import YOLO
m = YOLO('yolov5s.pt')
results = m('image.jpg')                       # or video, webcam, batch

# OpenCV DNN (portable, runs any ONNX detector)
net = cv2.dnn.readNet('yolov5s.onnx')
blob = cv2.dnn.blobFromImage(img, 1/255, (640,640), [0,0,0], swapRB=True, crop=False)
net.setInput(blob); outputs = net.forward(net.getUnconnectedOutLayersNames())

# torchvision (RetinaNet, SSD)
torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
torchvision.models.detection.ssd300_vgg16(pretrained=True)

# Post-processing
cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
```

### 🧩 Code patterns

```python
# 1. YOLOv5 in 4 lines
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
results = model('test.jpg')
results[0].show()

# 2. ONNX inference via OpenCV
net = cv2.dnn.readNet('yolov5s.onnx')
def detect(img):
    blob = cv2.dnn.blobFromImage(img, 1/255, (640,640), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(net.getUnconnectedOutLayersNames())

# 3. Post-process YOLO outputs
def postprocess(outputs, img_shape, conf_t=0.45, nms_t=0.45):
    boxes, scores, class_ids = [], [], []
    for row in outputs[0][0]:                     # iterate predictions
        if row[4] < conf_t: continue
        scores_per_class = row[5:]
        cid = int(np.argmax(scores_per_class))
        if scores_per_class[cid] * row[4] < conf_t: continue
        # convert center-format to corner-format and scale to image
        ...
        boxes.append([x, y, w, h]); scores.append(...); class_ids.append(cid)
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_t, nms_t)
    return boxes, scores, class_ids, keep

# 4. Video inference
cap = cv2.VideoCapture('road.mp4')
while True:
    ok, frame = cap.read()
    if not ok: break
    outs = detect(frame)
    annotated = draw(frame, *postprocess(outs, frame.shape))
    cv2.imshow('det', annotated); cv2.waitKey(1)
```

### 🎯 Q&A — Module 6

> Mix of original + questions adapted from `andrewekhalel/MLQuestions` and `chiphuyen/ml-interviews-book` (system design).

1. **Why is YOLO faster than Faster R-CNN?** One forward pass over the entire image vs propose-then-classify (separate networks). YOLO trades a few percentage points of mAP for ~3× speedup.
2. **What is Focal Loss and what problem does it solve?** *(classic interview question)* `−(1−p)^γ log(p)` — when the predicted probability for the true class is high (easy example), the `(1−p)^γ` term shrinks the loss to near-zero, focusing gradient on **hard** examples. Solves the single-stage class imbalance problem (millions of background boxes vs a few objects).
3. **What does the "objectness" output of YOLO represent?** `P(object) × IoU(pred, truth)` — combines "is there anything here" with "if so, how good is my box."
4. **Single-stage vs two-stage — when which?** Single-stage: real-time (autonomous driving, video, mobile). Two-stage: maximum accuracy, latency not critical (medical, security review).
5. **What is FPN (Feature Pyramid Network)?** A top-down pathway that combines features from different scales — high-resolution features (good for small objects) get semantic context from deeper layers. Used in modern YOLOs, RetinaNet, Mask R-CNN.
6. **Anchor-free detection — what's the idea?** Predict bbox directly (center + size) without predefined anchor templates. Examples: CenterNet, FCOS, DETR. Removes one set of hyperparameters but adds others.
7. **How do you train YOLO on a custom dataset?** Convert annotations to YOLO format (`class x_center y_center w h`, normalized), point `data.yaml` at it, run `yolo train data=data.yaml model=yolov5s.pt epochs=...`.

[🔝 Back to top](#top)

---

<a id="7-module7"></a>
## 7. Module 7 — Object Segmentation

> Notebook 7 — **portrait mode**: classify *every pixel* as person/background. Encoder-decoder architectures (FCN, **U-Net**), skip connections that preserve spatial detail, Dice loss for imbalanced masks, **Mask R-CNN** (instance segmentation), DeepLab v3 intuition. Demonstrated on a 18,698-image portrait segmentation dataset.

### 🪜 Mental model

**Segmentation = classification per pixel.** The output is the same shape as the input image, but each pixel carries a class label (or a probability over classes).

The big architectural trick is **encoder-decoder with skip connections**:
- **Encoder** downsamples: each level captures *more abstract* features at *lower spatial resolution* (good at "what").
- **Decoder** upsamples back to the original resolution (good at "where").
- **Skip connections** between matching encoder/decoder levels carry fine-grained spatial info forward — without them, edges blur into oblivion.

Think of the U-Net's "U-shape" as a memory-mover: information flows down, then back up, with side-channels at every level.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Semantic segmentation | Per-pixel class label; all "cat" pixels share one label |
| Instance segmentation | Per-pixel class label + instance ID — different cats have different IDs |
| Panoptic segmentation | Unifies semantic (stuff) + instance (things) |
| FCN (Fully Convolutional Network) | First end-to-end pixel-wise CNN (2015) |
| U-Net | Symmetric encoder-decoder with skip connections |
| Transposed convolution | Learnable upsampling (`Conv2DTranspose`) |
| UpSampling2D | Fixed (nearest/bilinear) upsampling — no learnable params |
| Dice coefficient | `2|A∩B| / (|A| + |B|)` — overlap metric robust to class imbalance |
| Dice loss | `1 − Dice` — used as a loss directly |
| Pixel accuracy | Fraction of correctly classified pixels — misleading on imbalanced classes |
| Mean IoU | Average IoU across classes — the standard segmentation metric |
| Mask R-CNN | Faster R-CNN + a mask head — instance segmentation |
| DeepLab v3 | Atrous (dilated) convolutions for multi-scale context without downsampling |

### 🪞 Basic → Intermediate → Advanced — Dice loss

**Basic** — the Dice coefficient (a smooth F1 for masks):
```python
def dice(y_true, y_pred, smooth=1):
    inter = K.sum(y_true * y_pred)
    return (2 * inter + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)
```

**Intermediate** — why Dice and not pixel cross-entropy? Cross-entropy is per-pixel and gets dominated by the majority class (background). Dice measures *overall mask overlap* — naturally robust to imbalance.

**Advanced** — in practice combine **`BCE + Dice`** (or `CE + Dice` for multi-class). BCE gives stable gradients early in training; Dice fine-tunes mask boundaries. The combo is the de-facto standard for medical and portrait segmentation.

### 🪞 Basic → Intermediate → Advanced — upsampling

**Basic** — `UpSampling2D(2)` doubles spatial dims via nearest-neighbor interpolation. No parameters, no learning.

**Intermediate** — `Conv2DTranspose(filters, kernel, strides=2)` is a learnable upsampler. Parameters: `(kH × kW × Cin + 1) × Cout`. Often produces *checkerboard artifacts* if `kernel % stride != 0`.

**Advanced** — alternatives that avoid checkerboard:
- `UpSampling2D + Conv2D` (used in U-Net, cheap and clean)
- Sub-pixel convolution (PixelShuffle) — output gets `r²` channels then reshapes spatially
- Bilinear `UpSampling2D` + learnable `Conv2D`

### ⚙️ Top APIs

```python
# Keras layers
layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')
layers.UpSampling2D(size=2, interpolation='nearest'|'bilinear')
layers.Concatenate()                         # for skip connections

# Pretrained segmentation models
# Keras / segmentation_models_pytorch:
import segmentation_models_pytorch as smp
m = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=2, activation='softmax')

# torchvision
torchvision.models.segmentation.fcn_resnet50(pretrained=True)
torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
```

### 🧩 Code patterns

```python
# 1. U-Net skeleton (encoder + decoder + skips)
def unet(input_shape=(128,128,3), n_classes=2):
    inp = Input(input_shape)
    # encoder
    c1 = layers.Conv2D(64, 3, 'same', activation='relu')(inp);    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(128, 3, 'same', activation='relu')(p1);    p2 = layers.MaxPooling2D()(c2)
    # bottleneck
    b  = layers.Conv2D(256, 3, 'same', activation='relu')(p2)
    # decoder + skip concat
    u2 = layers.UpSampling2D()(b)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(128, 3, 'same', activation='relu')(u2)
    u1 = layers.UpSampling2D()(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(64, 3, 'same', activation='relu')(u1)
    out = layers.Conv2D(n_classes, 1, activation='softmax')(d1)
    return Model(inp, out)

# 2. Dice loss
def dice_loss(y_true, y_pred):
    inter = K.sum(y_true * y_pred); union = K.sum(y_true) + K.sum(y_pred)
    return 1 - (2*inter + 1) / (union + 1)

# 3. Portrait blur effect after prediction
mask = np.argmax(model.predict(img[None])[0], axis=-1)      # (H, W) labels
blurred = cv2.GaussianBlur(img, (21, 21), 0)
out = np.where(mask[..., None] == 1, img, blurred)

# 4. Mask R-CNN inference (torchvision)
m = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
out = m([img_tensor])
masks = out[0]['masks']        # (N, 1, H, W) probability masks per instance
```

### 🎯 Q&A — Module 7

> Mix of original + questions adapted from `andrewekhalel/MLQuestions` (encoder-decoder, ResNets) and `alexeygrigorev/data-science-interviews` (segmentation).

1. **Semantic vs instance vs panoptic — define each.** Semantic: per-pixel class. Instance: per-pixel class + instance ID. Panoptic: union of both — "stuff" classes (sky, road) get semantic labels, "thing" classes (cars, people) get instance IDs.
2. **What does U-Net's "U-shape" buy you?** *(from `andrewekhalel`)* The encoder destroys spatial detail to gain semantic abstraction; the decoder upsamples back; the skips deliver fine-grained appearance straight to the decoder so boundaries stay sharp.
3. **Transposed conv vs UpSampling2D?** Transposed conv has learnable parameters and can produce checkerboard artifacts. UpSampling2D is fixed interpolation, no params. U-Net usually uses `UpSampling2D + Conv` for efficiency.
4. **Why Dice loss over cross-entropy for masks?** CE is per-pixel and gets dominated by the majority class. Dice measures global overlap and is naturally robust to imbalance.
5. **Mask R-CNN = ?** Faster R-CNN + a per-ROI mask head (a small FCN). One forward pass yields boxes, classes, *and* masks.
6. **How does Mask R-CNN differ from U-Net?** Mask R-CNN is **instance-aware** (detects each object then segments it). U-Net is **semantic only** (no notion of instances). For "count and segment each car separately" → Mask R-CNN. For "produce a person/background mask" → U-Net.
7. **What does atrous (dilated) convolution buy DeepLab?** It expands the receptive field *without* downsampling — keeps spatial resolution high while seeing global context.

[🔝 Back to top](#top)

---

<a id="8-module8"></a>
## 8. Module 8 — Siamese Networks

> Notebook 8 — **one-shot learning**: verify whether two signatures (or faces, fingerprints, etc.) belong to the same identity. **Twin networks with shared weights** + a distance metric. **Contrastive loss** on pairs, **triplet loss** with anchor/positive/negative + margin, hard-negative mining. Demonstrated on **BHSig260** (260 signers × ~54 signatures each) with a ResNet-50 backbone.

### 🪜 Mental model

**Two networks with shared weights produce embeddings; distance between them = similarity.** Because weights are shared, the function is **symmetric**: `f(A)` and `f(B)` come from the same map, so the distance `d(f(A), f(B))` is symmetric and zero when `A = B`.

You don't classify "what identity is this?" — you ask "are these two from the same identity?" That's why Siamese nets handle **open-set** problems (new identities at deploy time) which standard CNNs can't.

**Contrastive vs triplet:**
- Contrastive operates on pairs. "Push together if same, pull apart if different (up to margin)."
- Triplet uses a triple. "Anchor's positive should be closer than anchor's negative by margin α."

Triplet is more stable in practice because it enforces a *relative* ordering instead of absolute distances.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Siamese network | Two (or more) twin networks with **shared weights** |
| Embedding | Output vector used for similarity comparison |
| Distance metric | Euclidean or cosine on embeddings |
| Contrastive loss | `y·d² + (1-y)·max(margin-d, 0)²` on pairs |
| Triplet loss | `max(0, d(a,p) - d(a,n) + α)` on (anchor, positive, negative) |
| Margin (α) | "Required gap" between positive and negative distances |
| Hard negative | A negative `n` such that `d(a,n) < d(a,p)` — violates the margin maximally |
| Semi-hard negative | `d(a,p) < d(a,n) < d(a,p) + α` — within the margin |
| Easy negative | `d(a,n) > d(a,p) + α` — already correct, gradient = 0 |
| Mining | Strategy for picking informative triplets/pairs (hard / semi-hard / random) |
| One-shot learning | Recognize from a single example by comparing to a reference embedding |
| Verification | "Same person?" — binary decision on a pair |
| Use cases | Face verification (FaceNet), signature verification, deduplication, near-duplicate search |

### 🪞 Basic → Intermediate → Advanced — contrastive loss

**Basic** — squared Euclidean distance with a margin.
```python
def contrastive_loss(y, d, margin=1.0):
    return K.mean(y * K.square(d) + (1-y) * K.square(K.maximum(margin - d, 0)))
```

**Intermediate** — `y = 1` for same-identity pairs (push together → minimize `d²`); `y = 0` for different (push apart, but only until distance exceeds margin → no incentive to push further).

**Advanced** — failure mode: if the margin is too small, the network collapses negatives to exactly margin distance (no incentive to do better). If too large, hard negatives never reach zero loss. Practical heuristic: start with margin = 1 on unit-norm embeddings; tune on validation.

### 🪞 Basic → Intermediate → Advanced — triplet loss

**Basic** — formula and idea.
```python
loss = max(0, d(a, p) - d(a, n) + α)
```
"Anchor's positive must be closer than its negative by at least α."

**Intermediate** — naive triplet sampling (random) yields mostly *easy* triplets where `d(a,n) ≫ d(a,p) + α`. Gradient = 0. Training stalls. Fix: **online hard-negative mining** — within a batch, pick negatives that violate the margin.

**Advanced** — Schroff et al. (FaceNet, 2015) found semi-hard negatives work better than hardest negatives (the hardest can dominate updates and destabilize training). Production setup: large batches (1000+), online semi-hard mining per batch, BN-statistic-friendly architectures.

### ⚙️ Top APIs

```python
# Build a Siamese encoder
def build_encoder(input_shape):
    base = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
    base.trainable = False
    x = layers.Dense(128)(base.output)
    return Model(base.input, x, name='embedding')

# Pair distance (Euclidean) as a Lambda layer
distance = layers.Lambda(lambda v: K.sqrt(K.sum(K.square(v[0]-v[1]), axis=1, keepdims=True)))
```

### 🧩 Code patterns

```python
# 1. Siamese pair model with contrastive loss
enc = build_encoder((180,180,3))
a, b = Input((180,180,3)), Input((180,180,3))
d = distance([enc(a), enc(b)])
model = Model([a, b], d)
model.compile(loss=contrastive_loss, optimizer='adam')

# 2. Triplet loss model
def triplet_loss(_, dists, alpha=0.5):
    ap, an = dists[:, 0], dists[:, 1]
    return K.mean(K.maximum(0.0, ap - an + alpha))

a, p, n = Input(shape), Input(shape), Input(shape)
ea, ep, en = enc(a), enc(p), enc(n)
d_ap = K.sum(K.square(ea - ep), axis=-1, keepdims=True)
d_an = K.sum(K.square(ea - en), axis=-1, keepdims=True)
out  = layers.Concatenate()([d_ap, d_an])
model = Model([a, p, n], out)
model.compile(loss=triplet_loss, optimizer='adam')

# 3. Threshold sweep at inference (verification)
def best_threshold(distances, labels):
    best_acc, best_t = 0, 0
    for t in np.arange(distances.min(), distances.max(), 0.01):
        pred = (distances <= t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc: best_acc, best_t = acc, t
    return best_t, best_acc
```

### 🎯 Q&A — Module 8

> Mix of original + questions adapted from `alexeygrigorev/data-science-interviews` (deep metric learning) and `chiphuyen/ml-interviews-book` (FaceNet / signature verification).

1. **Why Siamese instead of a classifier for face/signature verification?** A classifier needs a fixed number of classes — adding a new person requires retraining. A Siamese network learns a similarity metric, so new identities can be enrolled by just storing their embedding.
2. **Why must the two branches share weights?** *(common opener)* Shared weights guarantee that `f(A) = f(B)` when `A = B`, making the distance metric symmetric and consistent. Without sharing, you'd learn two different functions and the comparison would be incoherent.
3. **Contrastive vs triplet loss — which to use?** Contrastive is simpler and works for pair-labeled data. Triplet enforces a *relative* ordering, often gives sharper embeddings, but needs triplet sampling. Production face recognition (FaceNet) uses triplet + semi-hard mining.
4. **What's the margin α and what does it control?** The minimum required gap between positive and negative distances. Too small → embeddings collapse (no incentive to spread out). Too large → optimization can't satisfy the constraint and gets stuck.
5. **Easy / semi-hard / hard negative — what does each do during training?**
   - Easy negatives: `d(a,n) > d(a,p) + α` → loss = 0, **no gradient**, wasteful.
   - Semi-hard: `d(a,p) < d(a,n) < d(a,p) + α` → moderate loss, healthy gradient.
   - Hard: `d(a,n) < d(a,p)` → biggest loss, can destabilize.
6. **One-shot learning enrollment workflow?** (1) Compute embedding for the new identity's reference image. (2) Store it. (3) At verification time, compute embedding of the query, measure distance to the reference, decide based on threshold.
7. **What is FaceNet's contribution?** *(common interview)* Showed that a deep network trained with triplet loss and online semi-hard mining can learn embeddings such that simple Euclidean distance solves face verification at human-level accuracy.

[🔝 Back to top](#top)

---

<a id="9-module9"></a>
## 9. Module 9 — GANs for Image Generation

> Notebook 9 — **Generative Adversarial Networks**: a **generator** creates fake images from random noise; a **discriminator** tries to tell real from fake. The two play a minimax game; at Nash equilibrium the generator has learned the data distribution. Demonstrated on the **Anime Faces** dataset (63k images, 64×64) with **DCGAN** architecture.

### 🪜 Mental model

**Counterfeiter (G) vs police (D).** The counterfeiter starts producing terrible fakes; the police easily catches them. As D learns to spot the fakes, G updates to fool D better. As G improves, D has to look more carefully. They co-evolve — neither wins permanently, and at equilibrium D outputs 0.5 for everything because G is making fakes indistinguishable from real.

Two practical implications:
- **Training is unstable.** Two losses chasing each other. Lots of hyperparameter sensitivity. Use DCGAN guidelines as a starting point.
- **You can't easily tell when it's "done."** Loss doesn't monotonically decrease. Use **FID** (Fréchet Inception Distance) or visual inspection of generated samples.

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Generator (G) | Network mapping random noise `z ~ N(0, I)` → fake image |
| Discriminator (D) | Network mapping image → P(real) |
| Latent vector `z` | Random input to G; controls what's generated (`dim` typically 100–256) |
| Minimax objective | `min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]` |
| Non-saturating G loss | `−log D(G(z))` — same gradient signal, doesn't vanish |
| DCGAN | Deep Convolutional GAN — strided convs + BN + LeakyReLU (the canonical "stable GAN" recipe) |
| Conditional GAN | G and D both see a class label → can generate per class |
| Mode collapse | G produces only a few "safe" outputs that fool D |
| Vanishing gradient | Optimal D → G receives no useful gradient → training stalls |
| FID | Fréchet Inception Distance — distance between real and generated distributions in Inception feature space |
| IS | Inception Score — assumes labels matter; less used today |
| DCGAN hyperparams | Adam, lr=2e-4, β1=0.5, BatchNorm everywhere, LeakyReLU(0.2) in D, tanh in G output |
| Variants worth knowing | DCGAN, cGAN, WGAN (Wasserstein), CycleGAN, StyleGAN, SRGAN |

### 🪞 Basic → Intermediate → Advanced — GAN training loop

**Basic** — alternate updates: train D one step, then G one step.
```python
for x_real in batches:
    z = sample_noise(batch_size)
    x_fake = G(z)
    # 1. D step: real vs fake
    d_loss = BCE(D(x_real), 1) + BCE(D(x_fake.detach()), 0)
    d_loss.backward(); d_optimizer.step()
    # 2. G step: try to fool D
    g_loss = BCE(D(G(z)), 1)
    g_loss.backward(); g_optimizer.step()
```

**Intermediate** — use the **non-saturating G loss** `−log D(G(z))` instead of `log(1 − D(G(z)))`. The latter saturates near 0 when D is winning (gradient vanishes). The former gives a strong gradient signal throughout.

**Advanced** — DCGAN training tricks: (1) `Adam(lr=2e-4, β1=0.5)` (low β1 is critical), (2) BatchNorm everywhere except D's input and G's output, (3) LeakyReLU(0.2) in D, ReLU in G, tanh on G's final layer, (4) Normalize images to `[-1, 1]` to match tanh output. Skip these and you'll spend a week wondering why your GAN doesn't train.

### 🪞 Basic → Intermediate → Advanced — mode collapse

**Basic** — G outputs the same image (or a few) regardless of z. D can't distinguish them as "fake" because they look real; G has found a local optimum.

**Intermediate** — diagnose by sampling many `z`s and looking at the variety. If all samples look identical → collapse.

**Advanced** — fixes: (1) **minibatch discrimination** — D sees stats across the batch and can detect lack of variety. (2) **Wasserstein loss** (WGAN) — uses Earth Mover's Distance, smoother gradients, fewer mode-collapse failures. (3) **Spectral normalization** — bounds D's Lipschitz constant.

### ⚙️ Top APIs

```python
# Generator (DCGAN style) — Keras
def make_generator(z_dim=128):
    return keras.Sequential([
        layers.Dense(4*4*512, input_shape=(z_dim,)), layers.Reshape((4,4,512)),
        layers.Conv2DTranspose(256, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose( 64, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(  3, 4, strides=2, padding='same', activation='tanh'),   # 64×64×3
    ])

# Discriminator
def make_discriminator():
    return keras.Sequential([
        layers.Conv2D( 64, 4, strides=2, padding='same', input_shape=(64,64,3)), layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(512, 4, strides=2, padding='same'), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Flatten(), layers.Dense(1),
    ])
```

### 🧩 Code patterns

```python
# 1. Custom Keras GAN model
class GAN(keras.Model):
    def __init__(self, G, D, z_dim):
        super().__init__(); self.G, self.D, self.z_dim = G, D, z_dim
    def compile(self, g_opt, d_opt, loss):
        super().compile(); self.g_opt, self.d_opt, self.loss_fn = g_opt, d_opt, loss
    def train_step(self, real):
        bs = tf.shape(real)[0]
        z = tf.random.normal((bs, self.z_dim))
        fake = self.G(z)
        combined = tf.concat([real, fake], axis=0)
        labels   = tf.concat([tf.ones((bs,1)), tf.zeros((bs,1))], axis=0)
        with tf.GradientTape() as t:
            d_loss = self.loss_fn(labels, self.D(combined))
        self.d_opt.apply_gradients(zip(t.gradient(d_loss, self.D.trainable_weights), self.D.trainable_weights))
        with tf.GradientTape() as t:
            g_loss = self.loss_fn(tf.ones((bs,1)), self.D(self.G(z)))
        self.g_opt.apply_gradients(zip(t.gradient(g_loss, self.G.trainable_weights), self.G.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}

# 2. Sample and visualize
z = tf.random.normal((16, 128))
imgs = (G(z) + 1) / 2          # rescale [-1, 1] to [0, 1]
plt.imshow(np.hstack([img.numpy() for img in imgs]))

# 3. Normalize real images to [-1, 1] (matches G's tanh)
real_ds = real_ds.map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5)

# 4. Conditional GAN — concatenate class label
inp = layers.Concatenate()([z, class_one_hot])
fake = G(inp)
```

### 🎯 Q&A — Module 9

> Mix of original + questions adapted from `andrewekhalel/MLQuestions`, `alexeygrigorev/data-science-interviews`, and common GAN interview classics.

1. **What's a GAN?** Two neural networks (Generator and Discriminator) trained adversarially — G learns to produce fakes that fool D, D learns to tell real from fake. At Nash equilibrium, G samples from the data distribution.
2. **Generator loss formulation — saturating vs non-saturating?** Saturating: `log(1 − D(G(z)))` (the original minimax loss). Non-saturating: `−log D(G(z))`. The non-saturating form gives stronger gradients when D is winning; almost always preferred in practice.
3. **What is mode collapse?** *(classic GAN interview)* G learns to produce only a few "safe" outputs that consistently fool D, ignoring most of the data distribution. Diagnose by looking at sample diversity; fix with minibatch discrimination, WGAN loss, or spectral normalization.
4. **Why LeakyReLU in D?** *(from `andrewekhalel`)* ReLU's zero slope for negative inputs kills gradients flowing back to G — D learns but G never gets feedback. LeakyReLU's `0.2x` slope keeps the gradient alive.
5. **Why tanh on G's output?** Outputs in `[-1, 1]`, matching how the real images are normalized. Using sigmoid `[0, 1]` works too but mismatched ranges hurt convergence.
6. **What is FID?** Fréchet Inception Distance — compute Inception features for real and generated images, fit Gaussians to each, compute the Fréchet distance between them. Lower = closer distributions. **Standard GAN evaluation metric.**
7. **WGAN — what's the key idea?** Replace the JS-divergence-based loss with the **Wasserstein distance** (Earth Mover's Distance). Gradients exist everywhere (not just where distributions overlap), training is more stable, less prone to mode collapse. Requires the discriminator (now called "critic") to be 1-Lipschitz, enforced via weight clipping or gradient penalty.
8. **DCGAN architectural rules?** (1) Strided conv for downsampling, transposed conv for upsampling. (2) BatchNorm in both G and D (except G's output and D's input). (3) LeakyReLU(0.2) in D, ReLU in G, tanh on G's output. (4) No fully connected layers. (5) Adam(lr=2e-4, β1=0.5).

[🔝 Back to top](#top)

---

<a id="10-terms"></a>
## 10. 📚 Master terms glossary

| Term | Definition |
|---|---|
| Anchor box | Predefined reference box; network predicts offsets from it |
| Augmentation | Random transforms (flip, rotate, crop, color jitter) applied to training images |
| Backbone | Pretrained convolutional portion of a model (no classifier head) |
| Batch Normalization | Per-batch, per-channel input normalization with learnable affine |
| Bounding box (bbox) | Rectangle around an object — `(x1,y1,x2,y2)` or `(xc,yc,w,h)` |
| Conv2D | Sliding learnable filter producing a feature map per filter |
| Contrastive loss | Pair-based metric loss: pull same closer, push different apart up to margin |
| DCGAN | "Deep Convolutional GAN" — canonical stable GAN recipe |
| Dice coefficient | `2|A∩B| / (|A| + |B|)` — overlap metric |
| Dilated/atrous conv | Convolution with spaced kernel taps; expands receptive field without downsampling |
| Discriminator | GAN network that classifies real vs fake |
| Dropout | Random activation zeroing during training (implicit ensemble) |
| Embedding | Dense feature vector used for similarity / retrieval |
| Encoder-decoder | Downsample → bottleneck → upsample architecture (FCN, U-Net) |
| Feature map | Output of a Conv layer — a tensor of activations |
| FID | Fréchet Inception Distance — GAN evaluation metric |
| Fine-tuning | Unfreeze top layers of a pretrained model and train with small LR |
| Focal Loss | `−α(1−p)^γ log(p)` — down-weights easy examples |
| FPN | Feature Pyramid Network — multi-scale feature combination |
| Generator | GAN network that maps noise → fake image |
| GlobalAveragePooling2D | Average each feature map to a single value; replaces Flatten+Dense |
| Hard negative | Negative sample close to the anchor — drives metric learning |
| Instance segmentation | Per-pixel class + instance ID |
| IoU | `area(intersection) / area(union)` — overlap quality |
| Latent vector (`z`) | Random input to a GAN's generator |
| LeakyReLU | `f(x) = x if x>0 else 0.2x` — keeps gradient alive for negatives |
| L1 / L2 regularization | Loss penalties — L1 sparsifies, L2 shrinks smoothly |
| mAP | Mean Average Precision — detection metric |
| Mask R-CNN | Faster R-CNN + per-ROI mask head |
| Margin (α) | Required separation between positive and negative distances in metric loss |
| Mode collapse | GAN G produces only a few outputs that fool D |
| MaxPooling2D | Take max in each window — fixed downsampling |
| NMS | Non-Maximum Suppression — deduplicate overlapping detections |
| Padding | Zeros added around input — `'same'` preserves output size, `'valid'` shrinks |
| Pixel accuracy | Fraction of correctly classified pixels (poor on imbalanced classes) |
| Pooling | Fixed downsampling — MaxPool or AvgPool |
| Receptive field | Region of input that influences a particular output activation |
| ReLU | `max(0, x)` — fast, non-saturating for positives |
| Residual block | `out = F(x) + x` — skip connection that enables very deep nets |
| ROI Align / ROI Pool | Resize variable-size proposals to fixed grid for the classifier head |
| RPN | Region Proposal Network — replaces Selective Search |
| Semantic segmentation | Per-pixel class (no instance differentiation) |
| Siamese network | Twin networks with shared weights |
| Skip connection | Direct path from encoder to decoder (U-Net) or block to block (ResNet) |
| Softmax | Convert logits to a probability distribution over classes |
| Stride | Pixels the filter moves per step |
| Transfer learning | Reuse a pretrained model's weights for a new task |
| Transposed convolution | Learnable upsampling — opposite of strided conv in shape |
| Triplet loss | `max(0, d(a,p) - d(a,n) + α)` — relative ordering with margin |
| t-SNE / UMAP | Non-linear dim-reduction for visualization (not for retrieval) |
| U-Net | Symmetric encoder-decoder with skip concatenations |
| Vanishing gradient | Deep-net gradients decay to ~0 — fixed by ReLU, BatchNorm, residuals |
| WGAN | Wasserstein GAN — uses Earth Mover's Distance for stable training |
| YOLO | "You Only Look Once" — grid-based single-stage detector |

[🔝 Back to top](#top)

---

<a id="11-apis"></a>
## 11. ⚙️ API cheat sheet — every method, one place

### Keras layers (CNN core)
| Call | Purpose |
|---|---|
| `layers.Conv2D(filters, kernel_size, strides, padding, activation)` | Sliding learnable filter |
| `layers.MaxPooling2D(pool_size, strides)` | Max downsampling |
| `layers.AveragePooling2D(pool_size)` | Average downsampling |
| `layers.GlobalAveragePooling2D()` | Avg each feature map to 1 value |
| `layers.Flatten()` | 2D/3D → 1D vector |
| `layers.Dense(units, activation)` | Fully connected |
| `layers.Conv2DTranspose(filters, kernel, strides)` | Learnable upsample |
| `layers.UpSampling2D(size, interpolation)` | Fixed upsample |
| `layers.Concatenate()` | Combine tensors along channel axis (skip connections) |

### Regularization & training control
| Call | Purpose |
|---|---|
| `layers.Dropout(rate)` | Random zero-out |
| `layers.BatchNormalization()` | Per-batch normalization |
| `regularizers.l1(λ)` / `regularizers.l2(λ)` | Weight-decay penalties |
| `layers.RandomFlip / Rotation / Crop / Brightness / Contrast` | Built-in augmentation layers |
| `keras.callbacks.EarlyStopping(monitor, patience)` | Halt when val plateaus |
| `keras.callbacks.ReduceLROnPlateau(factor, patience)` | Decay LR on stall |
| `keras.callbacks.ModelCheckpoint(filepath, save_best_only)` | Persist best model |

### Pretrained backbones
| Call | Purpose |
|---|---|
| `tf.keras.applications.VGG16/19(weights='imagenet', include_top=False)` | VGG backbones |
| `tf.keras.applications.ResNet50/101/152(...)` | ResNet backbones |
| `tf.keras.applications.InceptionV3(...)` / `Xception(...)` | Inception family |
| `tf.keras.applications.EfficientNetB0–B7(...)` | Efficient backbones |
| `tf.keras.applications.MobileNetV2(...)` | Mobile-grade backbone |

### Detection (torchvision)
| Call | Purpose |
|---|---|
| `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` | Faster R-CNN |
| `torchvision.models.detection.retinanet_resnet50_fpn(...)` | RetinaNet (focal loss) |
| `torchvision.models.detection.ssd300_vgg16(...)` | SSD |
| `torchvision.ops.nms(boxes, scores, iou_thr)` | NMS |
| `torchvision.ops.box_iou(b1, b2)` | Pairwise IoU |
| `torchvision.ops.roi_align(features, boxes, output_size)` | ROI align |

### Segmentation
| Call | Purpose |
|---|---|
| `torchvision.models.segmentation.fcn_resnet50/101(...)` | FCN |
| `torchvision.models.segmentation.deeplabv3_resnet50/101(...)` | DeepLab v3 |
| `torchvision.models.detection.maskrcnn_resnet50_fpn(...)` | Mask R-CNN |
| `segmentation_models_pytorch.Unet(encoder_name=...)` | U-Net builder |

### Similarity / NN search
| Call | Purpose |
|---|---|
| `sklearn.neighbors.NearestNeighbors(n_neighbors, metric)` | Brute-force NN |
| `annoy.AnnoyIndex(dim, metric='angular'/'euclidean')` | Approximate NN |
| `faiss.IndexFlatL2(dim)` / `faiss.IndexIVFFlat(...)` | FAISS indices |
| `sklearn.decomposition.PCA(n_components)` | Dim reduction |
| `sklearn.manifold.TSNE(n_components=2)` | Visualization |

### OpenCV (preprocessing / I/O)
| Call | Purpose |
|---|---|
| `cv2.imread(path)`, `cv2.imwrite(path, img)` | Load / save |
| `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | BGR ↔ RGB (OpenCV is BGR!) |
| `cv2.resize(img, (w, h))` | Resize |
| `cv2.GaussianBlur(img, (k, k), 0)` | Blur |
| `cv2.rectangle(img, pt1, pt2, color, thickness)` | Draw box |
| `cv2.dnn.readNet(weights, config)` | Load ONNX/Caffe/Darknet |
| `cv2.dnn.blobFromImage(img, scale, size, mean, swapRB)` | Preprocess for DNN |
| `cv2.dnn.NMSBoxes(boxes, scores, c_t, n_t)` | NMS |

[🔝 Back to top](#top)

---

<a id="12-gotchas"></a>
## 12. ⚠️ Gotchas & traps (CV)

### Shape & preprocessing
1. **OpenCV loads images as BGR, not RGB.** Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` or feed BGR consistently to your network.
2. **Pretrained models have model-specific preprocessing.** VGG, ResNet, MobileNet, EfficientNet all use different mean/std. Use `tf.keras.applications.<model>.preprocess_input` — not just `/255`.
3. **`input_shape` mismatch silently breaks pretrained weights.** Pretrained models expect specific input sizes (224, 299, etc.). Resize before feeding.
4. **Conv2D output shape error?** Re-derive `O = (N + 2P − F) / S + 1`. If you get a non-integer, padding or stride is off.

### Training
5. **`training=False` on BatchNorm at inference.** BN uses running stats during eval, batch stats during train. Forgetting this leads to "works in dev, breaks in prod."
6. **Dropout rescales by `1/(1−p)` at training-time.** Don't apply your own scaling.
7. **Augmentation must be train-only.** Applying it to val/test corrupts the metric.
8. **Forgetting `.eval()` (PyTorch) at inference.** Same issue as `training=False` — BN/Dropout misbehave.
9. **Optimizer choice for fine-tuning.** Use `Adam(lr=1e-5)` or smaller — pretrained weights need gentle updates.

### Detection
10. **Bbox format confusion.** YOLO uses normalized `(xc, yc, w, h)`; PASCAL VOC uses pixel `(x1, y1, x2, y2)`. Verify before any IoU computation.
11. **Per-class NMS, not global.** Different classes can legitimately overlap (a person riding a bike).
12. **Plain IoU has zero gradient when boxes don't overlap.** Modern detectors use GIoU/DIoU/CIoU loss instead.
13. **Anchor box mismatch with dataset.** Default anchors (COCO) may not fit your data — K-means cluster your training bbox sizes.

### Segmentation
14. **`Conv2DTranspose` produces checkerboard artifacts when `kernel % stride != 0`.** Use `UpSampling2D + Conv2D` to avoid them.
15. **Pixel accuracy is misleading on imbalanced masks.** A model that predicts "background" everywhere scores 99% on portrait data. Use mean IoU or Dice instead.
16. **Mask R-CNN output masks are at low resolution (28×28 per ROI).** Upsample to bbox size before display.

### Embeddings & similarity
17. **Forgetting to L2-normalize embeddings.** Cosine then behaves like raw dot product — wrong scale.
18. **Using t-SNE for retrieval.** t-SNE distorts neighborhoods to "look pretty" in 2D — useless for actual NN search.
19. **`metric='angular'` in Annoy = cosine, NOT angular distance.** Confusing naming.

### Siamese / metric learning
20. **Random triplet sampling stalls training.** Most triplets are easy; gradient = 0. Use semi-hard mining.
21. **Margin too small → embedding collapse.** All embeddings shrink toward zero.
22. **Margin too large → no progress.** Loss stays high indefinitely.

### GANs
23. **`tanh` output on G, but data normalized to `[0, 1]`.** Mismatch — normalize images to `[-1, 1]`.
24. **Using `BatchNorm` in D's first layer.** Disables D's ability to learn the input distribution. Skip BN on D's input.
25. **Saturating G loss.** Use `−log D(G(z))` instead of `log(1 − D(G(z)))` to avoid vanishing gradient.
26. **No way to "test" a GAN with a loss curve.** Loss oscillates. Use FID or visual inspection.
27. **Mode collapse — checking only loss won't reveal it.** Sample many images and look at variety.

### Computation
28. **Forgetting `.detach()` (PyTorch) / `.stop_gradient` (TF).** Gradients flow where you didn't intend; D's update accidentally updates G.
29. **`with torch.no_grad()` for inference.** Saves memory dramatically; forget it and OOM at val time.
30. **GPU OOM on val.** Larger val batch + no gradients = use `eval()` + `no_grad()` + smaller batch.

[🔝 Back to top](#top)

---

<a id="13-advanced"></a>
## 13. 🎯 Advanced interview Q&A

Cross-module questions a senior interviewer would actually ask.

### Architectural design

**Q1. Why do ResNet's skip connections enable very deep networks?**
Without skips, gradients have to flow through many layers and can vanish (or explode). The residual `F(x) + x` provides an identity shortcut — gradients can flow directly through `x` even when `F(x)` has near-zero gradient. Networks of 50–1000+ layers became trainable for the first time.

**Q2. Inception's "multi-scale" intuition — why does it help?**
Real objects span multiple scales (a face at varying distances). Inception modules compute conv at several kernel sizes (1×1, 3×3, 5×5) **in parallel** and concatenate — the next layer can use whichever scale fits. Bonus: 1×1 convs cheaply reduce channels before expensive 3×3/5×5 convs.

**Q3. Why are 3×3 kernels (stacked) preferred over 5×5 or 7×7?**
Two stacked 3×3 layers have the same receptive field as one 5×5 but use `2 · (3·3·C·C) = 18C²` params vs `5·5·C·C = 25C²`. Plus two non-linearities instead of one → richer functions for fewer parameters. This is VGG's central insight.

**Q4. Why does DenseNet outperform ResNet on small datasets?**
DenseNet concatenates *all* previous feature maps (not just adds them), giving each layer access to all prior representations. More parameter-efficient, stronger regularization, but uses more memory. Wins on small data because it reuses features.

### Training & optimization

**Q5. Why do you need a validation set if you have a test set?**
Validation guides hyperparameter selection (LR, model size, augmentation strength). If you tune on test, you're contaminating it — test must remain untouched to give an unbiased estimate of generalization. Three sets: train → fit, val → tune, test → final report.

**Q6. Adam vs SGD-momentum — when each?**
Adam: faster initial convergence, robust to LR choice, dominant for transformer-style training. SGD-momentum: often reaches a better final minimum on vision tasks (sharper minima generalize less; flat minima generalize more — SGD finds flatter). Production deep-learning vision recipes often use SGD with cosine LR.

**Q7. What's the bias-variance trade-off in deep learning?**
Modern deep nets are *over-parameterized* — they live in a regime where the classical bias-variance trade-off doesn't apply cleanly (double descent). In practice: more data → less variance; better architecture → less bias; regularization controls overfitting.

### Detection & segmentation

**Q8. Why does Faster R-CNN have higher mAP than YOLO?**
Two reasons: (1) RPN proposes well-shaped regions before classification, so the classifier doesn't have to handle as much background. (2) Two-stage gets two passes at refinement — RPN bbox + final head bbox. Single-stage trades these for speed.

**Q9. How would you build a detector for an entirely new class category (e.g., specific cell types in microscopy)?**
1. Curate a labeled dataset with bounding boxes (CVAT, LabelImg).
2. Start with a pretrained detector (Faster R-CNN or YOLOv5).
3. Replace/augment the classifier head for the new class set.
4. Cluster bbox sizes from training data to derive task-specific anchors.
5. Train with heavy augmentation; the backbone freezes for the first epochs.
6. Evaluate with mAP@0.5 *and* mAP@0.5:0.95; iterate.

**Q10. Why does Mask R-CNN use ROIAlign instead of ROIPool?**
ROIPool quantizes ROI boundaries to integer grid positions — fine for classification, but mask predictions are pixel-precise. ROIAlign uses **bilinear interpolation** to avoid quantization, producing sharper masks. The difference is ~3% mAP on COCO segmentation.

### Embeddings & metric learning

**Q11. You have a face recognition system with 1M enrolled identities — how do you scale?**
Don't iterate over all 1M every query. (1) Embed all faces to a unit-norm vector (FaceNet style). (2) Build a FAISS or ScaNN index. (3) Query in ~1 ms per face. (4) Cache hot identities. (5) Re-index nightly. For 1B-scale, partition the index by hash bucket or region.

**Q12. Why does triplet loss + semi-hard mining beat triplet + random mining?**
Random triplets are mostly *easy* — `d(a,n) ≫ d(a,p) + α` already → loss = 0 → no learning. Semi-hard triplets are within the margin, contributing meaningful gradient. **Hardest** triplets often blow up training (they may represent label noise or extreme outliers).

### GANs

**Q13. Why are GANs hard to train?**
Two losses chasing each other → no monotonic decrease. Three failure modes: (1) **vanishing gradient** (D too good), (2) **mode collapse** (G fools D with a single image), (3) **non-convergence** (both oscillate). Mitigations: DCGAN architectural guidelines, WGAN-GP loss, spectral normalization, two-time-scale updates (TTUR), and patience.

**Q14. How would you evaluate a generative model?**
Quantitative: **FID** (Fréchet Inception Distance — lower is better), Precision/Recall (Kynkäänniemi 2019). Qualitative: visual diversity, mode coverage, edge-case generation. For **conditional** generation: per-class FID. Avoid Inception Score — it has documented flaws.

**Q15. What's the difference between conditional GAN, CycleGAN, and StyleGAN?**
- **Conditional GAN (cGAN):** G(z, y) — feeds the class label so you can generate per class.
- **CycleGAN:** unpaired image-to-image translation (horses ↔ zebras). Two G's, two D's, cycle consistency loss.
- **StyleGAN:** controls fine-grained style via AdaIN; latent space is disentangled (interpolate to morph between identities).

### Systems & deployment

**Q16. Your YOLO is slow on edge devices — what do you optimize?**
(1) Smaller backbone (`yolov5n` instead of `yolov5x`). (2) Quantize to INT8 (TensorRT, ONNX Runtime). (3) Prune unused channels. (4) Lower input resolution (320 instead of 640). (5) Knowledge distillation from a larger teacher. (6) Hardware-specific kernels (CoreML on Apple, NNAPI on Android).

**Q17. Your model overfits on a 5,000-image dataset — list 6 things to try, in order of expected gain.**
1. **Heavy augmentation** (flip, rotate, crop, color jitter, MixUp/CutMix).
2. **Transfer learning** from ImageNet (frozen backbone first).
3. **More regularization** (Dropout 0.5 in dense, weight decay).
4. **Smaller model** (start with MobileNet, not ResNet152).
5. **Early stopping with restore-best-weights.**
6. **Get more data** — labeled or via semi-supervised techniques.

[🔝 Back to top](#top)

---

<a id="sourced-bank"></a>
## 🌐 Sourced interview questions

> **Real questions paraphrased from canonical CV / DL interview-prep sources.** Use this as a standalone practice bank — no internet required. Each batch keeps its source.

### Batch 1 — from [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) (CV & DL)

| # | Question | One-liner answer |
|---|---|---|
| 1 | How can we use neural nets for computer vision? | Use CNNs — they exploit locality, weight sharing, and compositionality, which MLPs throw away by flattening. |
| 2 | What's a convolutional layer? | Apply a learnable filter as a sliding window over the input, producing a feature map per filter. |
| 3 | Why do we actually need convolutions? | Drastic param reduction via weight sharing — FC layers would explode on `(H, W, C)` inputs. |
| 4 | What's pooling? Why use it? | Fixed downsampling (max or avg) that reduces spatial dims, adds small translation invariance, and bounds memory. |
| 5 | How does max pooling work? | In each `k×k` window, output only the maximum value; typical default `k=2, stride=2` → halves spatial dims. |
| 6 | Are CNNs rotation-invariant by design? | **No.** Only translation-equivariant. Rotation invariance needs data augmentation (random rotation). |
| 7 | What are data augmentations? Name 5. | Flip, rotation, crop, color jitter, noise, random translation, MixUp, CutMix — virtual data multiplication. |
| 8 | Name 5 famous CNN classification architectures. | AlexNet, VGG, Inception, ResNet, DenseNet, EfficientNet, MobileNet — milestones in ImageNet history. |
| 9 | What is transfer learning? | Reuse a model pretrained on a large dataset (ImageNet) as the starting point for a related task. |
| 10 | How does transfer learning work in practice? | Freeze the backbone; replace the classifier head; train only the head; optionally unfreeze top blocks with a 10× smaller LR. |
| 11 | What's object detection? | Find bboxes (and class labels) for *all* objects in an image. |
| 12 | Which detection architectures do you know? | YOLO (v1–v8), Faster R-CNN, RetinaNet, SSD, CenterNet, DETR. |
| 13 | Semantic vs instance segmentation? | Semantic: per-pixel class (all cats one label). Instance: per-pixel class + ID (different cats get different IDs). |
| 14 | What is dropout? | Stochastically zero activations during training only — acts as an implicit ensemble of sub-networks. |
| 15 | Why is sigmoid problematic for hidden layers? | Saturates for large |x| → derivative ~0 → gradients die in deep nets. |
| 16 | Why does ReLU help? | No saturation for positive inputs → gradients flow → faster training and deeper networks. |
| 17 | What problems does Batch Normalization solve? | Reduces internal covariate shift, stabilizes training, allows larger LR, lightly regularizes. |
| 18 | What is weight initialization, and what if all weights = 0? | All neurons in a layer compute the same output → symmetric gradients → no learning. Use He/Glorot init. |
| 19 | What is backpropagation? | Algorithm that computes gradient of the loss w.r.t. each parameter using the chain rule, then updates via gradient descent. |
| 20 | Name 4 optimizers and their key idea. | SGD (raw gradient), Momentum (smoothed gradient), Adam (per-param adaptive LR + momentum), RMSProp (per-param adaptive LR). |
| 21 | What's the difference between Adam and SGD? | Adam adapts LR per param → faster convergence but sometimes worse final minimum. SGD-momentum often generalizes better. |

### Batch 2 — from [`andrewekhalel/MLQuestions`](https://github.com/andrewekhalel/MLQuestions) (CV-focused)

| # | Question | One-liner answer |
|---|---|---|
| 22 (#15) | Describe convolution with grayscale vs RGB inputs and how kernel properties determine output shape. | Filter has `Cin` channels (1 for gray, 3 for RGB); output shape `(N+2P−F)/S + 1`; filters get summed across input channels. |
| 23 (#22) | Why convolutions over FC layers? | Preserve spatial info, exploit locality, share weights across positions → far fewer params + better inductive bias. |
| 24 (#23) | What is CNN translation invariance? | Each conv kernel slides over the image — the same feature detected anywhere triggers the same response (equivariance); pooling adds full invariance. |
| 25 (#24) | Why max-pooling in CNNs? | Reduces computation; keeps the strongest activation (most informative feature) in each region. |
| 26 (#25) | Describe encoder-decoder segmentation. | Encoder downsamples for semantic abstraction; decoder upsamples for pixel-level prediction; skip connections preserve spatial detail. |
| 27 (#26) | Why residual networks? | Identity shortcuts let gradients flow through deep nets without vanishing — enables training networks with 50–1000+ layers. |
| 28 (#28) | Why small kernels (3×3)? | Two stacked 3×3 have the same receptive field as one 5×5 but use fewer params and add another non-linearity. |
| 29 (#19) | How is NMS implemented? | Sort boxes by confidence; for each, drop later boxes with IoU > threshold; repeat. Per-class to allow overlapping classes. |
| 30 (#51) | What are the components of a GAN? | Generator (noise → image) and Discriminator (image → P(real)). Trained adversarially as a minimax game. |
| 31 (#8) | What is a receptive field? | The region of input pixels that influences a given output activation. Grows as you stack conv layers. |

### Batch 3 — from [`Sroy20/machine-learning-interview-questions`](https://github.com/Sroy20/machine-learning-interview-questions) (DL)

| # | Question | One-liner answer |
|---|---|---|
| 32 (#1) | Implement dropout during forward and backward propagation. | Forward: multiply activations by a Bernoulli(1−p) mask, then divide by (1−p) to keep expected magnitude. Backward: multiply incoming gradients by the same mask. |
| 33 (#9) | What is transfer learning and why is it useful? | Reuse a pretrained model — saves compute, fights small-data overfitting, leverages universal low-level features. |
| 34 (#10) | L1 vs L2 regularization? | L1 = `λΣ|w|` → sparse (feature selection). L2 = `λΣw²` → smooth shrinkage. |
| 35 (#11) | What does regularization do? | Adds a penalty to the loss that discourages large weights → smoother decision boundaries → less overfitting. |
| 36 (#16) | Why can't 0-1 loss optimize deep networks? | Non-differentiable, gradient = 0 almost everywhere. Use BCE / CE as smooth surrogates. |
| 37 (#18) | What is gradient clipping? | Cap the gradient norm at some threshold before the update; prevents the explosion problem in RNNs/GANs. |
| 38 (#20) | Can BN be applied to RNNs? | Not directly — sequence length varies. **LayerNorm** is the standard replacement for RNNs and Transformers. |

### Batch 4 — common FAANG-style design / system questions

| # | Question | One-liner answer |
|---|---|---|
| 39 | Design a near-duplicate image detection service for 100M images. | Pretrained backbone → 2048-D embedding → PCA to 256 → FAISS IVFFlat index → query in <10 ms. Use perceptual hashing as a coarse pre-filter. |
| 40 | Your portrait segmentation works on Pixel 4 but is too slow on Pixel 3a. What do you do? | Switch to a lighter U-Net backbone (MobileNetV3), quantize to INT8, downscale input, use depthwise-separable convs. |

### Citations & where to drill more
- 🎯 [`andrewekhalel/MLQuestions`](https://github.com/andrewekhalel/MLQuestions) — CV-focused.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — DL theory.
- 🎯 [`Sroy20/machine-learning-interview-questions`](https://github.com/Sroy20/machine-learning-interview-questions) — DL pool.
- 🎯 [`chiphuyen/ml-interviews-book`](https://huyenchip.com/ml-interviews-book/) — DL chapter + system design.
- 🎯 **Papers With Code** — SOTA per task.

[🔝 Back to top](#top)

---

<a id="14-drill"></a>
## 14. 🔁 100-question revision drill

Designed as a **timed pre-interview tool**. Read each question, answer in your head, peek. Aim for under 15 seconds per question.

### Block A — CNN fundamentals (Q1–20)
1. Image tensor shape? → **(H, W, C)** — channels last in TF/Keras
2. Pixel range? → **0–255**, rescale to [0, 1]
3. Output-shape formula? → **(N + 2P − F) / S + 1**
4. Kernel size convention? → **Odd** (3, 5, 7) — single integer center
5. `'same'` vs `'valid'` padding? → **Preserves vs shrinks output size**
6. `MaxPooling2D` default? → **`pool_size=2, strides=2`** → halves dims
7. Pooling params? → **None** — fixed reduction
8. Conv2D params formula? → **`(kH × kW × Cin + 1) × Cout`**
9. Stride doubles → output? → **Halves spatial dim, ~4× fewer ops**
10. MLP vs CNN params for `(128,128,3) → Dense(1024)`? → **~50M vs ~150k**
11. ReLU formula? → **`max(0, x)`**
12. Softmax purpose? → **Probabilities over classes** (sum to 1)
13. CNN translation invariance? → **Equivariance** (shifted in → shifted out) + pooling for invariance
14. Three CNN principles? → **Locality + stationarity + compositionality**
15. Flatten layer purpose? → **2D/3D feature map → 1D vector** for Dense
16. `GlobalAveragePooling2D` over `Flatten`? → **Avg each feature map → 1 value; ~100× fewer params**
17. Output channels = ? → **= number of filters**
18. Receptive field grows? → **As you stack conv layers (and pool)**
19. Why odd kernel? → **Single integer center pixel**
20. Cross-entropy vs MSE for classification? → **CE — gradient stays alive for correctly-classified examples**

### Block B — Regularization & training (Q21–40)
21. Dropout rate for dense? → **0.5** typical
22. Dropout rate for conv? → **0.1–0.25**
23. Dropout active at inference? → **No** — only training
24. BN at inference uses? → **Running mean/var** from training
25. BN learnable params? → **γ (scale) and β (shift)**
26. L1 → ? → **Sparse weights** (feature selection)
27. L2 → ? → **Smooth shrinkage**
28. Augmentation should be on val? → **No** — train-only
29. Augmentation should be on test? → **No**
30. `EarlyStopping(patience=10)`? → **Stop if val_loss doesn't improve for 10 epochs**
31. `ReduceLROnPlateau` does? → **Multiply LR by factor when val stalls**
32. Adam default `β1`? → **0.9** (but **0.5 for GANs**)
33. SGD vs Adam? → **Adam adapts LR per param; SGD-momentum often generalizes better**
34. Overfit signal? → **Train ↑, val plateaus** — large gap
35. Underfit signal? → **Both low / not improving**
36. ReLU dead-neuron problem? → **Negative weights → output always 0 → no gradient → use LeakyReLU**
37. `Conv → BN → ReLU` or `Conv → ReLU → BN`? → **Both work; pick one and stick**
38. Weight init for deep ReLU nets? → **He** (`kaiming_normal_`) — variance ~ 2/n_in
39. Augmentation library? → **`albumentations`** (PyTorch), `keras.layers.Random*` (Keras)
40. Validation purpose? → **Hyperparameter selection** — not final reporting

### Block C — Transfer learning & embeddings (Q41–60)
41. Transfer learning idea? → **Reuse pretrained features**
42. Freeze backbone how? → **`base.trainable = False`** or per-layer
43. Feature extraction first, then? → **Fine-tune top layers with small LR**
44. Fine-tune LR vs train-from-scratch LR? → **10× smaller**
45. VGG input size? → **(224, 224, 3)**
46. Inception input size? → **(299, 299, 3)**
47. Preprocess for VGG? → **`vgg16.preprocess_input`** — NOT just `/255`
48. AlexNet year? → **2012** (ImageNet winner that started DL boom)
49. ResNet's trick? → **Residual skip connections** for very deep nets
50. Inception's trick? → **Multi-scale conv blocks in parallel**
51. EfficientNet's idea? → **Compound scaling** (depth + width + resolution jointly)
52. MobileNet uses? → **Depthwise-separable convolutions** for efficiency
53. Embedding = ? → **Dense feature vector representing input**
54. Why penultimate layer? → **Rich semantic features before classifier collapses them**
55. ResNet-50 embedding dim? → **2048**
56. L2-normalize embeddings? → **Yes** — makes cosine ≡ L2 on unit sphere
57. Cosine vs Euclidean (after normalize)? → **Equivalent**
58. PCA from 2048 → 150 dims? → **~13× speedup, <0.5% accuracy loss**
59. Annoy = ? → **Approximate nearest-neighbour index (tree-based)**
60. FAISS = ? → **Facebook's NN library** — GPU-accelerated, billion-scale

### Block D — Detection & segmentation (Q61–80)
61. Classification vs localization vs detection? → **Label / label+bbox / many labels+bboxes**
62. IoU formula? → **Intersection / Union of bboxes**
63. Typical IoU threshold for match? → **0.5**
64. NMS purpose? → **Deduplicate overlapping detections**
65. Anchor box? → **Predefined reference; network predicts offsets**
66. R-CNN vs Fast R-CNN vs Faster R-CNN? → **Selective Search → shared features → learnable RPN**
67. RPN replaces? → **Selective Search**
68. ROI pooling does? → **Resize variable proposals to fixed grid (7×7)**
69. ROIAlign vs ROIPool? → **Bilinear interp vs quantized grid; ROIAlign for masks**
70. YOLO grid output? → **`S × S × B × (5 + C)`**
71. Focal Loss formula? → **`−α(1−p)^γ log(p)`**
72. Focal Loss for? → **Class imbalance in single-stage detection**
73. YOLOv3 scales? → **3** (13×13, 26×26, 52×52)
74. Detection eval metric? → **mAP**
75. Semantic vs instance segmentation? → **Per-pixel class vs +instance ID**
76. U-Net "U-shape"? → **Symmetric encoder-decoder with skip concatenations**
77. Mask R-CNN = ? → **Faster R-CNN + per-ROI mask head**
78. `Conv2DTranspose` problem? → **Checkerboard artifacts** when `kernel % stride ≠ 0`
79. Dice loss for? → **Class imbalance** in segmentation
80. DeepLab's atrous conv? → **Dilated kernel; receptive field without downsampling**

### Block E — Siamese, GANs, miscellaneous (Q81–100)
81. Siamese network = ? → **Twin networks with shared weights**
82. Contrastive loss inputs? → **Pair + binary label (same/different)**
83. Triplet loss inputs? → **(anchor, positive, negative)**
84. Margin α in triplet? → **Required gap between positive and negative distances**
85. Hard negative? → **`d(a,n) < d(a,p)`** — most informative for training
86. Semi-hard negative? → **`d(a,p) < d(a,n) < d(a,p) + α`** — practical sweet spot
87. Easy negative? → **`d(a,n) > d(a,p) + α`** — gradient zero, useless
88. One-shot learning enrollment? → **Store embedding of reference image, compare new ones**
89. FaceNet contribution? → **Triplet + semi-hard mining → embedding for face verification**
90. GAN components? → **Generator + Discriminator** (counterfeiter vs police)
91. GAN minimax objective? → **`min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]`**
92. Non-saturating G loss? → **`−log D(G(z))`** — stronger gradient when D wins
93. Mode collapse? → **G produces only a few outputs that consistently fool D**
94. DCGAN: optimizer? → **`Adam(lr=2e-4, β1=0.5)`**
95. DCGAN: activation in G? → **ReLU + tanh output**
96. DCGAN: activation in D? → **LeakyReLU(0.2)**
97. GAN data normalization? → **`[-1, 1]`** — match tanh output
98. FID? → **Fréchet Inception Distance** — distance between real & fake distributions
99. WGAN replaces JS-divergence with? → **Wasserstein (Earth Mover's) distance**
100. CycleGAN does? → **Unpaired image-to-image translation** (horses ↔ zebras)

**Score yourself:** 90+ = strong, 75–89 = solid, 60–74 = revise, <60 = re-read modules.

[🔝 Back to top](#top)

---

<a id="15-bestpractices"></a>
## 15. ✅ Best practices

### Architecture choice

1. **Start with a pretrained backbone** unless you genuinely have ImageNet-scale data.
2. **Match input size to the backbone** (224 for ResNet/VGG, 299 for Inception, 380+ for EfficientNet larger variants).
3. **Use `GlobalAveragePooling2D`** before the classifier head to slash params.
4. **For new tasks, freeze backbone first**, then fine-tune top blocks only after head converges.

### Training

5. **Always run `model.summary()`** before training — catches shape bugs early.
6. **Compile with the right loss** — `sparse_categorical_crossentropy` for integer labels, `categorical_crossentropy` for one-hot, `binary_crossentropy` for binary.
7. **Three callbacks always**: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
8. **Augmentation in the data pipeline, not preprocessing.** Keep val/test pipelines clean.
9. **Verify train/val accuracy curves visually** every run — overfit/underfit is obvious.

### Inference

10. **Set `training=False`** (TF) or `.eval()` (PyTorch) at inference — BN/Dropout misbehave otherwise.
11. **Wrap inference in `torch.no_grad()`** (PyTorch) — halves memory.
12. **Match preprocessing exactly** to training (`preprocess_input`, BGR/RGB, normalization).

### Detection / Segmentation

13. **Audit bbox format** before any IoU op — `(xc, yc, w, h)` normalized vs `(x1, y1, x2, y2)` pixel are easy to confuse.
14. **Per-class NMS**, not global.
15. **Cluster anchor sizes** from your training data with K-means.
16. **For masks, use BCE + Dice** loss combo on imbalanced datasets.

### Embeddings & similarity

17. **L2-normalize embeddings.** Always.
18. **PCA before NN search** for 10× speedup with negligible accuracy loss.
19. **Use Annoy / FAISS** above 100K items; brute-force is fine below.

### GANs

20. **Follow the DCGAN recipe** strictly until you understand what you can vary.
21. **`Adam(lr=2e-4, β1=0.5)`** for both G and D.
22. **Normalize real images to `[-1, 1]`** to match `tanh` output of G.
23. **Inspect samples**, not the loss curve.
24. **WGAN-GP or spectral normalization** when DCGAN won't stabilize.

### Reproducibility

25. **Set `tf.random.set_seed` / `torch.manual_seed`** AND `np.random.seed`.
26. **Pin library versions** in `requirements.txt`.
27. **Save model architecture + weights** separately for portability.
28. **Track experiments** with W&B, MLflow, or even a CSV of hyperparams → metrics.

[🔝 Back to top](#top)

---

<a id="16-mapping"></a>
## 16. 📦 Notebook mapping

| # | Notebook (folder) | What it covers | Section here |
|---|---|---|---|
| 1 | [1.Intro to CV and CNN Fundamentals](./1.Intro%20to%20CV%20and%20CNN%20Fundamentals/) | Image tensors, MLP vs CNN, Conv2D, padding/stride, pooling, output-shape formula, simple CNN on Clothing-Small | [§1](#1-module1) |
| 2 | [2.Tackling Overfitting in CNN](./2.Tackling%20Overfitting%20in%20CNN/) | Dropout, BatchNorm, L2 weight decay, augmentation, EarlyStopping, ReduceLROnPlateau, GlobalAveragePooling | [§2](#2-module2) |
| 3 | [3.Transfer learning 1](./3.Transfer%20learning%201/) | Pretrained VGG/ResNet, feature extraction, freezing, fine-tuning, classifier head replacement, 10-class landmarks | [§3](#3-module3) |
| 4 | [4.Image Similarity : Understanding Embeddings](./4.Image%20Similarity%20:%20Understanding%20Embeddings/) | Embeddings, penultimate layer, cosine/L2, PCA, brute-force vs Annoy NN search, t-SNE visualization | [§4](#4-module4) |
| 5 | [5.Object localization and detection 1](./5.Object%20localization%20and%20detection%201/) | Bbox, IoU, NMS, anchor boxes, R-CNN → Fast R-CNN → Faster R-CNN, RPN, ROI pooling | [§5](#5-module5) |
| 6 | [6.Object localization and detection 2](./6.Object%20localization%20and%20detection%202/) | YOLO grid+anchor, SSD, RetinaNet, Focal Loss, FPN, ONNX/OpenCV DNN, video inference | [§6](#6-module6) |
| 7 | [7.Object segmentation](./7.Object%20segmentation/) | FCN, U-Net, transposed conv vs UpSampling, Dice loss, Mask R-CNN, DeepLab v3 | [§7](#7-module7) |
| 8 | [8.Siamese network](./8.Siamese%20network/) | Twin networks, shared weights, contrastive loss, triplet loss + margin, hard-negative mining, signature verification | [§8](#8-module8) |
| 9 | [9.GANs for Image Generation](./9.GANs%20for%20Image%20Generation/) | Generator, Discriminator, minimax, DCGAN guidelines, mode collapse, FID, WGAN/cGAN | [§9](#9-module9) |

[🔝 Back to top](#top)
