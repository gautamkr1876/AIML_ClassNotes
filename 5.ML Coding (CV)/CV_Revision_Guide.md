<a id="top"></a>
# Computer Vision — Master Revision Guide

> **Consolidated, interview-ready revision notes for all 9 CV notebooks** (CNN fundamentals → Tackling overfitting → Transfer learning → Image similarity & embeddings → Object detection (2-stage + 1-stage) → Segmentation → Siamese networks → GANs). Every concept, every API, every gotcha — in scannable form, with mental models, basic→advanced ladders, real sourced interview questions, and a 100-question revision drill at the end.

**Companion guides (Data Foundation, for prerequisites):**
- 🐍 [`Data_Foundation_Revision_Guide.md`](../Data%20Foundation/Data_Foundation_Revision_Guide.md) — NumPy
- 🐼 [`Pandas_Revision_Guide.md`](../Data%20Foundation/Pandas_Revision_Guide.md) — Pandas
- 📊 [`Amazon_Sachin_EDA_Revision_Guide.md`](../Data%20Foundation/Amazon_Sachin_EDA_Revision_Guide.md) — applied pandas + probability

**How to use:**
- **First-time learning a concept:** open the module's **📖 Guided concept walkthrough** ([M1](#1g-guided) · [M2](#2g-guided) · [M3](#3g-guided) · [M4](#4g-guided) · [M5](#5g-guided) · [M6](#6g-guided) · [M7](#7g-guided) · [M8](#8g-guided) · [M9](#9g-guided)). Each CV concept is introduced *what → why → how → where → related → code → gotcha* — no follow-up search needed.
- **Pre-interview revision:** [🚀 Topic finder](#topic-finder) → skim a module's recap cheat sheet → drill the Q&A.
- **Just before a coding round:** run the [§14 Drill](#14-drill).
- **Quick term lookup:** [§10 Glossary](#10-terms) — every term has a 2–4 sentence beginner-friendly definition with a link back to its full walkthrough.
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
| 📖 First-time intro to a concept (what / why / how / where / related) | [M1](#1g-guided) · [M2](#2g-guided) · [M3](#3g-guided) · [M4](#4g-guided) · [M5](#5g-guided) · [M6](#6g-guided) · [M7](#7g-guided) · [M8](#8g-guided) · [M9](#9g-guided) |
| Image as tensor, channels, MLP vs CNN, locality/stationarity/compositionality | [Module 1](#1-module1) → [walkthrough](#1g-guided) |
| Conv2D math, padding (same/valid), stride, output-shape formula, pooling | [Module 1](#1-module1) → [walkthrough](#1g-guided) |
| Overfitting symptoms, train/val curves, dropout, BatchNorm, weight decay (L1/L2) | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| Augmentation pipeline, early stopping, LR scheduling, GlobalAveragePooling | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| Pretrained models, feature extraction vs fine-tuning, freezing, classifier head | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
| ResNet / VGG / Inception architectures, ImageNet, top-k accuracy | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
| Embeddings, penultimate-layer features, cosine vs L2, normalization | [Module 4](#4-module4) → [walkthrough](#4g-guided) |
| NN search (brute / Annoy / FAISS), PCA, t-SNE, reverse image search | [Module 4](#4-module4) → [walkthrough](#4g-guided) |
| Classification vs localization vs detection, bbox formats, IoU, NMS | [Module 5](#5-module5) → [walkthrough](#5g-guided) |
| Two-stage detection: R-CNN → Fast R-CNN → Faster R-CNN, RPN, ROI pooling | [Module 5](#5-module5) → [walkthrough](#5g-guided) |
| Single-stage detection: YOLO/SSD/RetinaNet, grid + anchors, focal loss | [Module 6](#6-module6) → [walkthrough](#6g-guided) |
| Real-time CV, speed-vs-accuracy trade-off, video inference | [Module 6](#6-module6) → [walkthrough](#6g-guided) |
| Semantic vs instance vs panoptic segmentation, encoder-decoder, U-Net | [Module 7](#7-module7) → [walkthrough](#7g-guided) |
| Transposed conv vs UpSampling, Dice loss, IoU metric, Mask R-CNN | [Module 7](#7-module7) → [walkthrough](#7g-guided) |
| Siamese architecture, contrastive vs triplet loss, hard-negative mining | [Module 8](#8-module8) → [walkthrough](#8g-guided) |
| One-shot / verification, embedding metric learning, signature use case | [Module 8](#8-module8) → [walkthrough](#8g-guided) |
| Generator / Discriminator, minimax game, BCE loss, DCGAN guidelines | [Module 9](#9-module9) → [walkthrough](#9g-guided) |
| Mode collapse, FID/IS, training instability, GAN variants | [Module 9](#9-module9) → [walkthrough](#9g-guided) |
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

1. [Module 1 — Intro to CV & CNN Fundamentals](#1-module1) · [📖 Guided walkthrough](#1g-guided)
2. [Module 2 — Tackling Overfitting in CNNs](#2-module2) · [📖 Guided walkthrough](#2g-guided)
3. [Module 3 — Transfer Learning](#3-module3) · [📖 Guided walkthrough](#3g-guided)
4. [Module 4 — Image Similarity & Embeddings](#4-module4) · [📖 Guided walkthrough](#4g-guided)
5. [Module 5 — Object Detection: Two-Stage](#5-module5) · [📖 Guided walkthrough](#5g-guided)
6. [Module 6 — Object Detection: Single-Stage](#6-module6) · [📖 Guided walkthrough](#6g-guided)
7. [Module 7 — Object Segmentation](#7-module7) · [📖 Guided walkthrough](#7g-guided)
8. [Module 8 — Siamese Networks](#8-module8) · [📖 Guided walkthrough](#8g-guided)
9. [Module 9 — GANs for Image Generation](#9-module9) · [📖 Guided walkthrough](#9g-guided)
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

<a id="1g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction to every Module 1 concept. Read top-to-bottom on a first pass; the recap cheat sheet below is for re-jogging your memory afterwards.

#### Computer Vision (the field)

> **🪜 Mental model:** *Teaching a computer to see.* Just as ML replaces hand-written rules with learned ones for tabular data, CV replaces hand-engineered image filters with learned filters.

**What it is.** Computer Vision (CV) is the subfield of AI that builds programs which take **images or video as input** and produce a useful answer — a class label ("dog"), a bounding box ("there's a face here"), a pixel mask, a generated image, an embedding for similarity search. Modern CV is a branch of **deep learning** (DL = ML using deep neural networks with many layers): you feed labeled images to a network, it learns what features matter, and you use the trained model on new images.

**Why it matters.** Before deep learning, CV engineers spent weeks hand-crafting features (SIFT, HOG, Haar cascades) and still got mediocre accuracy. Since 2012 (AlexNet on ImageNet), **CNNs trained end-to-end** have crushed every hand-engineered pipeline. Today CV powers self-driving cars, medical imaging, face unlock, content moderation, and visual search. Interviewers ask "why is deep learning better than classical CV" to test whether you understand this regime shift.

**How it works.**
1. **Collect labeled image data** (e.g., 10 000 labeled photos of cats and dogs).
2. **Choose an architecture** — almost always a CNN or a Vision Transformer (ViT).
3. **Train** by minimising a loss (a number that measures how wrong predictions are) via gradient descent.
4. **Evaluate** on a held-out test set (images the model never saw during training).
5. **Deploy** — run the model on new images at inference time.

**Where it's used.** Image classification (ImageNet, medical X-ray triage), object detection (autonomous driving, retail self-checkout), segmentation (portrait mode on phones, tumour outlining), face/biometric verification, image generation (Midjourney, Stable Diffusion), reverse-image search (Google Lens). In this guide, every module is a different CV task.

**Related terms.**
- **Deep learning (DL)** — ML with deep neural nets; the backbone of modern CV.
- **CNN (Convolutional Neural Network)** — the workhorse architecture for images; covered in this module.
- **Vision Transformer (ViT)** — newer alternative that replaces convolutions with self-attention; not covered in depth here.
- **Classical CV** — the pre-DL era (OpenCV filters, SIFT, HOG). Still useful for preprocessing.

**Gotcha.** CV ≠ image processing. Image processing (blur, threshold, edge filter) is rule-based pixel manipulation; CV *learns* the right operation from data.

#### Image as tensor

> **🪜 Mental model:** *A photo is a stack of numbers.* Height × Width × Channels — each cell is one pixel's intensity in one colour.

**What it is.** Inside the computer, an image is a **3-dimensional array** (called a **tensor** when it sits in a deep-learning framework). For a colour image of size 224×224, the tensor has shape `(H, W, C) = (224, 224, 3)` — 224 rows, 224 columns, and 3 channels (red, green, blue). Each entry is an integer pixel value in `[0, 255]` (8-bit unsigned) or a float in `[0, 1]` after normalisation. A *batch* of `B` images is 4-D: `(B, H, W, C)` in Keras/TensorFlow, or `(B, C, H, W)` in PyTorch.

**Why it matters.** Every CV model expects a tensor of a specific shape and value range. Mismatches here are the #1 silent bug — the model trains but produces nonsense because pixels are 0–255 when the model expected 0–1. Interviewers always probe "what's the shape of an RGB image?" because half of candidates answer with the wrong axis order.

**How it works.** When you load an image with `cv2.imread` or `PIL.Image.open`, the library reads the file (PNG, JPEG, etc.), decompresses it, and returns a NumPy array. From there it's just an ndarray — you can slice, reshape, and feed it to a model. To put it on the GPU, frameworks convert NumPy arrays to their own tensor type (`tf.Tensor`, `torch.Tensor`).

**Where it's used.** Every CV pipeline starts with "load image → resize → normalise → batch → model." `Rescaling(1./255)` (Keras) or `transforms.ToTensor()` (PyTorch) does the 0–255 → 0–1 conversion. Pretrained models add a model-specific mean/std normalisation on top.

**Related terms.**
- **Tensor** — multidimensional array; basically a NumPy `ndarray` that can live on a GPU.
- **Channel** — one colour plane; see next entry.
- **Batch** — multiple images stacked along a new leading axis for efficient parallel processing.
- **Channels-last (`HWC`) vs channels-first (`CHW`)** — Keras default vs PyTorch default; same data, transposed layout.

```python
import cv2
img = cv2.imread('cat.jpg')                # (H, W, 3), dtype uint8, BGR order
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
img = img.astype('float32') / 255.0        # normalise to [0, 1]
img = img[None]                            # add batch axis → (1, H, W, 3)
```

**Gotcha.** OpenCV reads images as **BGR**, not RGB. Forget to convert and your network will produce subtly wrong predictions for months.

#### Channel (RGB, grayscale)

> **🪜 Mental model:** *Each channel is a black-and-white snapshot of one colour.* A colour image is 3 stacked snapshots: red intensity, green intensity, blue intensity.

**What it is.** A **channel** is one of the parallel 2-D layers that make up an image tensor. **Grayscale** images have 1 channel — each pixel is one intensity number. **RGB** images have 3 channels (red, green, blue). Some pipelines use **RGBA** (4 channels, with an alpha/transparency channel) or domain-specific channels like depth or thermal. The number of channels is the third dimension of the tensor in HWC layout.

**Why it matters.** Convolutional filters operate **per channel**: a kernel for an RGB input has 3 weights per spatial position (one per colour), so the same kernel sees all colours at once. Forgetting that grayscale data is single-channel will lead to a shape mismatch when you try to feed it to a pretrained RGB model.

**How it works.** Each channel is just a `(H, W)` 2-D array. They sit stacked along the channel axis. When you display a colour image, the renderer combines the three channels into the final pixel colour via additive light mixing.

**Where it's used.** `keras.layers.Conv2D(filters=16, ...)` builds 16 filters, each with `Cin` channels matching the input. Medical X-ray models often have `C=1` (grayscale) input. Satellite imagery has 10+ channels (visible + infrared bands).

**Related terms.**
- **RGB / BGR** — same data, different channel order.
- **Alpha channel** — transparency (used in PNG, not for ML normally).
- **Grayscale** — single intensity channel.
- **Multispectral / hyperspectral** — 10+ channel images (satellite, scientific).

**Gotcha.** A grayscale image stored as `(H, W)` (rank-2) won't feed into a CNN that expects `(H, W, 1)` (rank-3). Add the axis: `img[..., None]`.

#### Convolution operation

> **🪜 Mental model:** *Sliding stamp.* A small grid of weights is pressed onto every patch of the image; at each position you multiply-and-sum to produce one output number.

**What it is.** A **convolution** in deep learning is a *sliding dot-product*: take a small `kH × kW` patch of the input, multiply element-wise by a learnable **kernel** (same size), sum all the products into a single number — this is one output pixel. Slide the kernel across the whole image, computing one number per position, to produce an output 2-D grid called a **feature map** (or activation map). For multi-channel inputs, the kernel has matching `Cin` channels — you sum across channels too.

**Why it matters.** Convolution is the *single most important* operation in deep learning for vision. It captures three biases that match how images work: (a) **local patterns matter** (you only look at neighbours), (b) **the same pattern can appear anywhere** (you reuse the same kernel everywhere — called *weight sharing*), and (c) **complex patterns are built from simple ones** (stacking conv layers builds edges → textures → object parts → objects).

**How it works.**
1. Place the `kH × kW` kernel over the top-left patch of the input.
2. Compute the **dot product**: `sum_{i,j,c} kernel[i,j,c] * input[i,j,c]`. (For a 3×3 RGB kernel that's `3 × 3 × 3 = 27` multiplications and one sum.)
3. Add a learnable **bias** scalar.
4. Slide the kernel by `stride` pixels and repeat.
5. The collection of outputs forms one feature map. If you have `Cout` kernels, you get `Cout` feature maps stacked along the channel axis.

**Where it's used.** Every CNN layer (`keras.layers.Conv2D`, `nn.Conv2d` in PyTorch). The first conv layer of every pretrained backbone (ResNet, VGG, EfficientNet) is doing exactly this. In interviews "explain a convolution in 30 seconds" is a near-guaranteed opener.

**Related terms.**
- **Kernel / filter** — the learnable weight grid (next entry).
- **Feature map** — the 2-D output of one filter.
- **Stride** — step size when sliding.
- **Padding** — zeros added at the edges so the kernel can sit there too.
- **Cross-correlation** — what conv layers *actually* compute (math purists distinguish this from "true convolution" which flips the kernel; deep-learning practice ignores the distinction).

```python
import keras
from keras import layers
conv = layers.Conv2D(filters=16, kernel_size=3, strides=1,
                     padding='same', activation='relu')
# Apply to a batch of (B, 28, 28, 3) → (B, 28, 28, 16)
```

**Gotcha.** A 3×3 conv on a 3-channel input has `3·3·3 = 27` weights per filter, **not 9**. Forgetting the input-channel dim is the classic "I thought conv was 2D" mistake.

#### Kernel / filter

> **🪜 Mental model:** *A tiny pattern detector.* Each filter learns to fire when its specific pattern (an edge, a curve, a stripe) appears in a patch.

**What it is.** A **kernel** (also called a **filter**) is the small array of learnable weights used in one convolution. Shape: `(kH, kW, Cin)` — typically `3×3` or `5×5` spatially, with `Cin` matching the number of input channels. A single Conv2D layer has `Cout` filters; each filter scans the whole input and produces one feature map.

**Why it matters.** Filters are *what the network learns*. After training, low-level filters often look like edge or colour detectors; deeper-layer filters fire on textures, then object parts, then whole objects. Understanding "filter = learned pattern detector" is the intuition that makes CNNs make sense.

**How it works.** Weights start random; gradient descent nudges them to minimise loss. Over training, each filter ends up tuned to a specific pattern that the data wanted detected. You can visualise filters from a trained network — early VGG layers reveal Gabor-like edge filters arising spontaneously.

**Where it's used.** Specified by `kernel_size` in `Conv2D`. The pretrained filters of ImageNet models are what you re-use in transfer learning (Module 3).

**Related terms.**
- **Convolution** — the operation that uses the filter.
- **Feature map** — what a single filter outputs.
- **Weight sharing** — the property that the same filter is applied at every position.
- **Kernel size** — `(kH, kW)`; almost always odd (3, 5, 7) so the kernel has a single integer centre.

**Gotcha.** Even-sized kernels (2×2, 4×4) have no integer centre and produce alignment artefacts when stacked — always use odd kernels in practice.

#### Feature map

> **🪜 Mental model:** *A heat-map of where one pattern appeared.* Bright spots mean "this filter's pattern matched here strongly."

**What it is.** A **feature map** (or *activation map*) is the 2-D output of one filter sliding over an input. Its dimensions are `(H_out, W_out)`, determined by the input size, kernel size, stride, and padding (see the output-shape formula). One Conv2D layer with `Cout` filters produces `Cout` stacked feature maps, giving an output tensor of shape `(H_out, W_out, Cout)`.

**Why it matters.** Feature maps are how information flows through a CNN — every intermediate "image" the network sees is a stack of feature maps. Their spatial dimensions shrink as you go deeper (because of pooling/stride), but the channel count grows (more filters detecting more patterns). The final feature maps of a backbone are the **embeddings** used in transfer learning and similarity search (Modules 3–4).

**How it works.** Given input `(H, W, Cin)` and a `Conv2D(Cout, kH, kW)` layer with stride `S` and padding `P`, output shape is `(H_out, W_out, Cout)` where `H_out = (H + 2P - kH)/S + 1`. The values in each feature map are post-activation (ReLU) responses.

**Where it's used.** Inspecting `model.summary()` shows feature-map shapes at every layer. Skip connections in U-Net (Module 7) and ResNet (Module 3) carry feature maps from earlier layers to later ones. Object detectors (Modules 5–6) attach detection heads to backbone feature maps at multiple scales.

**Related terms.**
- **Activation map** — synonym.
- **Channel** — one feature map = one channel of the output tensor.
- **Receptive field** — the input region that influences a single output activation in a deep feature map.

**Gotcha.** Don't confuse "number of feature maps" (= output channels = `Cout`) with the spatial size of each map (= `H_out × W_out`).

#### Stride

> **🪜 Mental model:** *How big a step the stamp takes.* Stride 1 = touch every pixel; stride 2 = skip every other one and halve the output.

**What it is.** The **stride** is the step size (in pixels) the kernel moves between consecutive positions. Default stride is 1. Stride 2 means the kernel jumps two pixels at a time, producing an output roughly half the input's spatial size. You can have different strides per axis (`strides=(2, 2)`) but they're usually equal.

**Why it matters.** Stride is the cheap way to **downsample** inside a Conv2D — fewer output positions = fewer ops in the next layer. Modern architectures (ResNet, EfficientNet) often replace MaxPool with strided convolutions because strided convs are *learnable* downsamplers, while pooling is fixed.

**How it works.** With stride `S`, the kernel's anchor moves by `S` pixels per step. Output spatial size becomes `(H + 2P - kH) / S + 1` (integer division — fractions are floored).

**Where it's used.** `Conv2D(strides=2)` in any downsampling block. Detection backbones use strided convs to build the multi-scale feature pyramid (Module 6's FPN).

**Related terms.**
- **Pooling** — the alternative way to downsample, with no learnable parameters.
- **Dilation (atrous)** — sibling concept that *spaces out* the kernel taps instead of stepping; expands receptive field without downsampling.

**Gotcha.** A non-integer `(H + 2P - kH)/S + 1` is silently floored — your output shape may be smaller than you expected. Always sanity-check with `model.summary()`.

#### Padding (`'valid'` vs `'same'`)

> **🪜 Mental model:** *A photo frame of zeros.* Padding adds zero pixels around the image so the kernel can sit at the edges.

**What it is.** **Padding** is the practice of adding extra (usually zero) pixels around the input before convolving. Two common modes in Keras:
- **`'valid'`** — no padding. Output shrinks by `(kH - 1)` per layer.
- **`'same'`** — zeros are added symmetrically so that, *with stride 1*, the output has the same spatial size as the input.

PyTorch takes a numeric `padding=P` directly (in pixels per side).

**Why it matters.** Without padding, every conv layer shrinks the image — stack 10 of them and you've lost a lot of edge information. `'same'` padding lets you stack as many layers as you want while keeping the spatial size constant. It also lets edge pixels participate in convolutions instead of being ignored.

**How it works.** For a kernel size `kH` and `'same'` padding with stride 1, Keras adds `P = (kH - 1) / 2` zeros on each side. For stride > 1 the math is fancier and Keras may pad asymmetrically.

**Where it's used.** `'same'` padding is the default for every layer in U-Net, modern ResNet variants, and almost every CNN you'll build from scratch. `'valid'` shows up in older nets (LeNet) and in classifier heads where shrinking is desired.

**Related terms.**
- **Stride** — interacts with padding to determine output size.
- **Output-shape formula** — see below.
- **Zero-padding vs reflection padding** — the latter mirrors edge pixels instead of using zeros; sometimes better for image generation.

**Gotcha.** `'same'` does **not** guarantee equal output size when `stride > 1` — it preserves size only at stride 1. Always verify with the formula or `model.summary()`.

#### Conv2D output-shape formula `(W − K + 2P)/S + 1`

> **🪜 Mental model:** *Tape-measure rule.* Given the input width, kernel size, padding, and stride, count how many kernel positions fit.

**What it is.** For input spatial size `N`, kernel size `K`, padding `P` (per side), and stride `S`, the output spatial size is `O = ⌊(N − K + 2P) / S⌋ + 1`. This formula is the most-tested CV interview formula and the single biggest source of silent shape bugs. In words: *"(input + extra padding on both sides − kernel) divided by step, then add one for the starting position."*

**Why it matters.** Without it you can't predict the shape of any conv layer's output, can't budget memory, can't catch bugs before they crash training. Interviewers ask you to compute output shapes on the spot.

**How it works.** Each output position corresponds to one kernel placement. The first placement covers indices `0` through `K-1`. Each subsequent placement shifts by `S`. You can fit `⌊(N − K) / S⌋ + 1` positions without padding; padding `P` on each side adds `2P` to the effective input length, giving the formula above.

**Where it's used.** Every Conv2D, every MaxPool2D (they share the same formula). Always reach for it when designing a new architecture or debugging "why is my output 13×13 instead of 14×14?"

**Related terms.**
- **`'same'` padding** — chooses `P` to make `O = N` when `S = 1`.
- **`'valid'` padding** — `P = 0`, so `O = (N − K)/S + 1`.

```python
# 28×28 input, 3×3 kernel, padding=0, stride=1
# O = (28 - 3 + 0) / 1 + 1 = 26  → output is 26×26
```

**Gotcha.** The division is **floor division** — `(7 - 3)/2 + 1 = 3`, not 3.5. A non-divisible combo silently floors and you lose pixels.

#### Pooling (max vs average)

> **🪜 Mental model:** *Shrink-and-keep-the-loudest.* Take a `2×2` window, output one number (the max or the average), then move on.

**What it is.** **Pooling** is a fixed, non-learnable downsampling operation. The two common types: **MaxPooling2D** outputs the largest value in each pooling window; **AveragePooling2D** outputs the mean. Default `pool_size=2, strides=2` halves both spatial dims. Pooling has **zero learnable parameters** — it's deterministic given the input.

**Why it matters.** Pooling does three useful things at once: (1) reduces spatial size so deeper layers run faster, (2) introduces small **translation invariance** (a small shift of the input → roughly the same pooled output), (3) acts as a coarse regulariser by discarding fine-grained position info the network doesn't need.

**How it works.** Slide a non-overlapping `2×2` (typically) window across each feature map. MaxPool keeps `max(window)`; AvgPool keeps `mean(window)`. No backprop through the unselected positions in MaxPool — the gradient flows only through the max element.

**Where it's used.** Classical CNNs (VGG, LeNet) pool after every conv block. Modern ResNets often replace mid-network pooling with strided convs and use a *global average pooling* at the end. Pooling is also key in YOLO's feature pyramid and U-Net's encoder.

**Related terms.**
- **Strided convolution** — learnable alternative.
- **Global Average Pooling** — pools the whole feature map to one number per channel; covered in Module 2.
- **MaxUnpool / unpool** — the inverse used in some decoders.

**Gotcha.** MaxPool destroys information — once a value is discarded it's gone. Don't pool too aggressively early in the network or you'll lose detail you need later.

#### ReLU activation

> **🪜 Mental model:** *Half-wave rectifier.* Negative numbers → 0; positives pass through unchanged.

**What it is.** **ReLU** ("Rectified Linear Unit") is the non-linear activation function `f(x) = max(0, x)`. It's applied element-wise to a conv or dense layer's output. ReLU is the default activation in CV — almost every Conv2D layer is followed by a ReLU.

**Why it matters.** Without a non-linearity between layers, stacking conv layers is mathematically equivalent to a single linear layer (no extra expressive power). ReLU adds non-linearity *cheaply* (one comparison) and has a *non-saturating* gradient for positive inputs (gradient = 1), which fixes the **vanishing gradient** problem that plagued earlier networks using sigmoid or tanh.

**How it works.** Forward: output equals input if positive, else 0. Backward: gradient equals 1 if input was positive, else 0. Computationally trivial — just a `max(0, x)`.

**Where it's used.** Every Conv2D in a CNN, hidden layers in MLPs, ResNet/VGG/EfficientNet — all default to ReLU. Output layers don't use ReLU (they use softmax for classification or tanh/linear for regression).

**Related terms.**
- **Sigmoid** — the older non-linearity; saturates and vanishes the gradient. Avoid in hidden layers.
- **LeakyReLU** — `f(x) = x if x > 0 else 0.2·x`; fixes the **dead-neuron** problem (a ReLU stuck at 0 forever).
- **GELU / Swish** — smoother modern alternatives, mainly used in transformers.
- **Vanishing gradient** — the problem ReLU was invented to solve.

```python
layers.Conv2D(32, 3, padding='same', activation='relu')
# or, equivalently:
layers.Conv2D(32, 3, padding='same'); layers.ReLU()
```

**Gotcha.** "Dead ReLU": if a neuron's weights drift to where input is always negative, the gradient is permanently 0 and the neuron never learns again. Cure: He initialisation + reasonable LR, or use LeakyReLU.

#### Flatten layer

> **🪜 Mental model:** *Squash 2-D feature maps into a 1-D row.* Collapses spatial structure so a Dense layer can consume the result.

**What it is.** A **Flatten** layer reshapes a tensor `(B, H, W, C)` into `(B, H·W·C)` — keeping the batch dimension, collapsing everything else into a single vector per example. It has no parameters; it just rearranges memory (or a view).

**Why it matters.** Conv layers preserve spatial structure; **Dense (fully-connected)** layers don't — they expect a flat vector. So whenever you transition from the convolutional part of a network to the classifier head, you need a Flatten (or its modern replacement, GlobalAveragePooling2D — see Module 2).

**How it works.** Just a reshape: `tf.reshape(x, [batch_size, -1])`. No math, no parameters, no learning.

**Where it's used.** At the boundary between the conv backbone and the dense classifier head in any classical CNN. Increasingly replaced by `GlobalAveragePooling2D`, which produces a much smaller flat vector (one number per feature map) and so cuts the head's parameter count drastically.

**Related terms.**
- **Dense / Fully-connected layer** — takes the flat vector and produces classification logits.
- **GlobalAveragePooling2D** — modern alternative; averages each feature map to one number, replacing `Flatten + Dense(huge)`.

**Gotcha.** After Flatten, the vector size depends on the feature map shape. For a `(16, 16, 256)` feature map → 65 536 numbers; a `Dense(256)` on top adds 16.7M parameters — most of your model's weights. GAP fixes this.

#### Fully-connected (Dense) layer

> **🪜 Mental model:** *Every input talks to every output.* Each output unit is a learned weighted sum of *all* input units.

**What it is.** A **Dense** (also "fully-connected", "FC", "linear") layer maps a vector of `Nin` numbers to a vector of `Nout` numbers via `y = W·x + b`, where `W` is a learnable `(Nout × Nin)` weight matrix and `b` is a length-`Nout` bias vector. Typically followed by a non-linearity (ReLU in hidden layers, softmax at the output of a classifier).

**Why it matters.** Dense layers can mix information from *anywhere* in the feature vector — useful for the final classification step where the network combines all the high-level features into a class score. But they explode in parameter count on raw image inputs (one dense layer on `128×128×3` flattens to 49 152 inputs; a 1024-unit Dense atop that is 50M parameters).

**How it works.** A single matrix multiply plus a bias-add plus an activation. Backpropagation is straightforward: gradient w.r.t. weights is `δ · xᵀ`, gradient w.r.t. input is `Wᵀ · δ`.

**Where it's used.** The final 1–2 layers of every classifier (after the conv backbone): `Dense(256, 'relu') → Dropout → Dense(num_classes, 'softmax')`. Also the entirety of MLPs (Multi-Layer Perceptrons) — pre-deep-learning networks.

**Related terms.**
- **MLP** — a stack of only Dense layers; the original 1980s neural net.
- **Logit** — the raw pre-softmax output of the final Dense layer.
- **Softmax** — turns logits into a probability distribution by exponentiating and normalising.

**Gotcha.** Dense layers throw away spatial structure entirely — they're why MLPs are bad for raw images and CNNs are good.

#### Typical CNN architecture

> **🪜 Mental model:** *Funnel that gets narrower and deeper.* Spatial dims shrink layer by layer; channel count grows; ends in a few Dense layers that pick a class.

**What it is.** The canonical classification CNN, from input to output:
1. **Input** `(H, W, 3)`.
2. **Conv → ReLU → (BN) → Pool** — repeated 3–5 times. Each block roughly doubles `Cout` and halves `H, W`.
3. **Flatten** (or **GlobalAveragePooling2D**).
4. **Dense → ReLU → Dropout** — one or two hidden layers.
5. **Dense(num_classes, softmax)** — final classifier.

Examples: LeNet-5 (1998), AlexNet (2012), VGG-16 (2014) all fit this template. Modern nets (ResNet, EfficientNet) add residual connections and replace pooling with strided convs, but the funnel shape is preserved.

**Why it matters.** Recognising this template is half the battle when reading any CNN paper or code. Once you see it, every block has a predictable role: "extract spatial features → flatten → classify."

**How it works.** Early conv blocks detect low-level features (edges, colours, textures); deeper blocks combine these into parts and objects. Pooling/stride shrinks spatial extent to focus on global structure. The final Dense layers project the spatial summary into a class score.

**Where it's used.** Every from-scratch CNN you'll write in Module 1. Pretrained backbones in Modules 3+ follow the same shape — you reuse the conv stack and swap the head.

**Related terms.**
- **Backbone** — the conv stack used for feature extraction.
- **Head** — the final layers (often just Dense + softmax) that produce task-specific output.
- **VGG / ResNet / EfficientNet** — concrete architectures that follow the template.

```python
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
```

**Gotcha.** Don't pool too many times — if `H, W` reach `1×1` before the classifier, you've effectively run a GAP early and may have lost spatial signal you wanted.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what it is + when you reach for it*. Full guided introductions live in [the walkthrough above](#1g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Computer Vision** | The branch of AI where the input is an image/video; modern CV is built on CNNs trained end-to-end. | Any task whose input is pixels — classification, detection, segmentation, generation, retrieval. |
| **Image as tensor** | A 3-D array `(H, W, C)` of pixel intensities (0–255 uint8, or 0–1 float after rescaling). Batched data adds a leading axis. | Every CV pipeline starts here — load, resize, normalise, batch, then feed to the model. |
| **Channel** | One 2-D slice of an image — e.g., red, green, or blue. Grayscale = 1 channel, RGB = 3, RGBA = 4. | Pick C to match your data + model. Convert grayscale to 3-channel when feeding pretrained RGB nets. |
| **Convolution** | Slide a small kernel over the input, computing a dot product at every position to produce a feature map. | Every conv layer in every CNN — the core operation of deep learning for images. |
| **Kernel / filter** | A `(kH, kW, Cin)` block of learnable weights. `Cout` filters → `Cout` feature maps. | Pick odd `kH/kW` (3, 5, 7); `Cout` grows deeper in the network. |
| **Feature map** | The 2-D output of a single filter — a heat-map of "where this pattern matched." | Inspect via `model.summary()`; pass between conv blocks as the network's internal representation. |
| **Stride** | Step size of the kernel — stride 2 roughly halves output spatial dims. | Cheap downsampling inside a Conv2D; modern alternative to MaxPool. |
| **Padding** | Zeros added around the input so the kernel fits at the edges. `'same'` preserves size (at stride 1); `'valid'` shrinks. | `'same'` for almost every layer in modern CNNs; `'valid'` only when you want shrinking. |
| **Output-shape formula** | `O = (N − K + 2P)/S + 1`, floor division. | Run this in your head before every layer — the only way to predict shapes deterministically. |
| **Pooling** | Fixed, non-learnable downsampling — MaxPool keeps the strongest activation, AvgPool the mean. | Shrink spatial dims, get mild translation invariance, bound memory. Default `pool_size=2, strides=2`. |
| **ReLU** | Non-linear activation `max(0, x)` — non-saturating, no vanishing gradient for positives. | After every conv/dense hidden layer; the default activation in modern CV. |
| **Flatten** | Reshapes `(B, H, W, C)` → `(B, H·W·C)`. No parameters; just a memory rearrange. | At the boundary between the conv backbone and a Dense classifier head. |
| **Dense (FC) layer** | `y = Wx + b` — every input talks to every output. The classifier head's main ingredient. | Last 1–2 layers of a classifier; small enough vectors only (after Flatten/GAP). |
| **Typical CNN** | `Conv → ReLU → Pool` blocks (3–5 of them) → Flatten/GAP → Dense → Softmax. Spatial shrinks, channels grow. | The default architecture for an image-classification scratch model in Module 1. |
| **Param count (Conv2D)** | `(kH · kW · Cin + 1) · Cout` — kernel weights plus one bias, times the number of filters. | Use to budget memory, compare to Dense layers (which explode much faster). |

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

<a id="2g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction to every Module 2 concept. The recap cheat sheet below is for after you've read this once.

#### Overfitting

> **🪜 Mental model:** *Memorising the textbook, failing the real exam.* The model has learned the training images so specifically that it can't recognise anything slightly different.

**What it is.** **Overfitting** happens when a model fits the **training** data too well — including its noise and quirks — and consequently does worse on **unseen** data. The hallmark symptom: **training accuracy keeps climbing while validation accuracy stalls or drops** (a widening "train/val gap"). CNNs are particularly prone because they're high-capacity (millions of parameters) and CV datasets are often small relative to that capacity.

**Why it matters.** Overfitting is the #1 reason a model "works in development but fails in production." If you don't detect and counteract it, the test-set accuracy you report won't match what users see. Every CV interview probes "how do you know your model is overfitting and what do you do?" — get this wrong and you'll fail the round.

**How it works.** Training drives the loss on the training set down via gradient descent. With enough capacity and not enough regularisation/data, the model finds weights that *interpolate* the training data exactly — including pixel-level noise that doesn't generalise. The validation loss (computed on held-out data) starts climbing while training loss keeps falling.

**Where it's used.** Diagnosing every CNN you train. Plot `train_loss` and `val_loss` per epoch — a widening gap = overfitting alarm. The remedy ladder in this module exists *because* overfitting is so common.

**Related terms.**
- **Underfitting** — the opposite (both train and val are bad).
- **Generalisation** — the goal: low loss on data the model has never seen.
- **Regularisation** — any technique that limits the model's capacity to memorise (L1/L2, dropout, augmentation).
- **Train/val gap** — the diagnostic signal.

**Gotcha.** A small train/val gap doesn't always mean "all good" — both could be plateauing at a bad level (underfit). Always check absolute accuracy, not just the gap.

#### Underfitting

> **🪜 Mental model:** *Couldn't even pass the practice exam.* The model is too weak (or under-trained) to capture the structure in the data.

**What it is.** **Underfitting** is when the model performs poorly on the **training** set itself — capacity or training is insufficient to learn even the patterns visible in the data. Hallmark: both training and validation accuracy are low *and* not improving with more epochs.

**Why it matters.** Underfitting is the easier failure mode to fix (make the model bigger / train longer), but it's also easy to confuse with overfitting if you only look at val accuracy. The cure is the *opposite* of the overfitting cure — adding more dropout or augmentation will make things worse.

**How it works.** Three common causes: (1) model too small (not enough conv layers / filters), (2) too few training epochs (loss is still dropping when you stopped), (3) too much regularisation (you've handcuffed the model). LR being mis-tuned (too high or too low) is also a frequent culprit.

**Where it's used.** Every diagnostic plot — if train and val curves both flatline at low accuracy, it's underfit. The fix: bigger model, longer training, less regularisation.

**Related terms.**
- **Overfitting** — the opposite failure (train great, val bad).
- **Bias-variance tradeoff** — the formal framework relating the two.
- **Model capacity** — total expressive power; too little = underfit, too much (without regularisation) = overfit.

**Gotcha.** Don't increase regularisation when the model is underfit — you'll make things worse. First check training accuracy.

#### Bias-variance tradeoff

> **🪜 Mental model:** *Two ways to be wrong.* Bias = the model is wrong on average (too simple). Variance = the model is jumpy across different training sets (too complex).

**What it is.** Classical statistical learning decomposes prediction error into three parts: **bias²** (error from wrong assumptions / too simple a model), **variance** (error from sensitivity to the specific training data), and **irreducible noise**. **High bias** → underfit; **high variance** → overfit. The "tradeoff" is that reducing one often increases the other — a simpler model has high bias but low variance; a richer model the reverse.

**Why it matters.** It's the conceptual scaffolding for *everything* in this module. Dropout, L2, augmentation, early stopping — they all reduce variance at the cost of a small bias increase. Knowing this framework lets you reason about which lever to pull when training fails. In interviews, "explain bias-variance tradeoff" is a staple ML opener.

**How it works.** Imagine training the same model on many different samples of training data. **Bias** is the average error of the resulting predictions (how far the average prediction is from the truth). **Variance** is how much predictions vary from sample to sample. A high-capacity neural net has low bias but high variance; a tiny logistic regression has high bias but low variance.

**Where it's used.** Whenever you decide "should I add more layers (cut bias) or more regularisation (cut variance)?" Whenever you choose a model size for a given dataset size. Modern deep nets live in an over-parameterised regime where the classical curve doesn't fully apply (see "double descent"), but the intuition still holds.

**Related terms.**
- **Overfitting** — the high-variance failure.
- **Underfitting** — the high-bias failure.
- **Regularisation** — reduces variance at the cost of some bias.

**Gotcha.** Modern deep networks can have *both* low bias and low variance once over-parameterised (double descent) — don't take the classical curve too literally.

#### Train / validation / test split

> **🪜 Mental model:** *Three drawers.* Train = what the model fits on; Validation = what you tune hyperparameters on; Test = sealed until the very end, used once.

**What it is.** Splitting your labelled data into three disjoint subsets:
- **Train** (~70%) — gradient descent fits the weights on this.
- **Validation** (~15%) — used during training to monitor for overfitting and to pick hyperparameters (LR, model size, augmentation strength).
- **Test** (~15%) — kept sealed. Used **once** at the end to report an unbiased estimate of generalisation.

**Why it matters.** If you tune hyperparameters on the test set, you've contaminated it — your final number is optimistic. The validation set absorbs the "tuning noise" so the test set stays clean. Skipping this discipline is the most common cause of "great test numbers, embarrassing production performance."

**How it works.** Use `train_test_split` once to peel off the test set; split the rest into train + val. For small datasets, k-fold cross-validation replaces the val split. For time-series, splits must respect time order (no leakage from future to past).

**Where it's used.** Every supervised CV pipeline. `keras.utils.image_dataset_from_directory(..., subset='training')` plus `validation_split=0.2` is a standard one-liner.

**Related terms.**
- **Holdout** — the simple train/val/test approach.
- **Cross-validation** — k-fold rotation; used when data is scarce.
- **Data leakage** — when test info sneaks into training (e.g., normalising the test set's stats into the train preprocessor).

**Gotcha.** Random splits leak when the same identity appears in train *and* val (e.g., the same person in face data). Split by **identity**, not by image.

#### Dropout

> **🪜 Mental model:** *Random sparring partner.* Each training step you knock out a random subset of neurons; the network can't rely on any single one.

**What it is.** **Dropout** is a regularisation layer that, **during training only**, randomly sets a fraction `p` of the previous layer's activations to zero. At inference the dropout layer is a no-op (and the framework rescales activations by `1/(1−p)` internally so the expected magnitude matches training). Typical rates: 0.5 after Dense layers, 0.1–0.25 inside conv blocks.

**Why it matters.** Dropout is the *single most effective* regulariser for Dense layers. It forces the network to spread information across redundant neurons (no single neuron is reliable). The model behaves at inference like an **ensemble** of many random sub-networks — a free averaging effect that famously cut test error on AlexNet.

**How it works.**
1. At each training forward pass, sample a Bernoulli `(1−p)` mask the same shape as the activation.
2. Multiply activations by the mask (some go to zero).
3. Backprop only through the surviving neurons.
4. At inference, no mask, no rescaling needed (Keras handles the rescaling at training time).

**Where it's used.** `layers.Dropout(0.5)` after `Dense` layers in a classifier head. Less commonly inside conv blocks (typically `0.1–0.25`). Some segmentation networks skip Dropout entirely.

**Related terms.**
- **Bernoulli mask** — the random 0/1 array used to zero out neurons.
- **DropConnect** — drops weights instead of activations.
- **BatchNorm** — orthogonal regulariser; often used together with Dropout.
- **Inverted dropout** — the default scheme; rescales during *training* (not at inference).

```python
layers.Dense(256, activation='relu')
layers.Dropout(0.5)
layers.Dense(num_classes, activation='softmax')
```

**Gotcha.** Don't apply Dropout at inference time. Calling `model(x, training=True)` accidentally during evaluation is a classic stealth bug.

#### Batch normalization (BN)

> **🪜 Mental model:** *Re-centre the assembly line.* For each batch, shift and scale each channel so values look "standard normal" before passing to the next layer.

**What it is.** **Batch Normalization** is a layer that, for each channel, normalises activations across the batch (and spatial dims for conv layers) to have **zero mean and unit variance**, then applies a learnable affine transform `γ·x + β`. The `γ` and `β` are trained like any other weights; the mean and variance are computed *from the current batch* at training time and from a **running average** at inference time.

**Why it matters.** BN dramatically stabilises training: it allows higher learning rates, reduces sensitivity to initialisation, and provides a mild regularising effect (the batch-level noise acts like a tiny dropout). Modern CNNs train roughly 2–10× faster with BN than without. Almost every architecture from 2015 onward uses BN (or its sibling LayerNorm).

**How it works.**
1. Compute batch mean μ and variance σ² over the batch (per channel for conv).
2. Normalise: `x̂ = (x − μ) / √(σ² + ε)`.
3. Affine: `y = γ·x̂ + β`.
4. Update running mean/variance with exponential moving average.
5. At inference, use the running stats (not the batch's).

**Where it's used.** Every modern Conv2D block typically has a BN after the conv (`Conv → BN → ReLU` or `Conv → ReLU → BN`). ResNet, EfficientNet, DenseNet — all rely on BN. The exception: very small batches (size 1–2), where BN's statistics are too noisy → use **GroupNorm** or **LayerNorm** instead.

**Related terms.**
- **LayerNorm** — normalises across features, not the batch; used in transformers.
- **GroupNorm** — for small batches.
- **Running mean/variance** — the EMA used at inference.
- **Internal covariate shift** — the original motivation in the BN paper (now somewhat disputed).

```python
x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
```

**Gotcha.** BN at inference uses **running** stats, not batch stats. If you forget `training=False` (TF) or `.eval()` (PyTorch), your val/test metrics look wildly different from training because BN is using current-batch stats.

#### Data augmentation

> **🪜 Mental model:** *Free extra training data.* Show the same image multiple times with random flips/rotations/colour shifts — the network sees it as different.

**What it is.** **Data augmentation** applies random, label-preserving transformations to training images on the fly: horizontal flip, rotation, crop, brightness, contrast, colour jitter, translation, MixUp, CutMix, RandAugment, etc. Each epoch the model sees slightly different versions of each image, learning *invariance* to those transformations.

**Why it matters.** When you have 5 000 images (typical for a real-world CV project), augmentation is the biggest single accuracy boost you can get for almost no cost. It's also the cheapest cure for overfitting — every random crop is a "new" datapoint.

**How it works.** Apply random transforms inside the data pipeline so each batch sees different versions. The transformations must **preserve the label** — flipping a digit "6" makes it look like a "9", so flip-augmentation is a bad idea for MNIST but great for cats vs dogs.

**Where it's used.** `keras.layers.RandomFlip`, `RandomRotation`, `RandomCrop`, `RandomBrightness`, `RandomContrast` chained as a `Sequential` model and applied via `train_ds.map(augment)`. PyTorch uses `torchvision.transforms` or the more powerful **albumentations** library.

**Related terms.**
- **Test-time augmentation (TTA)** — augment at *inference* and average predictions; gives a small accuracy boost.
- **MixUp / CutMix** — modern augmentations that blend two images.
- **Albumentations** — the leading augmentation library for PyTorch.

```python
augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomCrop(120, 120),
])
train_ds = train_ds.map(lambda x, y: (augment(x), y))
# val_ds: NO augmentation — must reflect deployment distribution
```

**Gotcha.** Never augment validation or test data — augmentation distorts the eval signal. Also: not all transforms preserve labels (don't flip digits).

#### L1 / L2 regularization (weight decay)

> **🪜 Mental model:** *Penalty for being too big.* Add a "weights-are-large" tax to the loss; the optimiser keeps weights small to dodge the tax.

**What it is.** **Weight decay** adds a penalty term to the training loss proportional to the size of the model's weights:
- **L2 (Ridge):** `λ · Σ w²` — penalises *squared* weights; shrinks every weight smoothly toward 0 (but never exactly to 0).
- **L1 (Lasso):** `λ · Σ |w|` — penalises *absolute* weights; drives many weights *exactly* to 0 (produces sparse models / does feature selection).

`λ` (often called the regularisation strength) controls the tradeoff between fitting and shrinkage.

**Why it matters.** Smaller weights → smoother decision boundaries → better generalisation. L2 is the default regulariser in deep learning; L1 is rare for CNNs (used more in linear models for feature selection). Most modern optimisers (AdamW) bake in L2 weight decay directly.

**How it works.** During gradient descent, the gradient of the penalty (`2λw` for L2, `λ·sign(w)` for L1) is added to the gradient of the data-loss. The result: every step, weights are nudged toward zero, in addition to whatever the data-fitting gradient says.

**Where it's used.** `kernel_regularizer=regularizers.l2(1e-3)` argument to any `Conv2D` or `Dense`. In PyTorch, `optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)`.

**Related terms.**
- **AdamW** — variant of Adam that decouples weight decay correctly; preferred for most modern training.
- **Sparsity** — what L1 produces; many weights are exactly zero.
- **Ridge / Lasso** — the linear-regression names of L2 / L1.

**Gotcha.** L1 + Adam interacts badly — Adam's adaptive LR distorts the L1 penalty. Use AdamW (or SGD with weight decay) instead.

#### Early stopping

> **🪜 Mental model:** *Quit while you're ahead.* Watch the val loss; if it stops improving for a while, stop training and restore the best weights.

**What it is.** **Early stopping** is a callback that monitors a chosen metric (almost always `val_loss`) during training. If the metric hasn't improved for `patience` consecutive epochs, training halts. The best-weights-so-far are typically restored. It's the cheapest, simplest regulariser — it just stops when overfitting begins.

**Why it matters.** Without it, you have to manually guess how many epochs to train; the network often starts overfitting after some sweet spot and your final weights are worse than weights you had three epochs ago. With early stopping you set a generous epoch budget (100, say) and the callback picks the right time to stop.

**How it works.** After each epoch, the callback compares the new validation metric to the best so far. If better (by at least `min_delta`), update the best and reset the patience counter. If worse, increment the patience counter; when it reaches `patience`, halt and restore the saved-best weights.

**Where it's used.** Every Keras training loop:
```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                              min_delta=0.001, restore_best_weights=True)
```
Coupled with `ReduceLROnPlateau` and `ModelCheckpoint` it's the standard callback trio.

**Related terms.**
- **ModelCheckpoint** — saves model weights when val improves; partial substitute.
- **ReduceLROnPlateau** — reduces LR instead of halting.
- **`patience`** — how many bad epochs to tolerate before stopping.

**Gotcha.** Without `restore_best_weights=True`, you keep the *final* (worse) weights instead of the best ones — defeats the purpose.

#### Learning rate schedules

> **🪜 Mental model:** *Big steps early, baby steps later.* Start the optimiser with a large learning rate to explore, then shrink it for fine-grained convergence.

**What it is.** A **learning-rate (LR) schedule** changes the optimiser's learning rate over the course of training instead of keeping it fixed. Common schedules:
- **`ReduceLROnPlateau`** — multiply LR by `factor` (e.g., 0.3) when val loss stalls for `patience` epochs.
- **Step decay** — LR ×= 0.1 every 30 epochs.
- **Cosine annealing** — smooth cosine-shaped decrease over the whole training run.
- **Warmup** — start with a small LR and ramp up linearly for the first few hundred steps; common for large batch sizes and transformers.

**Why it matters.** A fixed LR is a compromise — too small ⇒ slow training, too large ⇒ unstable. Schedules give you both: aggressive early progress and precise late convergence. Modern recipes (super-convergence, OneCycleLR) combine schedules with momentum tuning for dramatic speed-ups.

**How it works.** A callback or optimiser wrapper queries the current epoch/step and computes the new LR. The optimiser uses this LR for its next update step.

**Where it's used.** Together with `EarlyStopping`. The standard 3-callback combo is:
```python
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5),
ModelCheckpoint(filepath, save_best_only=True),
```

**Related terms.**
- **Warmup** — gradually increase LR at the start to stabilise.
- **OneCycleLR** — cyclic schedule popular with super-convergence.
- **Optimizer** — the algorithm (SGD, Adam) that uses the LR.

**Gotcha.** A too-aggressive schedule (factor 0.1) early in training can stall the model in a bad local minimum. Start with `factor=0.3, patience=5` and adjust.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#2g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Overfitting** | The model fits training noise; train acc keeps rising while val acc stalls or drops. | Whenever your train/val gap widens — apply augmentation, dropout, L2, early stopping. |
| **Underfitting** | The model can't even fit the training set; both train and val are low. | Diagnose first: if train acc is low, *don't* add regularisation — make the model bigger. |
| **Bias-variance tradeoff** | Error decomposes into bias² (too simple) + variance (too jumpy) + noise. | Mental framework for picking levers — regularisation cuts variance, capacity cuts bias. |
| **Train / val / test split** | Three disjoint subsets — train fits, val tunes, test reports. Test is opened *once*. | Every supervised pipeline; never tune hyperparameters on the test set. |
| **Dropout** | Randomly zero a fraction `p` of activations during training only; disabled (just rescaled) at inference. | `p=0.5` after Dense, `0.1–0.25` after Conv. Best regulariser for fully-connected heads. |
| **BatchNorm** | Per-batch, per-channel normalisation + learnable affine `γ·x + β`. Running EMA at inference. | After almost every Conv2D in modern CNNs. Allows higher LR, faster convergence. |
| **Data augmentation** | Random label-preserving image transforms (flip, rotate, crop, jitter) applied at train time only. | Always on small datasets — single biggest accuracy boost for free. |
| **L2 (weight decay)** | Add `λ Σ w²` to loss → smoothly shrink weights toward 0 (but not to 0). | Default regulariser for CNNs; bake it in via `kernel_regularizer=regularizers.l2(1e-3)`. |
| **L1** | Add `λ Σ |w|` → drives weights *exactly* to 0 → sparse model / feature selection. | Rare for CNNs; common in linear models. |
| **Early stopping** | Halt training when val metric hasn't improved for `patience` epochs; restore best weights. | Every Keras training loop with `restore_best_weights=True`. |
| **LR schedule** | Change LR over time — `ReduceLROnPlateau`, step decay, cosine annealing, warmup. | When fixed LR plateaus; combine with EarlyStopping for "set and forget" training. |
| **GlobalAveragePooling2D** | Averages each feature map to a single value, replacing Flatten + huge Dense. | At the end of any modern CNN backbone — slashes parameter count by ~100×. |

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

<a id="3g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 3 concept. The recap cheat sheet below summarises after.

#### Transfer learning

> **🪜 Mental model:** *Don't relearn what someone else already mastered.* Start with weights that already know edges, textures, and object parts — only train the last few layers for your specific task.

**What it is.** **Transfer learning** is the practice of taking a neural network already trained on a large dataset and reusing its learned weights as the starting point for a new, related task. In CV that almost always means: start with a CNN pretrained on **ImageNet** (1.28 M images, 1 000 classes), throw away its 1 000-class classifier head, attach a new head for your task, and train. The pretrained backbone has already learned generic visual features (edges, textures, parts) that transfer to almost any image task.

**Why it matters.** Training a CNN from scratch needs ~10⁵–10⁶ labeled images plus days of GPU time. With transfer learning you can hit 80%+ accuracy on a custom task with **5 000 labelled images and 30 minutes of training**. It's the most impactful trick in practical CV — and one of the most common interview topics ("you have 1 000 images for a new task — what's your approach?").

**How it works.**
1. Pick a pretrained backbone (e.g., `tf.keras.applications.ResNet50(weights='imagenet', include_top=False)`).
2. **Freeze** the backbone weights (so they don't change during training).
3. Attach a fresh classifier head matched to your task (e.g., `GlobalAveragePooling2D → Dense(num_classes, softmax)`).
4. Train only the head with a normal LR (1e-3) until it converges.
5. *Optional fine-tuning step:* unfreeze the top conv blocks, train everything with a **10× smaller** LR (1e-5).

**Where it's used.** Almost every real-world CV project. The Module 3 notebook uses it to go from 12% test accuracy (from-scratch VGG) to 79% (pretrained VGG + frozen backbone + new head) on a 10-class landmarks task. In FAANG interviews, "fine-tune a pretrained model" is the default answer to any small-data CV problem.

**Related terms.**
- **Pretrained model** — the source model whose weights you reuse.
- **Backbone** — the convolutional body, minus the classifier (next entry).
- **Feature extraction** — variant where backbone is fully frozen.
- **Fine-tuning** — variant where top layers of the backbone are unfrozen and retrained.
- **Catastrophic forgetting** — what happens if you fine-tune with too large an LR — the model forgets ImageNet knowledge.

**Gotcha.** Domain mismatch matters — ImageNet → medical X-rays transfers poorly. Use a pretrained medical-imaging model (CheXNet) if you have the option.

#### Pre-trained model

> **🪜 Mental model:** *Borrow someone's gym progress.* They put in the millions-of-reps training; you get to start strong.

**What it is.** A **pre-trained model** is a neural network whose weights have already been learned on some (usually huge) dataset. In CV the canonical pretrained source is **ImageNet** (1 000 classes, 1.28M images), but you can also pretrain on JFT-300M (Google's billion-image private set), or on self-supervised tasks (SimCLR, MAE) where labels aren't needed. Pretrained weights are downloaded once and cached — `keras.applications.ResNet50(weights='imagenet')` will grab them automatically.

**Why it matters.** Pretrained weights encode an enormous amount of visual knowledge — they're the closest thing CV has to a "starter kit." Without them, anyone working on a small custom dataset would be stuck.

**How it works.** The original team trained the model on ImageNet for weeks. The resulting weights file (50–500 MB depending on the model) is uploaded to a model zoo (TensorFlow Hub, PyTorch Hub, HuggingFace). You download it once; the framework caches it locally. Loading the model + weights takes a few seconds.

**Where it's used.** As the starting point for almost every transfer-learning task. `from keras.applications import ResNet50, VGG16, EfficientNetB0, MobileNetV2`.

**Related terms.**
- **ImageNet** — the dataset most CV models are pre-trained on.
- **Model zoo** — collection of pretrained models (TF Hub, PyTorch Hub, HuggingFace).
- **Self-supervised pretraining** — no labels needed; SimCLR, BYOL, MAE.

**Gotcha.** Each pretrained model expects its **own preprocessing**. VGG subtracts ImageNet mean (in BGR order!); ResNet uses a different mean; EfficientNet uses `/255` then a divisor. Use `<model>.preprocess_input` — never just `/255`.

#### Backbone

> **🪜 Mental model:** *The skeleton without the hat.* All the conv layers, but without the final classifier — ready for you to attach a new head.

**What it is.** The **backbone** of a CNN is the convolutional portion — typically all layers up to (but not including) the final classifier head. In code this is what you get from `keras.applications.ResNet50(weights='imagenet', include_top=False)` — the `include_top=False` strips the 1 000-class output layer, leaving the backbone. The backbone's output is a feature map (e.g., `(7, 7, 2048)` for ResNet-50 at 224×224 input).

**Why it matters.** The backbone is where 99% of the parameters and ImageNet knowledge live. The classifier head is a couple of Dense layers — trivial to retrain. So when adapting to a new task, you keep the backbone and replace the head.

**How it works.** Conceptually identical to a regular CNN — Conv → BN → ReLU → Pool blocks stacked into a deep funnel. The specific shape depends on the architecture (50 layers for ResNet-50, 16 for VGG-16, etc.). You can call the backbone like a function on input: `features = backbone(x, training=False)`.

**Where it's used.** As the frozen feature extractor in transfer learning. As the embedding extractor in image similarity (Module 4, where the backbone's output is the embedding). As the feature provider for detection heads (Modules 5–6) and segmentation decoders (Module 7).

**Related terms.**
- **Head** — the task-specific layers atop the backbone (classifier, detector, mask head).
- **Stem** — the very first conv block of a backbone (often distinct, e.g., 7×7 conv in ResNet).
- **`include_top=False`** — Keras flag that strips the classifier.

**Gotcha.** Different backbones have different output shapes and channel counts. Always run `backbone.summary()` and verify the output shape before attaching a head.

#### Feature extraction (frozen backbone)

> **🪜 Mental model:** *Use the pretrained eyes; learn only a new tongue.* Backbone is frozen — only the new head trains.

**What it is.** **Feature extraction** is the simpler transfer-learning recipe: the entire pretrained backbone is **frozen** (its weights don't update during training), and only the new classifier head is trained. Effectively the backbone is acting as a fixed feature extractor — it consumes images and produces embeddings, and you train a small classifier on top of those embeddings.

**Why it matters.** It's fast (no backprop through the huge backbone), it's safe (no risk of destroying ImageNet features), and it's the right starting point when your dataset is small (1 000–10 000 images). Get this working before you fine-tune.

**How it works.**
1. Load pretrained backbone with `include_top=False`.
2. Set `base.trainable = False`.
3. Stack new layers on top: GAP → Dense(softmax).
4. Compile and train as normal; only the new layers update.

**Where it's used.** As the default first step of any transfer-learning project. The Module 3 notebook achieves 79% test accuracy on landmarks using *only* feature extraction (no fine-tuning).

**Related terms.**
- **Fine-tuning** — sibling technique that unfreezes some layers.
- **Embedding** — the backbone's output, used as input to the new head (and also for similarity in Module 4).
- **`base.trainable = False`** — the Keras one-liner.

```python
base = keras.applications.VGG16(weights='imagenet', include_top=False,
                                 input_shape=(224, 224, 3))
base.trainable = False
model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax'),
])
```

**Gotcha.** Even with the backbone frozen, BatchNorm layers can still update their running stats. Pass `training=False` when calling the backbone, or set `base.trainable = False` *before* compile.

#### Fine-tuning

> **🪜 Mental model:** *Adjust the tail of the funnel.* Unfreeze the last few conv blocks (the most task-specific ones) and train them gently with a tiny LR.

**What it is.** **Fine-tuning** is the second-stage transfer-learning recipe: after feature extraction has trained the new head, unfreeze the top few backbone layers and continue training with a **10× smaller learning rate** (e.g., `1e-5` instead of `1e-3`). The bottom layers (edges, textures — universal) stay frozen; only the top, task-specific layers adapt to your data.

**Why it matters.** Fine-tuning typically gains another 2–10% accuracy over feature extraction alone — useful when you have enough data (5 000+) and need every point. But it's risky: too-large LR can **catastrophically forget** ImageNet features, leaving you worse off than feature extraction.

**How it works.**
1. After the head has converged via feature extraction, unfreeze top N layers: `base.trainable = True; for layer in base.layers[:-N]: layer.trainable = False`.
2. **Re-compile** with a much smaller LR (`Adam(1e-5)`).
3. Continue training for a few more epochs.
4. Monitor val loss carefully — if it gets worse, you've over-fine-tuned; back off.

**Where it's used.** When you have enough data and feature extraction has plateaued. Modern recipes (discriminative LR per layer, `fastai`'s `learner.fine_tune`) automate the details.

**Related terms.**
- **Catastrophic forgetting** — losing pretrained knowledge due to too-large fine-tuning LR.
- **Discriminative learning rates** — using different LRs for different depths (smaller at the bottom, larger at the top).
- **Layer-wise unfreezing** — gradual unfreezing schedule popularised by ULMFiT.

**Gotcha.** **Always do feature extraction first, then fine-tune.** Unfreezing on day one — when the head is still random — back-propagates large gradients through the backbone and destroys the pretrained weights.

#### ImageNet

> **🪜 Mental model:** *The CV pretraining dataset.* 1.28M labelled images over 1 000 categories — what almost every pretrained CNN started from.

**What it is.** **ImageNet** is a large image dataset organised by WordNet noun synonyms. The version used in CV is **ILSVRC-2012**: ~1.28M training images, 50 000 validation, 100 000 test, distributed across 1 000 fine-grained classes (specific dog breeds, mushroom types, etc.). "ImageNet-pretrained" almost always means trained on this 1 000-class subset.

**Why it matters.** ImageNet's diversity (1 000 classes covering animals, objects, scenes) makes a model pretrained on it a great general-purpose visual feature extractor. Features that distinguish 'husky' from 'malamute' are precisely the kind that generalise to your custom landmark or product-classification task.

**How it works.** The ILSVRC competition (2010–2017) drove the field — AlexNet (2012) was the breakout, ResNet (2015) reached human-level on the test set. Every architecture in `keras.applications` was originally trained on this dataset.

**Where it's used.** As the source domain for almost all CV transfer learning. State-of-the-art top-1 accuracies on ImageNet-1K (~88% for modern ConvNeXt/ViT) are still the standard benchmark.

**Related terms.**
- **ILSVRC** — the official competition / benchmark.
- **Top-1 / top-5 accuracy** — fraction of test images where the correct class is the top prediction / among top 5.
- **JFT-300M** — Google's private 300M-image dataset used for newer pretraining.

**Gotcha.** ImageNet skews heavily toward animals, objects, and scenes. For specialised domains (X-rays, satellite, fashion) you may want a domain-specific pretrained model.

#### Popular backbones (VGG, ResNet, Inception, EfficientNet)

> **🪜 Mental model:** *A menu of pretrained backbones — pick the one that fits your speed / accuracy / size tradeoff.*

**What they are.** A short tour of the most common pretrained CNNs you'll see in interview questions and real projects:
- **VGG-16 / VGG-19** (2014) — uniform stacks of 3×3 convolutions; simple, but 138M parameters (huge).
- **Inception (GoogLeNet, v1–v3)** (2014–2016) — parallel multi-scale conv blocks (1×1, 3×3, 5×5 in parallel); parameter-efficient (~11M for v1).
- **ResNet (18 / 50 / 101 / 152)** (2015) — adds **residual skip connections** `out = F(x) + x`; enables training of much deeper networks. The default modern backbone.
- **EfficientNet (B0–B7)** (2019) — uses **compound scaling** (jointly scales depth, width, resolution); best accuracy/parameter tradeoff at most sizes.
- **MobileNet (V1–V3)** (2017–2019) — designed for mobile devices; uses **depthwise-separable convolutions** for ~10× fewer FLOPs.

**Why it matters.** Picking the right backbone is a 30-second decision that drives everything downstream. ResNet-50 is the safe default; EfficientNetB0 wins on accuracy/size; MobileNet is required on phones; VGG is mostly historical now.

**How they differ at a glance.**

| Backbone | Year | Params | Top-1 (ImageNet) | Notable trick |
|---|---|---|---|---|
| VGG-16 | 2014 | 138M | 71.3% | Stacked 3×3 convs |
| Inception-v3 | 2015 | 24M | 78.0% | Parallel multi-scale |
| ResNet-50 | 2015 | 26M | 76.0% | Residual skips |
| EfficientNet-B0 | 2019 | 5.3M | 77.1% | Compound scaling |
| MobileNet-V2 | 2018 | 3.5M | 71.8% | Depthwise-separable |

**Where they're used.** Every transfer-learning project picks one of these. `keras.applications.<Name>(weights='imagenet', include_top=False)`. ResNet-50 is the universal default; EfficientNetB0 when you want fewer parameters at similar accuracy.

**Related terms.**
- **Residual connection** — ResNet's key idea (`out = F(x) + x`).
- **Depthwise-separable conv** — MobileNet's trick: split the conv into per-channel + 1×1 cross-channel.
- **Compound scaling** — EfficientNet's principle: scale depth, width, and resolution jointly.
- **Inception module** — multi-scale parallel branches.

**Gotcha.** Each backbone expects a specific input size (224 for ResNet/VGG, 299 for Inception, 224–600 for EfficientNet variants). Mismatch silently breaks the pretrained weights.

#### Learning-rate strategy for fine-tuning

> **🪜 Mental model:** *Big LR for new layers, tiny LR for borrowed ones.* The head needs fast learning; the backbone needs gentle adjustment.

**What it is.** The recipe for choosing learning rates during fine-tuning. **Discriminative learning rates** assign different LRs to different parts of the network: typically `1e-3` for the new head, `1e-5` (10–100× smaller) for the unfrozen backbone layers. With a single optimiser you usually just use the smaller LR for all unfrozen params — fine for most projects.

**Why it matters.** Mis-tuning LR is the #1 cause of fine-tuning disasters. Too large → the backbone's ImageNet features get destroyed (catastrophic forgetting). Too small → no actual fine-tuning happens; the backbone barely budges.

**How it works.**
1. Feature-extraction stage: train new head at normal LR (`1e-3`).
2. Fine-tuning stage: unfreeze top N backbone layers, switch optimiser to `Adam(1e-5)`, recompile, continue.
3. Optional: use cosine annealing or `ReduceLROnPlateau` to gently decay further.

**Where it's used.** Every fine-tuning workflow. `keras.optimizers.Adam(1e-5)` is the standard fine-tune LR. PyTorch/`fastai` uses `learner.fine_tune` with a `slice(1e-5, 1e-3)` discriminative-LR pattern.

**Related terms.**
- **Catastrophic forgetting** — what too-large LR causes.
- **Discriminative LRs** — per-layer LRs; smaller at bottom, larger at top.
- **Warmup** — increase LR slowly for the first few hundred steps; helps avoid early instability.

**Gotcha.** When you switch optimisers/LRs between feature-extraction and fine-tuning, you **must** recompile the model — otherwise it keeps the old LR.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#3g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Transfer learning** | Reuse a model pretrained on a huge dataset (ImageNet) as the starting point for your task. | Default approach for any CV task with <100k labelled images; ~10× faster + much higher accuracy. |
| **Pretrained model** | A network whose weights are already learned (e.g., ResNet50-imagenet). Auto-downloaded by Keras. | As the starting point for almost every CV project. |
| **Backbone** | The convolutional portion of a pretrained model — strip the classifier with `include_top=False`. | Wherever you want pretrained features without the original 1000-class head. |
| **Feature extraction** | Freeze the entire backbone; train only a new classifier head. Fast, safe, the default first step. | Small datasets (1k–10k), quick iteration, before fine-tuning. |
| **Fine-tuning** | Unfreeze the top few backbone layers and train them with a 10× smaller LR. | After feature extraction has converged, when you need 2–10% more accuracy. |
| **ImageNet** | 1.28M-image, 1000-class dataset that pretrained CV models started from. | The canonical "source" dataset for transfer learning; assume "pretrained" = "ImageNet" unless told otherwise. |
| **VGG** | 2014 architecture, uniform 3×3 conv stacks, 138M params. Simple, slow, historical. | Mostly reference / teaching; rarely a production choice. |
| **ResNet** | Residual connections (`out = F(x) + x`) enable very deep nets. Default modern backbone. | When in doubt, start with ResNet-50. |
| **Inception** | Parallel multi-scale convs (1×1, 3×3, 5×5) — efficient (~11–24M params), 299×299 input. | When you want strong accuracy at a moderate parameter count. |
| **EfficientNet (B0–B7)** | Compound-scaled depth/width/resolution. Best accuracy/parameter tradeoff at most sizes. | When you want SOTA accuracy with constrained model size. |
| **MobileNet** | Depthwise-separable convs; designed for phones / edge. ~3M params. | Mobile or edge deployment, real-time constraints. |
| **Fine-tune LR** | 10× smaller than scratch LR (typically `1e-5`); recompile when switching from extraction to fine-tune. | Always smaller, always after head converges. Use `ReduceLROnPlateau` to gently decay further. |

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

<a id="4g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 4 concept. The recap cheat sheet below summarises after.

#### Embedding

> **🪜 Mental model:** *Image as a coordinate.* Each image becomes a fixed-length vector (its "address") in a high-dimensional space; similar images land near each other.

**What it is.** An **embedding** is a fixed-length dense vector (typically 128–2048 numbers) that summarises an input — here, an image — in a way that makes "semantic similarity = vector closeness." Embeddings are usually extracted from the **penultimate layer** (the one just before the classifier) of a pretrained CNN — that layer holds rich, object-level features without the final class-collapsing softmax. For ResNet-50 the embedding dim is 2 048; for VGG-16 it's 4 096; for MobileNet-V2 it's 1 280.

**Why it matters.** Embeddings unlock everything that classification can't: similarity search, clustering, near-duplicate detection, face verification, recommendations. They let you ask "which image is most like this one" instead of "which class is this." Almost every modern retrieval / recommendation system uses learned embeddings.

**How it works.**
1. Load a pretrained CNN with `include_top=False, pooling='avg'`.
2. Run an image through: `e = backbone.predict(preprocess(img))` → a 2 048-D vector.
3. (Optional) L2-normalise: `e = e / ‖e‖`.
4. Store the embedding alongside the image ID.
5. At query time, compute the query's embedding and find nearest neighbours in the stored set.

**Where it's used.** Reverse image search (Google Lens, Pinterest). Face verification (FaceNet stores one embedding per identity). Product recommendation by visual similarity. Deduplication (find near-duplicates by embedding distance). Module 8's Siamese networks *train* embeddings explicitly for similarity.

**Related terms.**
- **Embedding space / latent space** — the high-dimensional space embeddings live in.
- **Penultimate layer** — the source of pretrained embeddings.
- **L2-normalisation** — projects embeddings to the unit sphere; required to make cosine ≡ Euclidean.
- **Word embedding** — same idea in NLP (word2vec, BERT [CLS] vector).

```python
backbone = keras.applications.ResNet50(weights='imagenet',
                                        include_top=False, pooling='avg')
emb = backbone.predict(preprocess(img))     # shape (1, 2048)
emb = emb / np.linalg.norm(emb)             # L2-normalise
```

**Gotcha.** Embeddings from pretrained classifiers are *good but not optimal* for similarity — they were trained to *classify*, not to cluster. For best results, train a Siamese / metric-learning model (Module 8).

#### Embedding space / latent space

> **🪜 Mental model:** *A huge atlas where each image is a pin.* "Near each other" in the atlas means "look similar"; "far apart" means "look different."

**What it is.** The **embedding space** (also called **latent space** or **feature space**) is the high-dimensional vector space in which embeddings live. For ResNet-50 it's ℝ²⁰⁴⁸. The structure of this space — how points are arranged — encodes what the network learned. After training, semantically similar images cluster together; semantically distinct images are separated.

**Why it matters.** The whole reason embeddings work is that this space has *useful structure*. If embeddings of the same object class form a tight cluster, then nearest-neighbour search finds you images of the same class. If embeddings instead were random, retrieval would be useless. Visualising the space (t-SNE/UMAP to 2D) often reveals interpretable structure: cars cluster, dogs cluster, beaches cluster.

**How it works.** During pretraining, gradient descent pushed embeddings of correctly-classified images apart by class while pulling them together within a class — a side-effect of optimising cross-entropy. The resulting geometry happens to be good for similarity even though that wasn't the explicit goal.

**Where it's used.** Anywhere you talk about "vector databases" (Pinecone, Weaviate, Qdrant) or "semantic search". Module 4's brute-force / Annoy / FAISS searches all happen inside this space.

**Related terms.**
- **Embedding** — a single point in the space.
- **Latent vector `z`** — sibling concept in GANs (Module 9) — a noise vector in *generative* latent space.
- **Manifold hypothesis** — the idea that real data lies on a low-dimensional manifold inside the high-dim space.

**Gotcha.** Don't visualise embedding space in 2D and assume the distances you see are accurate — t-SNE distorts distances dramatically (see below).

#### Cosine similarity vs Euclidean distance

> **🪜 Mental model:** *Angle vs straight line.* Cosine asks "do these point in the same direction?"; Euclidean asks "are these points close?"

**What they are.** Two ways to measure how similar two vectors are.
- **Cosine similarity:** `cos(θ) = (x · y) / (‖x‖ · ‖y‖)` — the cosine of the angle between vectors `x` and `y`, ignoring their magnitudes. Range `[-1, 1]`; 1 = identical direction. In words: "divide the dot product by the product of lengths."
- **Euclidean (L2) distance:** `‖x − y‖₂ = √(Σ (x_i − y_i)²)` — the straight-line distance in the vector space. Range `[0, ∞)`; 0 = identical.

**Why it matters.** Picking the wrong metric for your data is a common bug. Pretrained-CNN embeddings have varying magnitudes (a bright image's embedding is larger), so raw Euclidean distance mixes "different content" with "different brightness." L2-normalising fixes this — and after L2-normalisation, **cosine and Euclidean rank neighbours identically** (mathematically, `‖x − y‖² = 2 − 2 cos(θ)` on the unit sphere). In code, pick whichever your NN library prefers.

**How they differ before normalisation.** Cosine compares only direction; Euclidean compares both direction and magnitude. With L2-normalised vectors (magnitude = 1), the two are monotonic-equivalent.

**Where they're used.** Sklearn's `NearestNeighbors(metric='cosine'|'euclidean')`. Annoy's `'angular'` metric (which is cosine on the inside). FAISS provides both `IndexFlatL2` and `IndexFlatIP` (inner product, equivalent to cosine on normalised vectors).

**Related terms.**
- **Dot product** — `x · y`; cosine's numerator.
- **L2 norm** — `‖x‖₂ = √(Σ x_i²)`; the vector's length.
- **L2-normalisation** — dividing a vector by its L2 norm so it lies on the unit sphere.
- **Manhattan / L1 distance** — sibling metric `Σ |x_i − y_i|`; rarely used for embeddings.

**Gotcha.** "Angular distance" in Annoy is actually cosine-based, *not* the geometric angle in radians. Confusing naming — read the Annoy docs.

#### Nearest neighbour search

> **🪜 Mental model:** *Closest-pin-in-the-atlas.* Given a query embedding, find the `k` stored embeddings that are closest.

**What it is.** **Nearest neighbour (NN) search** is the operation "given a query point, return the `k` stored points closest to it by some metric." For embeddings, the metric is cosine or Euclidean. Brute-force NN compares the query against *every* stored point — `O(N · D)` per query for `N` items in `D` dimensions. For 100 000 ResNet-50 embeddings that's ~200M ops per query — still under a second on a modern CPU.

**Why it matters.** NN search is the algorithm under reverse-image search, face verification, deduplication, and content recommendation. It's also a frequent system-design question — "how would you scale image search to 1B images?" The answer is *not* brute force.

**How it works.** Brute force: compute distance from query to every stored point, sort, return top `k`. With `sklearn.NearestNeighbors` (which uses ball-trees or KD-trees in moderate dimensions) you get sub-linear time for low-dim data, but the curse of dimensionality kicks in for `D > ~20` and you're back to brute force. For high-dim embeddings, use **approximate nearest neighbour (ANN)** methods.

**Where it's used.** `sklearn.neighbors.NearestNeighbors(n_neighbors=5).fit(features)` for small / medium sets. For 100K+ items, switch to Annoy or FAISS (next entry).

**Related terms.**
- **k-NN classifier** — uses the same search but votes among neighbours for a label.
- **Approximate NN (ANN)** — sub-linear methods that trade a tiny accuracy loss for huge speedup.
- **Curse of dimensionality** — why tree methods fail in high dim.

**Gotcha.** Don't forget to **L2-normalise** stored and query embeddings before NN search if you want cosine semantics — otherwise you'll get magnitude-biased neighbours.

#### FAISS / Approximate nearest neighbours (ANN)

> **🪜 Mental model:** *Phone directory by region.* Instead of scanning every name, jump straight to the right region, then scan within.

**What it is.** **Approximate nearest neighbour (ANN)** algorithms trade a small accuracy loss (~0.1–1%) for a huge speedup (100–10 000×). Three families dominate:
- **Tree-based:** Annoy (Spotify), KD-trees. Builds a forest of random splits; queries traverse to find candidate leaves.
- **Hash-based:** Locality-Sensitive Hashing (LSH). Hashes embeddings into buckets so that close points collide.
- **Graph-based / quantization:** **FAISS** (Facebook) is the industrial-strength library — supports inverted indices (IVF), product quantization (PQ), and HNSW graphs; runs on CPU or GPU; scales to billions of vectors.

**Why it matters.** Brute force on 100M images is impossibly slow (~200s per query). FAISS gets it to milliseconds. Every production image-search system uses ANN under the hood (Pinterest, Google Lens, TikTok). System-design interviews ask "scale image search to 1B" — the answer is FAISS + sharding.

**How FAISS works (high level).** `IndexFlatL2` is brute force. `IndexIVFFlat` clusters embeddings into Voronoi cells; at query time it only searches the cells nearest the query — sub-linear. `IndexIVFPQ` additionally compresses embeddings via product quantization to fit billions in RAM. `IndexHNSW` builds a navigable small-world graph; very fast lookups.

**Where it's used.** Any retrieval pipeline above ~100K items. `annoy.AnnoyIndex(dim, 'angular')` for moderate-scale Python. `faiss.IndexIVFFlat(...)` for billion-scale GPU search.

**Related terms.**
- **Recall@k** — what fraction of true nearest neighbours the ANN returns; how you measure accuracy loss.
- **PQ (Product Quantization)** — embedding compression, e.g., 2048 floats → 64 bytes.
- **HNSW (Hierarchical Navigable Small World)** — graph-based ANN; current SOTA on most benchmarks.

```python
from annoy import AnnoyIndex
idx = AnnoyIndex(2048, 'angular')          # cosine on normalised vectors
for i, v in enumerate(features): idx.add_item(i, v)
idx.build(40)                              # 40 trees
idx.save('items.ann')
neighbours = idx.get_nns_by_vector(query, 10, include_distances=True)
```

**Gotcha.** Annoy's `'angular'` is cosine; `'euclidean'` is L2. FAISS's `IndexFlatIP` is inner product (which equals cosine on L2-normalised vectors). Mixing metrics silently produces wrong neighbours.

#### Image-retrieval pipeline

> **🪜 Mental model:** *Embed once, query forever.* Pre-compute embeddings for the gallery (offline). At query time, embed the query, search the index, return matches.

**What it is.** The standard end-to-end recipe for content-based image retrieval (CBIR):
1. **Offline (one-time):** Pick a pretrained backbone. For every image in the gallery, compute and L2-normalise the embedding. Optionally PCA-reduce to ~150 dims. Build an ANN index (Annoy/FAISS).
2. **Online (per query):** Take the query image, compute its embedding (same backbone, same preprocessing, same normalisation, same PCA). Query the index for top-`k` neighbours. Return the corresponding images.

**Why it matters.** This is the production architecture behind Google Lens, Pinterest visual search, TinEye, fashion-similarity systems, and "find more like this" features. Knowing the pipeline cold is a frequent CV system-design interview ask.

**How it works.** The offline step is parallelisable (embed batches on GPU); the online step is dominated by ANN search (sub-millisecond with FAISS). The full system can serve thousands of queries/sec from a single machine if you size the index right.

**Where it's used.** Reverse image search. Recommendation by visual similarity (Pinterest "Lens"). Near-duplicate detection in social media moderation. Deduplication in dataset curation.

**Related terms.**
- **Gallery / corpus** — the set of stored images to search against.
- **Query** — the input image whose neighbours you want.
- **CBIR (Content-Based Image Retrieval)** — academic name for the task.
- **PCA** — common dimensionality-reduction step inside the pipeline.

**Gotcha.** The query must use **identical preprocessing** to the gallery (same backbone, same `preprocess_input`, same normalisation). Mix them up and your "similar" images will be random.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#4g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Embedding** | A fixed-length dense vector summarising an image; pulled from the penultimate layer of a pretrained CNN. | Anywhere you need "is X similar to Y" — retrieval, clustering, dedup, face verification. |
| **Penultimate layer** | Layer just before the classifier; output is the embedding. | When extracting embeddings — get it via `include_top=False, pooling='avg'`. |
| **Embedding space** | The high-dim vector space embeddings live in. Similar inputs cluster; dissimilar ones spread. | Conceptual framing for all retrieval; visualise with t-SNE/UMAP (for *display* only). |
| **L2 normalisation** | Divide each embedding by its L2 norm; puts everything on the unit sphere. | Always normalise before NN search — makes cosine ≡ Euclidean and removes magnitude bias. |
| **Cosine similarity** | `(x · y) / (‖x‖ · ‖y‖)` — angle between vectors, magnitude-independent. | When magnitudes are irrelevant (text, sparse features, post-normalised image embeddings). |
| **Euclidean distance** | `‖x − y‖₂` — straight-line distance in the embedding space. | After L2-normalisation, equivalent to cosine; pick whichever your index supports. |
| **NN search** | Find the `k` stored points closest to a query in some metric. | The core operation of every retrieval system; `sklearn.NearestNeighbors` for small data. |
| **ANN (Annoy / FAISS)** | Approximate NN — trades ~0.1% accuracy for 100–10000× speedup. | Above ~100K items; production-scale retrieval; billion-vector FAISS for face systems. |
| **PCA** | Linear projection to keep top-`k` principal components; retains most variance. | Compress embeddings (2048 → 150) for ~13× speedup with <0.5% recall loss. |
| **t-SNE / UMAP** | Non-linear projection to 2D/3D for visualisation; distorts distances. | *Visualisation only* — never for retrieval, neighbourhoods aren't preserved metrically. |
| **Retrieval pipeline** | Embed gallery offline → build ANN index → online: embed query, search, return top-k. | Every CBIR system; ensure identical preprocessing between gallery build and query time. |

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

<a id="5g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 5 concept. The recap cheat sheet below summarises after.

#### Object detection (vs classification)

> **🪜 Mental model:** *Find them all, draw boxes, label each.* Classification = one label for the whole image; detection = many labels + box coordinates.

**What it is.** **Object detection** is the CV task of finding *all* instances of objects in an image and reporting both **what** each one is (a class label) and **where** it sits (a bounding box). Output for each detection: `(class_id, confidence, x1, y1, x2, y2)`. The image can contain 0, 1, or many objects, and the model must figure out how many there are.

Three related but increasingly hard tasks:
- **Classification** — "is there a dog in this image?" — one label.
- **Localization** — "where is the (single) dog?" — one label + one bbox.
- **Detection** — "where are all dogs and cats?" — many labels + many bboxes.

**Why it matters.** Almost every applied CV system beyond pure classification needs detection: self-driving cars find pedestrians and other vehicles, retail uses it for shelf monitoring, security uses it for intrusion. Detection is *the* defining CV task for 2015–2022 — every interview hour spent on CV will hit it.

**How it works.** Two architectural families:
- **Two-stage** (this module): *propose* candidate regions, then *classify* each.
- **Single-stage** (Module 6): predict everything in one forward pass.
Both predict `(class probs, bbox offsets)` per candidate, post-process with NMS to deduplicate, and evaluate with mAP.

**Where it's used.** Autonomous driving (Tesla, Waymo), retail analytics (Trax, Standard Cognition), security/surveillance, medical imaging (lesion detection), wildlife monitoring, robotics. In FAANG interviews, "design an object detector for X" is the most common CV system-design prompt.

**Related terms.**
- **Bounding box (bbox)** — the rectangle around an object (next entry).
- **IoU** — how detection match quality is measured.
- **NMS** — deduplication of overlapping detections.
- **Segmentation** (Module 7) — even denser: per-pixel labels instead of boxes.

**Gotcha.** Classification outputs are a single label; detection outputs are *lists* of (label, bbox, score). Different evaluation metrics, different loss formulations — don't conflate them.

#### Bounding box (bbox)

> **🪜 Mental model:** *A rectangle around the thing.* Defined by 4 numbers — either two corners, or a centre + size.

**What it is.** A **bounding box** is the axis-aligned rectangle that encloses an object. There are two common 4-number representations:
- **Corner format `(x1, y1, x2, y2)`** — coordinates of the top-left and bottom-right corners. Used by PASCAL VOC, COCO, torchvision.
- **Centre format `(xc, yc, w, h)`** — centre of the box plus width and height. Used by YOLO. Often **normalised** to [0, 1] by dividing by image dimensions.

**Why it matters.** Bbox-format bugs are the #1 cause of "my detector reports nothing useful." If you compute IoU between a YOLO-format box and a COCO-format box, the numbers are nonsense. Always confirm format before any operation on bboxes.

**How it works.** Given an image of shape `(H, W)`:
- Corner ↔ centre: `xc = (x1 + x2)/2; yc = (y1 + y2)/2; w = x2 - x1; h = y2 - y1`.
- Pixel ↔ normalised: divide all coords by `W` and `H` (and multiply to invert).
- All four numbers describe the same rectangle in both formats.

**Where it's used.** As ground-truth labels in training data (CVAT, LabelImg). As model outputs at inference. As input to IoU, NMS, and any downstream display step.

**Related terms.**
- **IoU** — overlap metric between two bboxes (next entry).
- **NMS** — uses IoU to deduplicate boxes.
- **Anchor box** — predefined reference bbox; the network predicts offsets from it.
- **Polygon / mask** — denser shape representation (Module 7).

**Gotcha.** When converting between formats, **double-check the axis order** — some libraries use `(y, x)` instead of `(x, y)`. Always print a sample box before computing anything.

#### Region proposal

> **🪜 Mental model:** *First guess, then refine.* Step 1: cheaply propose ~2 000 candidate boxes that *might* contain something. Step 2: the expensive network classifies each one.

**What it is.** A **region proposal** is a candidate bounding box generated *before* classification, indicating "something interesting might be here." Two-stage detectors generate ~300–2 000 proposals per image and then process each one. The early R-CNN family used a hand-crafted algorithm (**Selective Search**) that grouped similar pixels; Faster R-CNN (2016) replaced that with a learned **Region Proposal Network (RPN)** — much faster and end-to-end trainable.

**Why it matters.** Without proposals, the classifier would have to consider every possible bounding box in the image — millions of windows. Proposals cut that to a few thousand worth examining, making detection tractable. This idea is what made R-CNN possible in the first place.

**How it works (RPN flavour).**
1. Run a CNN over the image to produce a feature map.
2. At every position of the feature map, predict (objectness score, bbox offset) for each of `k` predefined anchor boxes.
3. Keep proposals with objectness > threshold; apply NMS to deduplicate.
4. Send the survivors (~300) to the second-stage classifier.

**Where it's used.** Inside every two-stage detector (R-CNN family). Not used in single-stage detectors (Module 6).

**Related terms.**
- **Selective Search** — the hand-crafted predecessor to RPN.
- **RPN (Region Proposal Network)** — Faster R-CNN's learned proposer.
- **Anchor box** — the reference box that each proposal is an offset from.
- **Objectness** — score saying "is there anything here?"

**Gotcha.** Proposals are *coarse* — they tell you "object likely here" but the bbox is approximate. The second stage's main job is refining the bbox in addition to classifying.

#### IoU (Intersection over Union)

> **🪜 Mental model:** *Overlap quality.* If two boxes mostly cover the same area, IoU is near 1; if they barely touch, IoU is near 0.

**What it is.** **IoU** (Intersection over Union, also called the Jaccard index for sets) is the standard measure of how well two boxes overlap:

`IoU = area(A ∩ B) / area(A ∪ B)`

i.e., *the area where the two boxes overlap, divided by the total area covered by either box*. Range `[0, 1]`. 0 = no overlap; 1 = identical boxes. Typical threshold for "the prediction matches the ground truth" is **IoU ≥ 0.5** (the PASCAL VOC convention).

**Why it matters.** IoU is *the* atomic metric of detection. It powers NMS (deduplication), mAP (the evaluation metric), and anchor-matching during training. Computing IoU correctly is a guaranteed coding question in any CV detection round.

**How it works.**
1. Compute the intersection: the rectangle where both boxes overlap. Width = `max(0, min(x2_A, x2_B) − max(x1_A, x1_B))`; height likewise.
2. Compute each box's area; the union area = `area_A + area_B − intersection`.
3. Divide.

```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-8)
```

**Where it's used.** NMS (drop boxes with IoU > 0.5 to the kept one). mAP at IoU thresholds (PASCAL@0.5, COCO@[0.5:0.05:0.95]). Anchor matching during training (an anchor is a "positive" for a ground-truth box if IoU > 0.7).

**Related terms.**
- **GIoU / DIoU / CIoU** — modern variants used as *loss functions*; ordinary IoU has zero gradient when boxes don't overlap.
- **mAP** — uses IoU as the matching criterion.
- **Jaccard index** — synonym from set theory.

**Gotcha.** Plain IoU is a fine *metric* but a poor *loss* — when boxes don't overlap at all, IoU = 0 and gradient = 0. Modern detectors use GIoU/DIoU/CIoU loss instead.

#### Non-Maximum Suppression (NMS)

> **🪜 Mental model:** *Survivor of overlaps.* Sort boxes by confidence, keep the best one, throw out any others that overlap it too much, repeat.

**What it is.** **NMS** is a post-processing step that deduplicates a list of overlapping detections so each object gets a single bbox in the output. Algorithm:
1. Sort candidates by confidence (highest first).
2. Take the highest-confidence box; add it to the output.
3. Drop all remaining boxes whose IoU with the kept box exceeds a threshold (e.g., 0.5).
4. Repeat from step 2 until no candidates remain.

**Why it matters.** Detectors typically fire many overlapping boxes for the same object — a YOLO grid might produce dozens of high-confidence boxes for one car. Without NMS your output has duplicates, your mAP tanks, and downstream consumers go crazy. Every interview round on detection asks you to write NMS pseudocode.

**How it works.** Greedy iteration. Per-class — run NMS separately for each class so a car and a pedestrian standing close together don't merge. Implementations (`torchvision.ops.nms`, `cv2.dnn.NMSBoxes`) are vectorised over boxes.

**Where it's used.** The very last step of every detector's inference pipeline (two-stage and single-stage alike). Default IoU threshold is 0.5, but for crowded scenes you'd tune it down.

**Related terms.**
- **Soft-NMS** — decays the confidence of overlapping boxes instead of zeroing them; helps when objects legitimately overlap (crowds of people).
- **DETR's bipartite matching** — modern transformer-based detectors skip NMS entirely.
- **IoU threshold** — the knob — lower = more aggressive dedup.
- **Class-aware NMS** — runs per-class, the default in practice.

**Gotcha.** Run NMS **per class**, not globally. A car and pedestrian close together can legitimately overlap and shouldn't be merged.

#### R-CNN / Fast R-CNN / Faster R-CNN lineage

> **🪜 Mental model:** *Three generations, each fixing the previous one's slowest part.*

**What they are.** The R-CNN family — the canonical two-stage detection lineage:
- **R-CNN (2014)** — Run Selective Search → 2 000 region proposals → push each through a CNN separately → SVM classifier on top. **Slow** (~47 s per image at inference).
- **Fast R-CNN (2015)** — Run the CNN *once* on the whole image → use **ROI Pooling** to extract per-region features from the shared feature map → single multi-task head (classifier + bbox regressor). **~10× faster** than R-CNN; ~2 s/image.
- **Faster R-CNN (2016)** — Replace Selective Search with a learned **RPN** sharing features with the detector → end-to-end trainable. **~5 fps** on COCO; finally practical.

**Why it matters.** Faster R-CNN is the canonical two-stage detector still in heavy use today (and the foundation of Mask R-CNN). Every CV interview probes the lineage — "explain how R-CNN evolved to Faster R-CNN" is a standard question.

**How they differ.**

| Method | Proposals from | CNN runs | Speed | mAP (VOC) |
|---|---|---|---|---|
| R-CNN | Selective Search | per-proposal | 47s | 66.0 |
| Fast R-CNN | Selective Search | shared whole-image | 2s | 70.0 |
| Faster R-CNN | RPN (learned) | shared whole-image | 0.2s | 73.2 |

**Where they're used.** `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` is two function calls away. Mask R-CNN (Module 7) extends Faster R-CNN with a mask head.

**Related terms.**
- **RPN (Region Proposal Network)** — Faster R-CNN's proposal stage.
- **ROI Pooling / ROI Align** — variable-size proposal → fixed-size feature.
- **Selective Search** — the hand-crafted predecessor.

**Gotcha.** "Fast R-CNN" and "Faster R-CNN" sound nearly identical but refer to different architectures. Get the names right in an interview — confusing them signals you read a blog post but not the papers.

#### RPN (Region Proposal Network)

> **🪜 Mental model:** *Mini-CNN that learns where to look.* A tiny network on top of the backbone outputs "objectness score + bbox refinement" at every position.

**What it is.** The **Region Proposal Network** is a small CNN inside Faster R-CNN that learns to generate region proposals end-to-end. It takes the shared feature map of the backbone (e.g., ResNet-50) as input, slides a 3×3 conv across it, and for every position predicts: (a) **objectness** scores for `k` predefined anchors (`k = 9` in standard Faster R-CNN: 3 scales × 3 aspect ratios), (b) **bbox refinement offsets** for each anchor. The proposals with highest objectness, post-NMS, are the candidates fed to the classifier head.

**Why it matters.** Replacing Selective Search (CPU-only, ~2s per image, non-trainable) with the RPN (GPU, ~10ms per image, end-to-end trainable) was the breakthrough that made Faster R-CNN possible. Today it's a default component of two-stage detectors.

**How it works.**
1. Backbone produces feature map of shape `(H', W', C)`.
2. RPN slides a 3×3 conv: at each spatial position, output `(k × 4 + k × 2)` numbers (4 bbox offsets + 2 obj/no-obj scores per anchor).
3. Decode proposals: anchor + offset → candidate bbox.
4. Filter (top-N by score) + NMS → ~300 proposals.
5. Send to second-stage classifier via ROI Align.

**Where it's used.** Inside every R-CNN-family detector (Faster R-CNN, Mask R-CNN, Cascade R-CNN). Internal — you rarely interact with it directly unless you're modifying the architecture.

**Related terms.**
- **Anchor box** — the reference shapes RPN predicts offsets for.
- **Selective Search** — the predecessor it replaced.
- **Objectness score** — RPN's "is there anything here?" output.

**Gotcha.** RPN runs *inside* the network — when you load Faster R-CNN from torchvision, the RPN is already there. You don't build it yourself unless you're doing research.

#### mAP (mean Average Precision)

> **🪜 Mental model:** *The detector's report card.* Combines precision and recall across confidence thresholds and IoU thresholds into one number.

**What it is.** **mAP** (mean Average Precision) is the standard evaluation metric for object detection. To compute it for one class:
1. Sort all predictions for that class by confidence (high to low).
2. For each prediction, mark it as TP (matches a GT box with IoU ≥ threshold and that GT hasn't been matched yet) or FP.
3. Compute precision and recall as you walk down the sorted list.
4. **AP** for the class = area under the precision-recall curve.
5. **mAP** = mean of AP over all classes.

Standard variants:
- **mAP@0.5** (PASCAL VOC) — single IoU threshold of 0.5.
- **mAP@0.5:0.95** (COCO) — average over IoU thresholds `[0.5, 0.55, …, 0.95]`. Harder, more discriminating.

**Why it matters.** mAP is *the* number everyone compares detectors on. Every detection paper reports it; every interview question on "how do you evaluate a detector" expects "mAP" as the first word.

**How it works at a glance.** Precision = TP / (TP + FP) = "of the boxes I called positive, how many were right?" Recall = TP / (TP + FN) = "of the GT boxes I should have found, how many did I?" Sweeping the confidence threshold traces the PR curve; AP is the integral of precision over recall.

**Where it's used.** COCO leaderboard scoring. Every detector's published results table. Your own validation metric during training.

**Related terms.**
- **Precision / Recall** — components of AP.
- **AP (Average Precision)** — per-class number, averaged across to give mAP.
- **IoU threshold** — what counts as a "match" — 0.5 (loose), 0.75 (tight), 0.5:0.95 (averaged).
- **F1 score** — alternative; less common for detection.

**Gotcha.** PASCAL VOC mAP@0.5 and COCO mAP@0.5:0.95 are **different numbers** — a model that scores 80 on the first might score 50 on the second. Always state the threshold.

#### RoI Pooling / RoI Align

> **🪜 Mental model:** *Variable-size patch → fixed-size patch.* The classifier head wants a 7×7 input, but proposals come in random sizes — this layer crops and resizes.

**What it is.** **ROI Pooling** (Fast R-CNN) and **ROI Align** (Mask R-CNN) are layers that take the shared backbone feature map plus a list of proposal boxes and return a *fixed-size* feature crop for each proposal (typically 7×7×C). The classifier head can then process them in a single batch.

- **ROI Pooling** quantises proposal coordinates to integer grid cells (snaps to the nearest feature-map pixel), then max-pools each cell. Fast, but quantisation loses precision.
- **ROI Align** uses **bilinear interpolation** to sample the feature map at exact (non-integer) proposal coordinates — no quantisation, sharper features. Required for pixel-accurate masks in Mask R-CNN.

**Why it matters.** Without one of these layers you can't share the backbone across proposals — the alternative is running the backbone separately for each proposal (R-CNN's slowness). ROI Align in particular boosts Mask R-CNN segmentation by ~3% mAP over plain ROI Pool.

**How they work.**
1. Map each proposal's coordinates onto the backbone's feature-map coordinates (divide by the cumulative stride).
2. Divide the proposal region into `7×7 = 49` bins.
3. Sample each bin (max pool for RoIPool; bilinear average for RoIAlign).
4. Output a `(7, 7, C)` tensor per proposal; stack into a batch.

**Where they're used.** Inside Faster R-CNN (RoIPool) and Mask R-CNN (RoIAlign). `torchvision.ops.roi_pool / roi_align` if you need to call them directly.

**Related terms.**
- **Quantisation** — what RoIPool does and RoIAlign avoids.
- **Bilinear interpolation** — how RoIAlign samples non-integer positions.
- **Spatial pyramid pooling (SPP)** — predecessor used in SPP-Net.

**Gotcha.** RoIPool's quantisation is fine for classification but **breaks pixel-accurate masks** — that's why Mask R-CNN had to invent RoIAlign.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#5g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Classification** | Predict one label for the whole image. | When the question is "what's the dominant content of this image?" |
| **Localisation** | Predict one label + one bbox for the (single) main object. | When there's exactly one object of interest and you need its location. |
| **Object detection** | Predict many labels + many bboxes, one per object. | When the image contains an unknown number of objects of varied classes. |
| **Bbox formats** | `(x1, y1, x2, y2)` corners vs `(xc, yc, w, h)` centre+size; pixel vs normalised. | Always confirm format before any IoU/NMS — wrong format = silently wrong neighbours. |
| **Region proposal** | A candidate box generated before classification — "something interesting here?" | Step 1 of any two-stage detector; from Selective Search (R-CNN) or RPN (Faster R-CNN). |
| **IoU** | `area(A ∩ B) / area(A ∪ B)` — overlap quality in `[0, 1]`. | Match detections to ground truth, drive NMS, evaluate mAP. Threshold 0.5 = standard "match." |
| **NMS** | Greedy dedup: keep highest-confidence, drop overlaps above IoU threshold, repeat. | Final post-processing step in every detector; run per-class to allow legitimate overlaps. |
| **R-CNN lineage** | R-CNN → Fast R-CNN → Faster R-CNN — each fixed the previous bottleneck. | Faster R-CNN is the canonical two-stage detector; default in production for accuracy-first jobs. |
| **RPN** | Learned mini-CNN that proposes regions from the backbone's feature map. | Inside every R-CNN-family detector since Faster R-CNN; replaces Selective Search. |
| **Anchor box** | Predefined reference shape; network predicts offsets from it. Typically 9 per location. | Inside RPN and inside YOLO; lets the network produce variable-shape boxes cheaply. |
| **mAP** | Mean Average Precision across classes; the standard detection metric. | Whenever you compare detectors; specify the IoU threshold (PASCAL@0.5 vs COCO@0.5:0.95). |
| **RoI Pooling / RoI Align** | Variable-size proposal → fixed `(7, 7, C)` feature crop. RoIAlign uses bilinear interp (no quantisation). | Inside two-stage detectors before the classifier head. RoIAlign for masks (Module 7). |

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

<a id="6g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 6 concept. The recap cheat sheet below summarises after.

#### One-stage vs two-stage detection

> **🪜 Mental model:** *Fast scan vs careful look.* One-stage = look once, predict everything (fast). Two-stage = propose, then classify (slower, more accurate).

**What they are.** Two families of object detectors, distinguished by whether they generate region proposals before classifying.
- **Two-stage** (Module 5) — propose ~300 candidate regions with an RPN, then classify and refine each. Accuracy-first. Examples: Faster R-CNN, Mask R-CNN, Cascade R-CNN.
- **Single-stage** (this module) — divide the image into a grid; predict (objectness, class probs, bbox offsets) for every grid cell + anchor in a single forward pass. Speed-first. Examples: **YOLO** (v1–v8), **SSD**, **RetinaNet**, **YOLOX**.

**Why it matters.** The choice is the classic speed-vs-accuracy tradeoff in detection. Self-driving cars and real-time video want single-stage (30+ FPS); medical imaging or security review want two-stage (max accuracy, latency irrelevant). Asked in nearly every CV interview as "which detector would you pick for X."

**How they differ.**

| | Two-stage | Single-stage |
|---|---|---|
| Speed | 5–10 FPS | 30–150 FPS |
| Accuracy (mAP@0.5:0.95 on COCO) | ~40–50 | ~35–50 (closing the gap) |
| Architecture | RPN + classifier head | Single dense prediction head |
| Anchor count | ~9 per location (300 proposals) | thousands per image |
| Hardest part | RPN training, NMS | class imbalance (focal loss) |

**Where they're used.** Two-stage: Tesla's older detectors, medical imaging, retail audits. Single-stage: real-time autonomous driving (Waymo, Cruise), video analytics, mobile / edge devices.

**Related terms.**
- **YOLO** — the canonical single-stage detector.
- **Faster R-CNN** — the canonical two-stage detector.
- **DETR** — newer transformer-based detector; neither classical one-stage nor two-stage.
- **Focal loss** — the trick that closed most of the accuracy gap for single-stage.

**Gotcha.** Modern YOLOs (v5+, v8) have closed the accuracy gap. Don't assume "single-stage = inaccurate"; benchmarks show YOLOv8 ≥ Faster R-CNN on COCO with 5× the speed.

#### YOLO (You Only Look Once)

> **🪜 Mental model:** *Grid + per-cell prediction.* Slice the image into an `S × S` grid; each cell predicts a few bounding boxes and their classes.

**What it is.** **YOLO** ("You Only Look Once", Redmon 2015) is the canonical single-stage detector. Its core idea: divide the image into an `S × S` grid; for each cell, predict `B` bounding boxes (each with x, y, w, h, objectness, and class probabilities). For YOLOv1 with `S=7, B=2, C=20` (PASCAL classes), the output shape is `7 × 7 × (2·5 + 20) = 7 × 7 × 30` — one forward pass produces every prediction.

The lineage:
- **YOLOv1 (2015)** — original grid idea, no anchors.
- **YOLOv2 / v3 (2017–2018)** — anchors, multi-scale prediction (3 scales), batch norm. Redmon's last.
- **YOLOv4, v5, v7, v8 (2020+)** — Alexey Bochkovskiy and the Ultralytics team — many engineering refinements (mosaic augmentation, FPN/PAN, DIoU-NMS, EMA weights, anchor-free variants).

**Why it matters.** YOLO is the default real-time detector. Its grid-based formulation is the prototypical example of single-stage thinking, and `from ultralytics import YOLO` is the fastest path from "I have images" to "I have detections." Every CV interview probes YOLO's grid + anchor + output decoding.

**How it works (high level).**
1. Backbone (e.g., CSPDarknet, ResNet) produces feature maps at multiple scales.
2. At each scale, an `S × S × (B · (5 + C))` tensor is predicted per location.
3. Each location's `5+C` numbers: `(tx, ty, tw, th, objectness, c_1, …, c_C)`.
4. Decode: convert raw outputs to absolute bbox coordinates (see "Grid cell / output decoding" below).
5. Filter by confidence; apply per-class NMS.

**Where it's used.** Real-time video (autonomous driving, surveillance), edge devices (Jetson, Raspberry Pi), Kaggle competitions. `from ultralytics import YOLO; YOLO('yolov5s.pt')('image.jpg')` is the modern one-liner.

**Related terms.**
- **Grid cell** — one of the `S × S` positions making predictions.
- **Anchor box** — predefined box shape per cell.
- **Objectness** — score saying "is there an object centred in this cell?"
- **SSD** — sibling single-stage detector with multi-scale prediction.

**Gotcha.** Different YOLO versions have **incompatible output formats**. v3's tensor layout differs from v5's; v8 is partially anchor-free. Always check the post-processing code matches the model version.

#### Anchor boxes

> **🪜 Mental model:** *Pre-shaped picture frames.* Instead of guessing a box from scratch, the network nudges one of a few standard rectangles into place.

**What it is.** **Anchor boxes** are predefined rectangle shapes assigned to each spatial position of the prediction grid. Instead of having the network output absolute bbox coordinates (hard — many possible scales and aspect ratios), each position offers `k` anchor "templates" and the network outputs **offsets and scaling** to nudge them into place. For Faster R-CNN, `k = 9` (3 scales × 3 aspect ratios). For YOLOv3, the anchors are clustered from training data via K-means.

**Why it matters.** Anchors transform the bbox-regression problem from "predict absolute coordinates" (high-variance, slow to learn) to "refine a starting guess" (much easier). They also give the network a way to encode *aspect ratios* — a tall anchor for pedestrians, a wide anchor for cars.

**How they work.**
1. Pick `k` anchors per location, covering varied scales and aspect ratios.
2. During training, each ground-truth box is matched to the anchor with highest IoU; that anchor becomes "responsible" for predicting it.
3. The network outputs per anchor: `(tx, ty, tw, th)` offsets relative to the anchor + objectness + class scores.
4. Decode: anchor + offsets → final bbox.

**Where they're used.** Faster R-CNN's RPN, YOLO v2–v5, SSD, RetinaNet. Modern **anchor-free** detectors (CenterNet, FCOS, YOLOX, DETR) skip them and predict bboxes directly.

**Related terms.**
- **Anchor-free detection** — predict bbox directly; removes anchor hyperparameters.
- **K-means anchor selection** — cluster training bbox sizes to derive task-specific anchors.
- **IoU matching** — how anchors get assigned to ground truths during training.

**Gotcha.** Default anchors (COCO-derived) may not fit your data. If you have unusual aspect ratios (long thin objects, tiny objects), K-means cluster your bboxes and use the resulting anchor sizes.

#### Grid cell / output decoding

> **🪜 Mental model:** *Spatial address + offsets.* Each grid cell is responsible for objects whose centre falls inside it; the network outputs offsets the cell adds to its own coordinates.

**What it is.** A **grid cell** is one `(i, j)` position in YOLO's `S × S` output grid. Each cell "claims" responsibility for any object whose centre falls within it and predicts `B` bounding boxes (one per anchor). The raw output for a cell is `(tx, ty, tw, th, objectness, class_probs)` per anchor — these are *offsets and log-scales* relative to the cell and anchor, not absolute coordinates.

**Decoding to absolute pixel coordinates:**
```
bx = (sigmoid(tx) + cell_x) / S * img_w     # centre x in pixels
by = (sigmoid(ty) + cell_y) / S * img_h     # centre y in pixels
bw = anchor_w * exp(tw)                      # width in pixels
bh = anchor_h * exp(th)                      # height in pixels
```

In words: *sigmoid of tx keeps the centre inside the cell; add the cell's grid index; divide by grid size to get a fraction; multiply by image size to get pixels. Width and height are the anchor's size times an exponential scale factor.*

**Why it matters.** Output decoding is the bridge between "raw network output tensor" and "usable bounding boxes." Bugs here are silent — you'll get boxes that look reasonable but are systematically shifted or rescaled. Every YOLO interview probes this math.

**Where it's used.** Inside every post-processing function for YOLO outputs. Bundled inside `ultralytics.YOLO` but exposed when you run via OpenCV DNN or ONNX Runtime.

**Related terms.**
- **Anchor box** — supplies `(anchor_w, anchor_h)` for decoding.
- **Sigmoid / exp** — the activations applied to raw outputs.
- **Stride** — `img_w / S`; the pixel size of one grid cell.

**Gotcha.** Older YOLO versions (v3) use `sigmoid` on `(tx, ty)`; newer (v5+) use a slightly different scaling. Check the post-processing code matches your model version.

#### Objectness score

> **🪜 Mental model:** *"Is there anything here?"* — a single number per bbox prediction saying how confident the network is that the box contains *some* object.

**What it is.** The **objectness score** is one of the numbers predicted per anchor in single-stage detectors. It's the probability that the box contains an object — independent of *which* object. Formally, it's defined as `P(object) × IoU(predicted_box, ground_truth)` — high only when both "there is an object" and "my box is well-aligned" hold.

The final per-class confidence used at inference is `objectness × class_prob`.

**Why it matters.** Objectness lets the detector separate "this looks like a thing" from "this thing is a car." It enables a cheap early filter: drop any prediction with objectness < 0.45 *before* multiplying with class probabilities, slashing the candidate set before NMS.

**How it works.** Trained as a binary cross-entropy target — 1 for anchors matched to a ground-truth box, 0 for unmatched anchors. The IoU-encoded version makes the network "self-aware" — it knows when its bbox is off, not just whether something's there.

**Where it's used.** YOLOv1–v8 (with minor variations). SSD doesn't use a separate objectness; it folds "background" into a class. RetinaNet uses focal loss on per-anchor classification without an explicit objectness.

**Related terms.**
- **Confidence threshold** — the cutoff applied to `objectness × class_prob` before NMS.
- **Focal loss** — used in RetinaNet to handle the same class imbalance objectness was created for.

**Gotcha.** Don't confuse objectness with class confidence — `objectness=0.9, class_prob=0.1` means "I'm sure something's here, but I'm not sure it's a car." Always multiply before thresholding.

#### Focal loss

> **🪜 Mental model:** *Pay attention to the hard ones.* Down-weight easy examples in the loss so the gradient focuses on the hard, misclassified ones.

**What it is.** **Focal loss** (Lin et al., RetinaNet, 2017) is a variant of cross-entropy designed for severe class imbalance — specifically the millions-of-background-boxes-vs-few-objects problem in single-stage detection. Formula:

`FL(p) = −α · (1 − p)^γ · log(p)`

i.e., *take the usual `-log(p)` cross-entropy and multiply by `(1 − p)^γ`, which is small when `p` is high (easy example) and large when `p` is low (hard example).* `α` balances class weights; `γ` (typically 2) controls how aggressively easy examples are down-weighted.

**Why it matters.** Without focal loss, single-stage detectors with thousands of anchors are dominated by easy negatives — the network just learns to predict "no object" everywhere. Focal loss was the key innovation that let RetinaNet match two-stage accuracy with single-stage speed.

**How it works.** When `p` (predicted probability of the true class) is high — say 0.9 — `(1 − p)^γ = 0.01` (with γ=2), so the loss contribution is 100× smaller than vanilla cross-entropy. Easy examples (mostly backgrounds) contribute negligible gradient; hard examples dominate.

**Where it's used.** RetinaNet, newer YOLO variants, EfficientDet. Available as `torchvision.ops.sigmoid_focal_loss` in PyTorch.

**Related terms.**
- **Cross-entropy** — the standard classification loss focal loss extends.
- **Class imbalance** — the problem focal loss solves.
- **Hard-negative mining** — older alternative; manually pick hard negatives during training.

**Gotcha.** Focal loss only helps when imbalance is severe. For balanced classes (cats vs dogs), plain cross-entropy is better — focal loss can actually hurt by suppressing the legitimate easy-example gradient signal.

#### SSD (Single-Shot Detector)

> **🪜 Mental model:** *Multi-scale YOLO.* Predict bboxes at multiple feature-map scales — small objects from shallow maps (high resolution), large objects from deep maps (low resolution).

**What it is.** **SSD** (Liu et al., 2016) is a single-stage detector with the same "one forward pass" idea as YOLO but with **multi-scale prediction**: instead of a single output grid, SSD attaches detection heads to *multiple* feature-map scales in the backbone. Shallow (high-res) maps predict small objects; deep (low-res) maps predict large ones. SSD uses anchor boxes (called "default boxes" in the paper).

**Why it matters.** YOLOv1 struggled with small objects because the grid was coarse (7×7). SSD's multi-scale outputs handled small objects much better. Later YOLOs adopted the same multi-scale idea via FPN. SSD is still used as a lighter alternative to YOLO on mobile.

**How it works.** Backbone (typically VGG-16) produces feature maps at several resolutions. At each scale, a small detection head predicts (objectness, class probs, bbox offsets) for `k` default boxes per location. All predictions are combined and NMS'd at the end.

**Where it's used.** Mobile / edge devices when YOLO is too heavy. `torchvision.models.detection.ssd300_vgg16(pretrained=True)`. SSD-MobileNet variants are popular on Coral Edge TPU and Jetson Nano.

**Related terms.**
- **FPN (Feature Pyramid Network)** — modern way to add multi-scale prediction to any detector.
- **YOLOv3+** — adopted multi-scale prediction (3 scales) inspired by SSD.
- **RetinaNet** — newer single-stage; multi-scale + focal loss.

**Gotcha.** SSD's multi-scale design helps small objects only modestly — the shallow feature maps are semantically weak (they've seen few conv layers). FPN's top-down pathway is what truly solved small-object detection.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#6g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **One-stage vs two-stage** | One-stage predicts everything in one pass (fast); two-stage proposes then classifies (accurate). | Real-time/edge → one-stage. Latency-tolerant + accuracy-critical → two-stage. |
| **YOLO** | Single-stage detector; image divided into `S×S` grid, each cell predicts `B` anchors. | The default real-time detector; `ultralytics.YOLO` is the fastest path from "I have images" to "I have boxes." |
| **YOLO output shape** | `S × S × B × (5 + C)` per scale (5 = x, y, w, h, objectness). | When decoding raw outputs to bboxes; always confirm shape matches your version (v3 vs v5 vs v8). |
| **Anchor box** | Predefined rectangle shapes per location; the network predicts offsets to nudge them. | Inside RPN, YOLO, SSD, RetinaNet. Cluster from training data with K-means for unusual aspect ratios. |
| **Grid cell** | One `(i, j)` position in YOLO's output grid; responsible for objects centred in it. | When decoding YOLO outputs — convert `(tx, ty)` offsets + cell coords to absolute pixels. |
| **Objectness** | `P(object) × IoU(pred, GT)` — combined "something here" and "box accurate" score. | Multiply with class probs before thresholding; cheap early filter before NMS. |
| **Focal loss** | `−α(1−p)^γ log(p)` — down-weights easy examples to focus on hard ones. | Single-stage detectors with severe class imbalance (millions of backgrounds vs few objects). |
| **RetinaNet** | First single-stage method to match two-stage accuracy, via focal loss + FPN. | When you want one-stage speed but two-stage accuracy. `torchvision.models.detection.retinanet_resnet50_fpn`. |
| **SSD** | Multi-scale single-stage detector — predictions at multiple feature-map resolutions. | Mobile/edge alternative to YOLO; SSD-MobileNet is the staple on Coral / Jetson Nano. |
| **FPN** | Feature Pyramid Network — top-down pathway adds semantic context to high-res features. | Inside every modern detector — fixes small-object detection by enriching shallow feature maps. |
| **ONNX** | Open Neural Network Exchange — portable model format. | When deploying via OpenCV DNN, ONNX Runtime, or TensorRT — runs on CPU/GPU/edge uniformly. |

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

<a id="7g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 7 concept. The recap cheat sheet below summarises after.

#### Object segmentation (vs detection)

> **🪜 Mental model:** *Outline, not box.* Detection draws a rectangle around an object; segmentation paints every pixel that belongs to the object.

**What it is.** **Object segmentation** is the CV task of assigning a class label to **every pixel** of an image. The output is a 2-D grid the same shape as the input where each cell carries a class ID (or a probability distribution over classes). Where detection gives you a bounding rectangle, segmentation gives you the exact outline.

**Why it matters.** Many real-world tasks need pixel-precision: portrait mode (where exactly is the person?), tumour segmentation in MRIs, lane-line detection in self-driving, satellite cropland mapping. A bounding box around a kidney is useless to a surgeon — you need the exact organ outline. Segmentation is also a frequent CV interview topic, especially for medical-imaging and AR/VR companies.

**How it works.** Most segmentation networks are **fully convolutional** (no Dense layers — preserves spatial structure throughout). The classic architecture is **encoder-decoder**: an encoder downsamples the image into a small, semantically rich representation; a decoder upsamples it back to the original resolution, producing a per-pixel prediction. **Skip connections** between matching encoder/decoder levels preserve spatial detail.

**Where it's used.** Portrait mode on phones (Pixel, iPhone), medical imaging (tumour, organ, lesion segmentation), self-driving (lane lines, drivable area), satellite analytics (crop classification, land use), AR/VR (background removal), industrial QC.

**Related terms.**
- **Semantic segmentation** — per-pixel class only.
- **Instance segmentation** — per-pixel class + instance ID.
- **Panoptic segmentation** — both, unified.
- **Detection** — sibling task, returns bboxes not masks.

**Gotcha.** Segmentation output has the same `(H, W)` as the input — losing spatial resolution anywhere in the network (over-aggressive pooling) destroys mask quality. Encoder-decoder + skips exist precisely to prevent this.

#### Semantic vs instance segmentation

> **🪜 Mental model:** *Class-mask vs labelled-instances.* Semantic = "this is cat-pixels"; Instance = "this is cat #1, this is cat #2."

**What they are.**
- **Semantic segmentation** — each pixel gets one of `C` class labels. All cats share the label "cat"; you can't tell two cats apart in the mask.
- **Instance segmentation** — each pixel gets a class label *and* an instance ID. Two cats next to each other get separate masks. Outputs are typically a list of (class, score, mask) triples — one per detected instance.
- **Panoptic segmentation** — combines both: "stuff" classes (sky, road, grass — uncountable) get semantic labels, "thing" classes (cars, people — countable) get instance IDs.

**Why it matters.** Picking the right task affects architecture, loss function, and metric. "Find the road" → semantic. "Count and segment each car separately" → instance. Confusing these in an interview signals you haven't actually shipped a segmentation system.

**How they differ.**

| | Semantic | Instance | Panoptic |
|---|---|---|---|
| Output | `(H, W)` class map | List of (class, mask) | `(H, W)` map of `(class, instance_id)` |
| Architecture | U-Net, DeepLab, FCN | Mask R-CNN, YOLACT | Panoptic-FPN, Mask2Former |
| Metric | mean IoU | mAP (mask-IoU based) | PQ (Panoptic Quality) |

**Where they're used.** Semantic — portrait mode, road/lane segmentation. Instance — counting (cells, retail items), AR object pinning. Panoptic — full scene understanding (autonomous driving).

**Related terms.**
- **Stuff vs things** — uncountable vs countable classes.
- **mIoU** — semantic metric.
- **PQ (Panoptic Quality)** — panoptic metric.

**Gotcha.** Don't use a semantic model when you need to *count* objects — overlapping or touching instances merge into one mask. Mask R-CNN exists for exactly this case.

#### Pixel-wise classification

> **🪜 Mental model:** *Classification, but one prediction per pixel.* The same softmax-over-classes idea, just repeated `H × W` times.

**What it is.** **Pixel-wise classification** is the operational view of semantic segmentation: instead of producing one class label per image, the network produces one class label per pixel. The output is a tensor of shape `(H, W, C)` where the channel axis is the per-pixel class probability distribution; `argmax` across channels gives the final label per pixel.

**Why it matters.** This framing tells you why segmentation networks (a) drop Dense/Flatten layers (which collapse spatial dims), (b) end in a `1×1 Conv` with `C` filters (per-pixel classifier head), (c) train with **per-pixel cross-entropy loss** — but watch for class imbalance.

**How it works.** Final layer: `Conv2D(num_classes, 1, activation='softmax')` produces the per-pixel distribution. Loss: `sparse_categorical_crossentropy` (or BCE for binary masks) averaged across all pixels.

**Where it's used.** Every semantic segmentation network. Also conceptually how dense prediction tasks like depth estimation and surface-normal regression are framed.

**Related terms.**
- **`1×1` convolution** — the final head layer that maps features to class scores per pixel.
- **Cross-entropy** — the standard loss.
- **Dice loss** — alternative loss robust to class imbalance.

**Gotcha.** Per-pixel cross-entropy averages over every pixel — if 95% of pixels are background, the loss is dominated by easy background pixels. Hence Dice (or focal) loss for imbalanced masks.

#### U-Net architecture

> **🪜 Mental model:** *A "U" of feature maps.* Information flows down (encoder shrinks spatially, grows in channels), hits the bottom, and flows back up (decoder grows spatially, shrinks in channels), with side-doors at every level (skips).

**What it is.** **U-Net** (Ronneberger et al., 2015) is the canonical encoder-decoder segmentation network, originally for biomedical microscopy. Its shape:
- **Encoder (contracting path):** repeated `Conv → ReLU → MaxPool` blocks; spatial dims shrink, channel count grows.
- **Bottleneck:** smallest spatial size, most semantic features.
- **Decoder (expansive path):** repeated `Upsample → Concatenate-with-encoder-skip → Conv → ReLU`; spatial dims grow back to input size.
- **Skip connections** between matching encoder/decoder levels concatenate feature maps, carrying fine spatial detail forward.
- **Final head:** `Conv2D(num_classes, 1, softmax)` per-pixel classifier.

**Why it matters.** U-Net is the single most-used segmentation architecture, especially in medical imaging where it was born. Its skip connections are *the* idea that makes high-resolution dense prediction work. Every segmentation interview asks about U-Net.

**How it works.** Encoder downsamples 4× (typical): `(H, W) → (H/16, W/16)`. Decoder upsamples back 4×. At each decoder level, the upsampled features are concatenated with the matching encoder-level features before the next conv — that's the skip connection.

**Where it's used.** Medical imaging (segmentation of organs, tumours, cells). Portrait mode. Satellite imagery. As the backbone of diffusion models (Stable Diffusion's denoising U-Net is structurally a U-Net). `segmentation_models_pytorch.Unet(encoder_name='resnet34')` is the standard production version.

**Related terms.**
- **FCN (Fully Convolutional Network)** — U-Net's predecessor (2015); FCN-8s, FCN-16s variants.
- **Skip connection** — next entry; the key U-Net feature.
- **Encoder-decoder** — the architectural pattern U-Net popularised for segmentation.
- **DeepLab** — sibling architecture using atrous (dilated) convs instead of pooling.

**Gotcha.** Skip connections concatenate channels — make sure the encoder and decoder levels have matching `(H, W)` before concat. Off-by-one errors here are common when input sizes aren't powers of 2.

#### Skip connections (in U-Net)

> **🪜 Mental model:** *Shortcut wire.* Take the encoder's high-resolution feature map and patch it directly into the decoder at the same level, bypassing the bottleneck.

**What it is.** A **skip connection** in U-Net is a direct path from an encoder block's output to the matching decoder block's input. The decoder concatenates the encoder feature map with its own upsampled feature map along the channel axis, then applies a conv. This means the decoder sees both "where the object is" (high-res from encoder skip) and "what the object is" (semantic from the bottleneck).

**Why it matters.** Without skips, fine spatial details (object boundaries, thin structures like blood vessels) are lost during encoder downsampling and the decoder has nothing to recover them from. With skips, edges stay sharp. The skip is what makes U-Net a U — without them it'd be a hourglass that throws away detail.

**How it works.**
1. At each encoder level, save the feature map (call it `c_k`).
2. After bottleneck, upsample and reach decoder level `k`.
3. Concatenate `[upsampled_features, c_k]` along channel axis.
4. Apply a `Conv → ReLU` block.
5. Repeat until back to input resolution.

```python
c1 = layers.Conv2D(64, 3, 'same', activation='relu')(inp); p1 = layers.MaxPooling2D()(c1)
# ... bottleneck ...
u1 = layers.UpSampling2D()(d2)
u1 = layers.Concatenate()([u1, c1])   # ← skip connection
d1 = layers.Conv2D(64, 3, 'same', activation='relu')(u1)
```

**Where it's used.** U-Net (concatenation). Also in **ResNet** (addition: `out = F(x) + x`) — same idea, different combination. In **DenseNet** every layer skip-connects to every later layer.

**Related terms.**
- **Residual connection** — ResNet's variant; addition instead of concatenation.
- **Dense connection** — DenseNet; everyone skips to everyone.
- **Encoder-decoder** — the broader architectural pattern.

**Gotcha.** Concatenation (U-Net) doubles the channel count at each skip; addition (ResNet) keeps it. Watch the memory budget — a deep U-Net at high resolution explodes memory because of all the kept-around encoder feature maps.

#### Upsampling (vs transposed convolution)

> **🪜 Mental model:** *Stretch the picture back up.* Either by simple interpolation (no parameters) or by a learnable "reverse-conv."

**What they are.** Two ways to enlarge a feature map's spatial dimensions inside the decoder.
- **UpSampling2D** — fixed interpolation. `size=2` doubles `(H, W)` by repeating (nearest neighbour) or smoothing (bilinear). **No learnable parameters.**
- **Conv2DTranspose** (transposed convolution, sometimes wrongly called "deconvolution") — learnable upsampling. It's effectively the gradient operation of a strided conv, run forward — outputs a larger spatial size, with learnable kernel weights. Parameters: `(kH × kW × Cin + 1) × Cout`.

**Why it matters.** The decoder must somehow grow spatial dims back to the input size. Picking between these two affects parameter count, training stability, and output quality. Modern best practice for U-Net is **UpSampling2D + Conv2D** (cheap, no checkerboard artefacts); GANs use Conv2DTranspose more often.

**How they work.**
- `UpSampling2D(size=2, interpolation='nearest')` — each input pixel is duplicated into a `2×2` block. With `interpolation='bilinear'`, intermediate pixels are interpolated.
- `Conv2DTranspose(filters, kernel, strides=2)` — for each input pixel, multiply by the kernel (learnable) and sum overlapping contributions into the larger output. Can produce **checkerboard artefacts** when `kernel_size % stride != 0`.

**Where they're used.** U-Net decoder (`UpSampling2D + Conv2D` is the common combo). GAN generator (`Conv2DTranspose` for learnable upsampling all the way up). Super-resolution networks (sub-pixel/PixelShuffle is yet another alternative).

**Related terms.**
- **Sub-pixel convolution / PixelShuffle** — outputs `r²` channels then reshapes spatially; avoids checkerboard.
- **Bilinear / nearest interpolation** — the two simple modes for `UpSampling2D`.
- **Checkerboard artefact** — Conv2DTranspose's signature failure mode.

**Gotcha.** **Conv2DTranspose** produces checkerboard artefacts when `kernel_size % stride != 0`. Use `kernel=4, strides=2` (safe) or switch to `UpSampling2D + Conv2D`.

#### Dice loss

> **🪜 Mental model:** *F1 score for masks.* Reward overlap as a *whole* rather than averaging across millions of pixels.

**What it is.** **Dice loss** is a segmentation loss derived from the **Dice coefficient** (a.k.a. F1 score for sets):

`Dice(A, B) = 2 · |A ∩ B| / (|A| + |B|)`

i.e., *twice the overlap area divided by the sum of both areas; equal to 1 when masks coincide, 0 when they don't overlap.* Dice loss = `1 − Dice`.

For predicted mask `p` and ground truth `y`:
```python
dice = (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
dice_loss = 1 - dice
```

**Why it matters.** Per-pixel cross-entropy on a portrait mask (95% background) is dominated by easy background pixels; the model learns "predict background" and never finds the foreground edges. Dice measures global overlap and is **naturally robust to imbalance** — a model that predicts all-background scores 0 in Dice.

**How it works.** The numerator (intersection area, smoothed) rewards true positive pixels. The denominator (sum of both areas) acts as a normaliser. Because Dice is a *whole-mask* metric, the loss gradient encourages the predicted mask to match the ground-truth mask *as a region*, not pixel-by-pixel.

**Where it's used.** Medical imaging (organ, tumour, lesion segmentation). Portrait masks. Any imbalanced binary segmentation. Often combined with BCE: `loss = BCE + Dice` — BCE provides stable early gradients; Dice fine-tunes boundaries.

**Related terms.**
- **Dice coefficient** — the metric (1 = perfect overlap).
- **Jaccard / IoU loss** — sibling overlap-based loss; smaller gradient near optimum.
- **Focal loss** — alternative class-imbalance fix.
- **BCE + Dice** — the de-facto combo loss.

**Gotcha.** Watch the `smooth` epsilon — too small (1e-8) can cause numerical issues; too large (10) softens the metric. `smooth=1` is the standard pick.

#### IoU for segmentation

> **🪜 Mental model:** *Same IoU as detection, but on masks instead of boxes.* Compare the predicted pixel set to the ground-truth pixel set.

**What it is.** For segmentation, **IoU** (a.k.a. Jaccard index) is computed between two **masks** (sets of pixels) rather than two bboxes:

`IoU(A, B) = |A ∩ B| / |A ∪ B|`

where `A` and `B` are the sets of pixels in the predicted and ground-truth masks. Range `[0, 1]`, 1 = perfect overlap.

**Mean IoU (mIoU)** is IoU averaged across all classes — the standard semantic-segmentation metric.

**Why it matters.** mIoU is the canonical leaderboard metric for semantic segmentation (Cityscapes, ADE20K, PASCAL VOC seg). It treats every class equally regardless of pixel count, fixing the imbalance issue that ruins pixel accuracy.

**How it works.** For each class, count true positives (pixels predicted as the class and actually the class), false positives, false negatives. `IoU = TP / (TP + FP + FN)`. Average across classes.

**Where it's used.** Every segmentation paper reports mIoU. `torchmetrics.JaccardIndex`, `tf.keras.metrics.MeanIoU` are built-in implementations.

**Related terms.**
- **Dice coefficient** — closely related: `Dice = 2·IoU / (1 + IoU)`.
- **Pixel accuracy** — fraction correctly classified; *poor* on imbalanced classes.
- **Boundary IoU** — variant emphasising edges, used in newer benchmarks.

**Gotcha.** mIoU averages across classes, *not* across images — a rare class with poor IoU pulls the mean down hard, which can look weird if your test set is small.

#### Mask R-CNN

> **🪜 Mental model:** *Faster R-CNN + a mask painter.* For each detected object, run a tiny FCN on the cropped feature region to produce a per-instance mask.

**What it is.** **Mask R-CNN** (He et al., 2017) extends Faster R-CNN with a third head: alongside the classifier and bbox regressor at each ROI, a small fully-convolutional **mask head** predicts a `28×28` binary mask for each detected object. The masks are predicted *per class* (the head outputs `C` masks; you pick the one for the predicted class).

Mask R-CNN replaces ROI Pooling with **ROI Align** (no quantisation), which is critical for pixel-accurate masks.

**Why it matters.** Mask R-CNN is the canonical **instance segmentation** model — used everywhere you need to count and outline objects separately (cell biology, retail, AR). It's the natural extension of Faster R-CNN and a frequent CV interview topic.

**How it works.**
1. Backbone + RPN propose regions (same as Faster R-CNN).
2. ROI Align extracts a `(7, 7, C)` feature crop per proposal.
3. Three parallel heads on the crop: classifier, bbox regressor, mask head.
4. Mask head is a small FCN producing `28×28×C` mask logits per ROI.
5. Pick the mask for the predicted class; upsample to bbox size.

**Where it's used.** `torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)` — three lines to get COCO-pretrained instance segmentation. Used in cell-counting, retail shelf audits, AR object tracking.

**Related terms.**
- **Faster R-CNN** — the base detector Mask R-CNN extends.
- **ROI Align** — Mask R-CNN's key new operation; avoids RoIPool's quantisation.
- **U-Net** — sibling architecture for *semantic* segmentation.
- **Cascade Mask R-CNN** — multi-stage refinement variant.

**Gotcha.** Mask R-CNN's output masks are at **28×28 resolution per ROI** — you must upsample them to the bbox's actual pixel size before displaying. Otherwise the mask looks blocky.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#7g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Object segmentation** | Per-pixel labelling — output is a class map the same `(H, W)` as the input. | When you need pixel-precise outlines, not just bboxes (portrait, medical, lane lines). |
| **Semantic segmentation** | Per-pixel class only; instances of the same class merge. | When you only need "where is the road / cat / tumour," not "how many." |
| **Instance segmentation** | Per-pixel class + instance ID; separate masks for each object. | When you need to count or track individuals (cells, retail items, people). |
| **Panoptic segmentation** | Semantic for "stuff," instance for "things." Unified output. | Full scene understanding (autonomous driving, robotics). |
| **Pixel-wise classification** | Operational framing of semantic seg — softmax over classes, per pixel. | Build the head as `Conv2D(num_classes, 1, softmax)`; train with per-pixel CE (or Dice for imbalance). |
| **U-Net** | Encoder-decoder with skip connections at every level. The canonical seg architecture. | Default for any semantic segmentation; basis of diffusion-model denoisers too. |
| **Skip connection** | Direct copy from encoder block to decoder block at the same level. Concatenated. | The U-shape of U-Net; what preserves spatial detail through the bottleneck. |
| **Upsampling vs transposed conv** | `UpSampling2D` (fixed interp, no params) vs `Conv2DTranspose` (learnable, can checkerboard). | U-Net decoder: `UpSampling2D + Conv2D` is cleanest; GAN generators use `Conv2DTranspose`. |
| **Dice loss** | `1 − 2|A∩B|/(|A|+|B|)` — mask-level overlap loss; robust to class imbalance. | Imbalanced binary masks (portrait, medical). Often combined with BCE for stable early gradients. |
| **IoU (segmentation)** | `|A∩B| / |A∪B|` on mask pixel sets; mean across classes = mIoU. | Standard semantic-segmentation evaluation metric. |
| **Mask R-CNN** | Faster R-CNN + a 28×28 mask head per ROI + ROI Align. | Instance segmentation; `torchvision.models.detection.maskrcnn_resnet50_fpn`. |
| **DeepLab v3** | Atrous (dilated) convs for multi-scale context without downsampling. | Semantic segmentation alternative to U-Net; good when you need large receptive field at full resolution. |

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

<a id="8g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 8 concept. The recap cheat sheet below summarises after.

#### Siamese network

> **🪜 Mental model:** *Twin networks with one set of brains.* Two copies of the same network process two inputs; their outputs (embeddings) are compared via a distance metric.

**What it is.** A **Siamese network** is an architecture made of **two (or more) identical sub-networks with shared weights**, used to compare inputs by producing comparable embeddings. Each input passes through one twin to produce an embedding; the embeddings are then combined (via L2 distance, cosine similarity, or a small learned merge head) to produce a similarity score. The fact that the twins share weights makes the comparison *symmetric*: `f(A)` and `f(B)` come from the same map, so `d(f(A), f(B)) = d(f(B), f(A))`.

**Why it matters.** Standard classification networks need a fixed number of classes and break when you add a new identity (you'd have to retrain). Siamese networks instead learn a *similarity function* — adding a new identity is just storing one new embedding, no retraining required. This makes them the right tool for face verification (Apple Face ID, FaceNet), signature verification, fingerprint matching, and deduplication.

**How it works.**
1. Define an encoder (e.g., a frozen ResNet-50 + a small Dense projection head).
2. Pass two inputs `A` and `B` through *the same* encoder → embeddings `e_A`, `e_B`.
3. Compute distance `d = ‖e_A − e_B‖₂` (or cosine).
4. Train with **contrastive** or **triplet** loss to push same-identity pairs closer and different-identity pairs farther.

**Where it's used.** Face verification (FaceNet, ArcFace), signature/fingerprint verification (the Module 8 notebook uses BHSig260 signatures), one-shot learning (Matching Networks), near-duplicate detection, dense retrieval embeddings (Sentence-BERT in NLP — same idea).

**Related terms.**
- **Embedding** — the encoder's output that gets compared.
- **Contrastive loss / triplet loss** — the two main training objectives.
- **One-shot / few-shot learning** — the use case Siamese solves.
- **Shared weights** — the architectural constraint that gives symmetry.

**Gotcha.** The twin networks **must** share weights. If you forget to share (i.e., you build two separate encoders) the comparison is incoherent — `f_1(A)` and `f_2(B)` aren't even in the same space.

#### One-shot / few-shot learning

> **🪜 Mental model:** *"Here's one example — find more like it."* Recognise a new identity from a single labelled example by comparing embeddings.

**What it is.** **One-shot learning** is the problem of correctly classifying a new instance after seeing only *one* labelled example of its class. **Few-shot learning** generalises to *k* examples (k = 1, 5, 20). Classification networks don't handle this — they need many examples per class. The Siamese approach reframes the problem: learn a *similarity function*, then at deploy time store one (or a few) embedding(s) per identity and classify new images by nearest-neighbour search.

**Why it matters.** Almost every face-recognition system in production is solving a one-shot problem: when a new user enrolls, you have *one* selfie of them. Same for signature verification, drug discovery (one example of a binding compound), and rare-disease imaging. It's also a popular research benchmark (Omniglot, miniImageNet).

**How it works.**
1. **Train** a Siamese network on lots of (pair, same/different) data so the embedding space cleanly separates identities.
2. **Enrol** a new identity by computing and storing its embedding from one reference image.
3. **Verify** a new image by computing its embedding and measuring distance to the stored reference. If distance < threshold → match.

**Where it's used.** Face Unlock (Apple, Android). Bank signature verification. Speaker verification (audio-side Siamese). Few-shot product classification in retail.

**Related terms.**
- **Verification** — binary "same person?" decision on a pair.
- **Identification** — multi-class "which person?" by 1-vs-N comparison.
- **Open-set recognition** — at deploy time the model meets new identities not seen at training; Siamese handles this naturally.

**Gotcha.** One-shot ≠ zero-shot. Zero-shot means *no* example of the target class — different problem solved by language-conditioned models (CLIP).

#### Contrastive loss

> **🪜 Mental model:** *Magnet that flips polarity by label.* Same-identity pairs are pulled together; different-identity pairs are pushed apart up to a margin.

**What it is.** **Contrastive loss** (Hadsell, Chopra, LeCun 2006) trains a Siamese network on **pairs**. Given a pair `(A, B)` with label `y` (1 = same identity, 0 = different) and distance `d = ‖f(A) − f(B)‖₂`:

`L = y · d² + (1 − y) · max(0, margin − d)²`

In words:
- For **same-identity** pairs (`y = 1`), the loss is `d²` — minimised by pulling embeddings together until `d = 0`.
- For **different-identity** pairs (`y = 0`), the loss is `max(0, margin − d)²` — non-zero only if the distance is *less than* the margin; the gradient pushes the embeddings apart until they're at least `margin` away.

**Why it matters.** Contrastive loss is the simpler of the two main metric-learning losses (the other being triplet). It only needs pairs (easier to construct than triplets), and the math is straightforward.

**How it works.** Backprop through the loss adjusts the encoder's weights so that the embedding geometry matches the pairs' labels: clustering same-class samples and separating different-class samples by at least `margin`.

**Where it's used.** Early Siamese papers, signature verification, simple dedup tasks. Modern face/embedding systems usually prefer triplet or softmax-based variants (ArcFace) for stronger results.

**Related terms.**
- **Triplet loss** — sibling loss using triples instead of pairs.
- **Margin** — the required gap for negative pairs.
- **Hard pair** — a pair that contributes high loss (informative for training).

**Gotcha.** Choose `margin` carefully. Too small → embeddings collapse to a point. Too large → optimisation can't satisfy the constraint, training stalls. Start with `margin = 1` on L2-normalised embeddings.

#### Triplet loss

> **🪜 Mental model:** *Anchor, friend, stranger.* Make sure the anchor is closer to its friend (positive) than to a stranger (negative) by at least margin α.

**What it is.** **Triplet loss** (Schroff et al., FaceNet, 2015) trains a Siamese network on **triples**: an **anchor** image, a **positive** (same identity as anchor), and a **negative** (different identity). With embeddings `e_a, e_p, e_n` and margin `α`:

`L = max(0, ‖e_a − e_p‖² − ‖e_a − e_n‖² + α)`

In words: *"the squared distance from anchor to positive must be at least α smaller than the squared distance from anchor to negative; if not, contribute the difference as loss."* When the constraint is already satisfied, loss = 0.

**Why it matters.** Triplet loss enforces a **relative** ordering (positive closer than negative) rather than absolute distances, which is more stable and produces sharper embeddings than contrastive loss. FaceNet showed triplet loss could push embeddings to a quality where simple L2 distance solves face verification at human level.

**How it works.**
1. Sample a triplet `(a, p, n)` from the dataset.
2. Compute embeddings via the shared encoder.
3. Compute `‖e_a − e_p‖² − ‖e_a − e_n‖² + α`.
4. If positive, take the value as loss; if negative, loss = 0.
5. Backprop.

The hard part is **triplet sampling** — most random triplets are "easy" (positive is already closer than negative by ≫ α, loss = 0, no gradient). Solution: **hard-negative mining** (next entry).

**Where it's used.** FaceNet, modern person re-identification, dense retrieval embeddings, speaker verification.

**Related terms.**
- **Contrastive loss** — pair-based sibling.
- **Margin (α)** — the required gap.
- **Hard / semi-hard / easy negative** — triplet difficulty classes (see "Hard negative mining").
- **N-pair loss** — generalisation using one positive and many negatives.

**Gotcha.** Random triplet sampling stalls training within an epoch — almost all triplets are easy. You must implement **online semi-hard mining** (within a batch, pick the toughest valid negative).

#### Margin (in contrastive / triplet loss)

> **🪜 Mental model:** *Required safety gap.* Even after the model is "right," push positives and negatives apart by at least this much.

**What it is.** The **margin** (α in triplet loss; `margin` in contrastive loss) is the minimum required separation between same-class and different-class pairs in embedding space. Without a margin the optimum is trivial (collapse all embeddings to one point — distance always zero between same-class pairs). The margin prevents collapse and forces meaningful spread.

**Why it matters.** The margin is the single most impactful hyperparameter in metric learning. Too small → embeddings collapse → no useful structure. Too large → optimisation can never satisfy the constraint → training stalls in high loss. Practical heuristic: start with `margin = 0.2` for unit-norm cosine, `margin = 1.0` for raw L2 embeddings.

**How it works.** It's just a constant in the loss formula. Its effect is purely on the gradient landscape: it shifts where the loss function starts contributing non-zero gradient.

**Where it's used.** Contrastive loss, triplet loss, ArcFace (additive angular margin), CosFace (additive cosine margin).

**Related terms.**
- **Embedding collapse** — what happens with zero margin.
- **Hard negative** — a triplet violating the margin maximally.
- **ArcFace** — modern margin-based softmax loss for face recognition.

**Gotcha.** When you change normalisation (e.g., add L2-norm to embedding output), you must re-tune the margin — the magnitude scale of distances changed.

#### Hard-negative mining

> **🪜 Mental model:** *Train on the borderline cases.* Most triplets are easy (loss = 0); pick the ones near the margin so the gradient actually teaches the network something.

**What it is.** **Hard-negative mining** is the practice of intentionally picking *informative* negatives during training. With random sampling, most triplets are *easy* (negatives already far away), produce zero loss, and contribute no gradient. By preferring hard or semi-hard negatives, you ensure every step is a meaningful update.

Three negative classes:
- **Easy:** `d(a, n) > d(a, p) + α` — already satisfied; loss = 0; no gradient.
- **Semi-hard:** `d(a, p) < d(a, n) < d(a, p) + α` — within the margin; moderate gradient.
- **Hard:** `d(a, n) < d(a, p)` — completely violates the constraint; biggest gradient but can destabilise training.

**Why it matters.** Without mining, triplet-loss training stalls within an epoch — all gradients are zero. With **online semi-hard mining** (per batch, pick the toughest semi-hard negative), training is stable and produces strong embeddings. FaceNet showed semi-hard works better than hardest negatives (the hardest can be label-noise outliers and destabilise things).

**How it works.**
1. At every training step, pull a large batch (1000+) from the dataset.
2. Compute all-pairs distances within the batch.
3. For each anchor and its positive, find the **semi-hard** negatives within that batch (distance > `d(a,p)` but < `d(a,p) + α`).
4. Construct triplets from those; compute loss.
5. Backprop.

**Where it's used.** Every modern Siamese / triplet training pipeline. PyTorch Metric Learning library, `tensorflow_addons.losses.TripletSemiHardLoss`.

**Related terms.**
- **Online vs offline mining** — online mines within the current batch; offline pre-mines across the whole dataset.
- **Semi-hard** — the practical sweet spot.
- **Easy negative** — wasteful.
- **Hard negative** — informative but can be label noise.

**Gotcha.** Hardest negatives often *are* label noise (a "different" person who actually looks identical to the anchor, or a mislabel). Use semi-hard, not hardest.

#### Face verification (canonical application)

> **🪜 Mental model:** *Two photos in, "same person?" out.* The killer app of Siamese networks — and the engine behind every face unlock on Earth.

**What it is.** **Face verification** is the task of deciding whether two face images depict the same person. Output: a binary yes/no, usually via a learned threshold on embedding distance. It's distinct from **face identification** (which person, from a known database) and **face detection** (where are the faces).

The standard architecture: pretrained face-recognition backbone (FaceNet, ArcFace) → 128- or 512-dim embedding → cosine distance → threshold.

**Why it matters.** Face verification is *the* canonical Siamese-network task — taught in every CV interview and powering every face-unlock system. Understanding it cements the why-Siamese argument: classification can't handle new identities; verification (and Siamese) can.

**How it works.**
1. Detect faces in both images with a face detector (MTCNN, RetinaFace).
2. Align faces (eyes-on-horizontal) for consistency.
3. Compute embeddings via the trained Siamese encoder.
4. L2-normalise.
5. Compute cosine distance.
6. Compare to a threshold (typically 0.4–0.6, tuned on validation).

**Where it's used.** Apple Face ID, Android FaceUnlock, airport e-gates, KYC (Know Your Customer) verification at banks, content moderation (deepfake detection adjacent), social-media tagging.

**Related terms.**
- **FaceNet** — Google's 2015 paper; canonical face Siamese.
- **ArcFace** — modern improvement using angular margin softmax.
- **MTCNN / RetinaFace** — face detectors used as preprocessing.
- **Threshold tuning** — sweep distance threshold on validation to pick the best operating point.

**Gotcha.** Real-world face verification needs much more than the model: lighting normalisation, alignment, liveness detection (to defeat photo-attack), and threshold calibration per population. The Siamese model is just one component.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#8g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Siamese network** | Two (or more) identical sub-networks with **shared weights**; compare inputs via embedding distance. | Verification, one-shot recognition, deduplication — anywhere classification can't handle new classes. |
| **One-shot learning** | Recognise a new class from a single labelled example by comparing embeddings. | Face unlock, signature verification — new identities enrolled at deploy time. |
| **Embedding** | Output of a Siamese twin; the vector used for similarity comparison. | Stored per identity at enrolment; compared at inference. Usually L2-normalised. |
| **Distance metric** | L2 (`‖x − y‖`) or cosine on embeddings. After L2-norm the two are equivalent. | Pick whichever the loss formula uses; verify with `torch.cdist` or `sklearn.metrics.pairwise.cosine_distances`. |
| **Contrastive loss** | `y·d² + (1−y)·max(0, m − d)²` on pairs — pull same together, push different apart up to margin. | Simpler than triplet, fine for pair-labelled data; rare in modern production. |
| **Triplet loss** | `max(0, d(a,p)² − d(a,n)² + α)` on (anchor, positive, negative) — enforce relative ordering. | Modern face-recognition standard (FaceNet); requires careful triplet sampling. |
| **Margin (α)** | Required gap between positive and negative distances. Prevents embedding collapse. | Tune via validation; `0.2` for unit-norm cosine, `1.0` for raw L2. |
| **Hard negative** | `d(a, n) < d(a, p)` — violates the margin most. Biggest gradient, can destabilise. | When you want max signal — but watch for label noise. |
| **Semi-hard negative** | `d(a, p) < d(a, n) < d(a, p) + α` — within margin. The practical sweet spot. | What FaceNet uses; default for online mining. |
| **Easy negative** | `d(a, n) > d(a, p) + α` — gradient = 0, useless for training. | Almost all random triplets — why mining is mandatory. |
| **Hard-negative mining** | Per-batch strategy to pick informative negatives (semi-hard typically). | Required for triplet training — without it, gradient stalls within one epoch. |
| **Face verification** | "Same person?" decision on a pair; threshold on embedding distance. | The canonical Siamese application; powers every face-unlock system. |

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

<a id="9g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 9 concept. The recap cheat sheet below summarises after.

#### Generative model (vs discriminative)

> **🪜 Mental model:** *Discriminative draws the line; generative draws the picture.* A discriminative model classifies; a generative model invents new samples.

**What it is.** A **generative model** is one that learns to *sample new data points* from the same distribution as the training data — given a noise input, produce a realistic image. By contrast, a **discriminative model** (the kind we've used in every prior module) learns to *classify* — given an input, predict a label.

Formally:
- Discriminative: models `P(y | x)` — "given this image, what's the class?"
- Generative: models `P(x)` directly (implicit or explicit) — "what does a realistic image look like?"
- Conditional generative: models `P(x | y)` — "what does an image of class y look like?"

**Why it matters.** Generative models are the engine behind every "AI art" tool (Midjourney, Stable Diffusion, DALL·E), face-aging apps, deepfakes, super-resolution, in-painting, data augmentation for rare classes, and synthetic training data. They've also become a huge interview topic for senior CV roles.

**How it works (high level).** Generative model families:
- **GANs** (this module) — train two networks adversarially (generator + discriminator).
- **VAEs** (Variational Autoencoders) — encode-decode with a probabilistic latent space.
- **Autoregressive** (PixelCNN, image transformers) — predict one pixel at a time.
- **Diffusion** (DDPM, Stable Diffusion) — iteratively denoise pure noise; current SOTA for image generation.

**Where it's used.** Image synthesis (Midjourney, DALL·E, Stable Diffusion). Style transfer (CycleGAN). Super-resolution (SRGAN, ESRGAN). Data augmentation for rare classes. Anomaly detection (anything that doesn't fit the learned distribution is anomalous).

**Related terms.**
- **Discriminative model** — the kind we trained in Modules 1–8.
- **VAE / Diffusion / Autoregressive** — sibling generative families.
- **Latent space** — the internal representation generative models learn to traverse.

**Gotcha.** GANs implicitly model `P(x)` — there's no closed-form density. You can sample but not evaluate "how likely is this specific image" — that's a VAE / diffusion strength, not a GAN one.

#### Generator (G)

> **🪜 Mental model:** *Counterfeiter starting from random scribbles.* Given a noise vector, produce a fake image that looks real.

**What it is.** The **generator** is a neural network that maps a low-dimensional random noise vector `z` (typically 100–256 dimensions, sampled from a Gaussian or uniform distribution) to a full-size image. Architecturally it's a *reverse CNN*: starts with a Dense layer that reshapes `z` into a small spatial tensor, then a stack of **transposed convolutions** (or upsample + conv) progressively grow the spatial size while shrinking the channel count, ending with a final layer that produces the output image's `(H, W, 3)` shape with a **tanh** activation.

**Why it matters.** The generator is the half of the GAN you keep at deployment time — at inference you sample noise, run it through G, and get a fresh image. Training G is the whole point; the discriminator is just a learned loss function.

**How it works.**
1. Sample `z ~ N(0, I)`, shape `(batch, z_dim)`.
2. Pass through Dense → reshape to small spatial tensor, e.g., `(4, 4, 512)`.
3. Apply repeated `Conv2DTranspose(stride=2) → BN → ReLU` blocks until spatial size = output size.
4. Final `Conv2DTranspose(3, kernel=4, strides=2, activation='tanh')` produces the image in `[-1, 1]`.

**Where it's used.** Anywhere you want to generate fresh images: art tools, data augmentation, anomaly detection (deviation from G's distribution = anomaly). At deployment, you use only G — never D.

**Related terms.**
- **Discriminator** — the other half (next entry).
- **Latent vector `z`** — G's input.
- **Conv2DTranspose** — the main upsampling op inside G.
- **`tanh` output** — matches the `[-1, 1]` normalisation of real images.

**Gotcha.** G's output range must match the real data's range. If real images are in `[-1, 1]` (tanh) but you normalised them to `[0, 1]`, training will fail silently — G produces nonsense.

#### Discriminator (D)

> **🪜 Mental model:** *Detective.* Given an image, output a single probability: "is this real or did G make it?"

**What it is.** The **discriminator** is a standard CNN classifier that takes an image and outputs a scalar — the probability the image is real. Architecturally it's just a regular CNN: strided convs (no pooling), LeakyReLU, optional BatchNorm, ending in a Dense layer producing a single logit or probability.

**Why it matters.** D's purpose during training is to act as a **learned loss function** for G. As D gets better at spotting fakes, G's gradient signal gets sharper. At deployment, D is discarded — you only keep G.

**How it works.**
1. Input: image in `[-1, 1]`.
2. Apply `Conv2D(stride=2) → LeakyReLU(0.2) → (BN)` blocks, halving spatial size each time.
3. Flatten and Dense(1).
4. Apply sigmoid (or output logits and use BCE-with-logits loss).
5. Trained with binary cross-entropy against label 1 for real, 0 for fake.

**Where it's used.** During GAN training, called twice per step (once on real, once on fake). After training, discarded. Modern GANs sometimes call it the **"critic"** (WGAN terminology) when it outputs an unbounded score instead of a probability.

**Related terms.**
- **Generator** — the other half.
- **LeakyReLU** — preferred over plain ReLU in D (keeps gradient alive for negatives).
- **Critic** — WGAN's name for D (outputs raw score, no sigmoid).

**Gotcha.** Don't put BatchNorm on D's first layer — it would prevent D from learning the input distribution's statistics. Common DCGAN rule: BN everywhere except D's input and G's output.

#### Adversarial training (minimax game)

> **🪜 Mental model:** *Counterfeiter vs police, co-evolving.* G learns to fool D; D learns to spot G's fakes. Each makes the other better.

**What it is.** **Adversarial training** is the procedure for training a GAN: G and D are trained in alternation, each trying to defeat the other. The original formulation is a **minimax** game:

`min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 − D(G(z)))]`

In words: *D wants to maximise the expression — assign log D ≈ 0 (high probability) to real images and log(1 − D(G(z))) ≈ 0 (low probability for fakes) to fakes; G wants to minimise the expression by fooling D so that D(G(z)) ≈ 1, making log(1 − D(G(z))) very negative.*

At Nash equilibrium, D outputs 0.5 for everything (it can't distinguish real from fake), and G samples from the true data distribution.

**Why it matters.** This co-evolution is what makes GANs work at all. Each network's progress depends on the other; if either gets too far ahead, training collapses. Understanding the minimax structure is essential for diagnosing GAN training failures.

**How it works (one training step).**
1. Sample noise `z`; G produces fake `x_fake = G(z)`.
2. **D step:** compute D loss = `BCE(D(x_real), 1) + BCE(D(x_fake), 0)`. Backprop, update D weights.
3. **G step:** compute G loss = `BCE(D(G(z)), 1)` (the **non-saturating** form). Backprop, update G weights *only*.
4. Repeat for many steps.

**Where it's used.** Inside the training loop of every GAN. Implementations differ in detail (Wasserstein critic, gradient penalty, two-time-scale updates) but the alternating structure is universal.

**Related terms.**
- **Nash equilibrium** — the theoretical stopping point where neither can improve.
- **Non-saturating G loss** — `−log D(G(z))` instead of `log(1 − D(G(z)))`; better gradients.
- **Vanishing gradient** — when D wins too hard; G receives near-zero gradient.

**Gotcha.** Use the **non-saturating** G loss — `−log D(G(z))` — instead of the saturating original. The saturating form has near-zero gradient exactly when D is winning, which is exactly when G needs the strongest signal.

#### Latent space / noise vector `z`

> **🪜 Mental model:** *Recipe dial.* The noise vector is a coordinate in an abstract space; different points produce different images. Smooth movements in `z` produce smooth interpolations in image space.

**What it is.** The **latent vector** `z` is the random input to the generator. Typically a 100–256-dimensional vector sampled from a standard Gaussian (or uniform). The **latent space** is the high-dim space of possible `z` values. After training, the generator has learned a mapping from this space to image space; different `z`s produce different images, and linear interpolations in `z`-space often produce smooth morphs between images.

**Why it matters.** Latent space is where all the "magic" of GANs lives — interpolating between two `z`s morphs one face into another; specific directions in `z` correspond to specific attributes (gender, age, glasses); StyleGAN's whole appeal is its disentangled latent space. Understanding `z` is foundational for any control over what G generates.

**How it works.** Each forward pass through G starts with a fresh `z ~ N(0, I)`. There's no memory of past `z`s. Conditional GANs (next entry) concatenate `z` with a class label or text embedding to control the output.

**Where it's used.** Image generation (sample `z`, get a new face). Latent-space arithmetic (`z_smiling = z_neutral + smiling_direction`). StyleGAN-style attribute editing. Generative super-resolution (`z` represents missing high-frequency content).

**Related terms.**
- **Embedding** — Module 4's notion; similar but learned for *similarity*, not generation.
- **Disentangled latent space** — what StyleGAN aims for: each `z` dimension controls one semantic attribute.
- **Truncation trick** — sample `z` from a truncated Gaussian to bias toward "average" images; common in StyleGAN.

**Gotcha.** Standard GANs have *entangled* latent space — moving in one direction changes many attributes at once. StyleGAN's `W` space is the engineered fix.

#### Training instability and mode collapse

> **🪜 Mental model:** *G finds a cheat code.* Instead of learning the full data distribution, G learns to produce only a handful of "safe" images that consistently fool D.

**What they are.** Two failure modes of GAN training:
- **Training instability** — losses oscillate wildly; samples flip between coherent and noisy; one player dominates and the other collapses. Caused by the delicate minimax balance.
- **Mode collapse** — G produces only a small subset of the data distribution (e.g., only one face, or only photos in profile view) because that subset reliably fools D. The variety of outputs collapses.

**Why it matters.** These are the two most-cited reasons GANs are hard to train. Recognising and fixing them is a frequent interview question. Modern variants (WGAN-GP, spectral normalisation, two-time-scale updates) exist specifically to mitigate these.

**How to diagnose.**
- **Loss curves alone don't reveal mode collapse** — G's loss may stay reasonable while diversity tanks.
- **Sample many `z`s** and look at the variety. If most samples look identical → collapse.
- **FID** (next entry) catches mode collapse because it measures distributional distance, not just per-sample quality.

**Fixes.**
- **Minibatch discrimination** — D sees stats across the batch and can detect "all samples look the same."
- **WGAN / WGAN-GP** — Wasserstein loss with smoother gradients.
- **Spectral normalisation** — bounds D's Lipschitz constant for stability.
- **Two-time-scale updates (TTUR)** — different LRs for G and D.
- **Larger batch size** — more diversity inside each batch.

**Related terms.**
- **Vanishing gradient** — D wins too hard; G stops learning.
- **Nash equilibrium** — the theoretical balanced state.
- **FID** — the metric that catches collapse.

**Gotcha.** Mode collapse doesn't show up in the loss curve — you must inspect samples. A GAN that "looks like it's converging" can be silently collapsed.

#### DCGAN (the convolutional baseline)

> **🪜 Mental model:** *The cookbook for stable GAN training.* A set of architectural rules that turn unstable GAN training into a reproducible recipe.

**What it is.** **DCGAN** (Radford et al., 2016) — the **D**eep **C**onvolutional **G**AN — is a set of architectural and hyperparameter guidelines that made GAN training reliable for the first time. The recipe:
- G and D are both fully convolutional (no Dense in the middle).
- Use **strided convolutions** for downsampling in D and **transposed convolutions** for upsampling in G — *not* pooling.
- **BatchNorm** in both G and D, **except** on G's output layer and D's input layer.
- **LeakyReLU(0.2)** in D, **ReLU** in G.
- **`tanh`** activation on G's output.
- **Adam optimiser** with `lr=2e-4, β1=0.5` (the low β1 is critical).
- Normalise real images to `[-1, 1]` to match G's tanh output.

**Why it matters.** Before DCGAN, GANs were research curiosities — too unstable to use. DCGAN is the baseline every modern GAN paper compares to. Memorise the recipe — it's the answer to "how do you build a stable GAN."

**How it works.** No new theory — just a careful combination of choices that empirically stabilise the minimax dynamics. The strided/transposed conv pair gives the network the right inductive bias for image generation. BatchNorm regularises training. Adam with low β1 helps the optimiser cope with non-stationary loss landscapes.

**Where it's used.** As the starting point for almost every GAN project. Anime-face generation, MNIST/CIFAR demos, learning GANs from scratch. Modern variants (StyleGAN, BigGAN) build on top of DCGAN's bones.

**Related terms.**
- **Conv2DTranspose** — G's upsampling block.
- **LeakyReLU** — D's activation.
- **Spectral normalisation** — modern stability improvement over BatchNorm in D.
- **WGAN-GP** — modern loss replacement.

**Gotcha.** The `β1=0.5` in Adam is non-negotiable. Default `β1=0.9` makes GAN training diverge — the momentum is too high for the non-stationary loss surface.

#### Conditional GAN (cGAN)

> **🪜 Mental model:** *Tell the generator what to draw.* Concatenate a class label (or text embedding) with `z`, so G can produce "a cat" or "a dog" on demand.

**What it is.** A **Conditional GAN** is a GAN where both G and D receive an extra input — a **condition** (typically a class label, but can be text, a sketch, another image, etc.). G's input becomes `(z, y)`; D's input becomes `(image, y)`. The model learns `P(x | y)` instead of just `P(x)`, letting you control what gets generated.

**Why it matters.** Vanilla GANs sample uniformly from the data distribution — you can't ask for "a face wearing glasses" specifically. cGANs add that control. The framework generalises to text-to-image (pix2pix, GauGAN), image-to-image translation, and super-resolution.

**How it works.** Concatenate the condition (one-hot label or learned embedding) with `z` for G's input, and with the image features for D's input. Train with the standard adversarial loss; the condition acts as an extra supervisor.

**Where it's used.** Class-conditional generation (per-digit MNIST, per-class ImageNet). Image-to-image translation (pix2pix). Text-to-image precursors (StackGAN). Super-resolution (SRGAN — condition on the low-res image).

**Related terms.**
- **Class embedding** — a learned vector per class, concatenated with `z`.
- **AC-GAN** — auxiliary classifier GAN; D also predicts the class.
- **pix2pix** — image-to-image cGAN with paired training data.
- **CycleGAN** — unpaired image-to-image translation; two cGANs + cycle consistency.

**Gotcha.** D must see the condition too. If only G is conditioned, D can't tell whether G is producing the right class — and G has no incentive to honour the condition.

#### Modern GANs (StyleGAN, briefly)

> **🪜 Mental model:** *Disentangled style control.* Instead of one tangled latent space, StyleGAN gives you a *layered* style code — coarse style at low resolution, fine style at high resolution.

**What it is.** **StyleGAN** (Karras et al., 2018; StyleGAN2 2020; StyleGAN3 2021) is the state-of-the-art GAN for high-resolution face synthesis. Its key innovations:
- **Mapping network** — `z` is first transformed by an MLP into an intermediate latent code `w`, which has a more disentangled structure.
- **Adaptive Instance Normalisation (AdaIN)** — `w` is injected at every layer of G, controlling the "style" at that resolution.
- **Noise injection** — random per-pixel noise at each layer adds stochastic detail (hair strands, skin pores).
- **Progressive growing / multi-resolution** — train at increasing resolutions.

**Why it matters.** StyleGAN's photorealism and *attribute control* (interpolate `w` to age a face, change hair colour, etc.) defined the SOTA for face synthesis for years and is the foundation of deepfake tech. Knowing StyleGAN's main ideas is expected of senior CV candidates.

**How it works at a glance.** The `w` space is *much more disentangled* than `z` — moving in one direction in `w` corresponds (often) to changing one attribute. Different layers of G control different scales — early layers govern face shape, late layers govern fine details. This separation is the "style" in StyleGAN.

**Where it's used.** Face generation (thispersondoesnotexist.com). Face editing / morphing apps. Deepfake creation (StyleGAN + face-swap pipelines). Synthetic training data.

**Related terms.**
- **BigGAN** — Google's large-batch class-conditional GAN; SOTA for ImageNet classes.
- **CycleGAN** — unpaired image translation.
- **Diffusion models** — newer generative family, now dominating image synthesis.
- **AdaIN** — StyleGAN's style-injection mechanism.

**Gotcha.** StyleGAN's photorealism is for a *single domain* (faces, cars, churches) per trained model. Cross-domain generation (faces → cats) requires retraining or extensions like StyleGAN-NADA.

### 🧠 Concept cheat sheet (recap)

> Recap table — every row is 2–3 lines: *what + when*. Full guided introductions live in [the walkthrough above](#9g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Generative model** | Learns `P(x)` (or `P(x | y)`) to sample new data points; sibling of discriminative `P(y | x)`. | When you need to *create* images — art tools, augmentation, super-res, anomaly detection. |
| **Generator (G)** | NN mapping `z` (noise) → fake image. Strided transposed convs, BN, ReLU, tanh output. | At deployment, the only network you keep. |
| **Discriminator (D)** | NN mapping image → P(real). Strided convs, LeakyReLU, sigmoid. | During training only — acts as a learned loss for G. Discarded at deployment. |
| **Adversarial training** | Minimax game: G fools D, D spots G. Alternate D-step and G-step in the training loop. | Every GAN training loop; the core procedure that makes GANs work. |
| **Latent vector `z`** | Random `z ~ N(0, I)`, dim 100–256. G's input. Coordinates in latent space. | At inference, sample `z` to produce a new image; interpolate `z` for morphs. |
| **Mode collapse** | G produces only a few outputs that reliably fool D. Loss curves don't reveal it — inspect samples. | Diagnose by sample variety; fix with minibatch discrimination, WGAN-GP, spectral norm. |
| **Training instability** | Losses oscillate; one player dominates; samples degenerate. | Use DCGAN guidelines, WGAN-GP, TTUR; be patient and inspect samples often. |
| **DCGAN** | Stable-GAN recipe: strided convs + BN + LeakyReLU + tanh + Adam(2e-4, β1=0.5). | Default starting point for any GAN project; memorise the recipe. |
| **Conditional GAN** | G(z, y) and D(image, y) — condition on a label/text/image to control generation. | When you need per-class generation, text-to-image, or image-to-image translation (pix2pix). |
| **StyleGAN** | SOTA face GAN with mapping network, AdaIN style injection, layered control. | When you need photorealistic single-domain generation with attribute control (faces, cars). |
| **FID** | Fréchet Inception Distance — distance between real and generated distributions in Inception space. | Standard GAN evaluation metric; lower is better. Catches mode collapse. |
| **WGAN** | Replace JS-divergence loss with Wasserstein (Earth Mover's) distance; smoother gradients. | When DCGAN won't stabilise — also try WGAN-GP (gradient penalty) for best results. |

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

> Every entry is a 2–4 sentence beginner-friendly definition. The bracketed link goes to the term's full walkthrough entry in the relevant module.

| Term | Definition |
|---|---|
| **Anchor box** | A predefined reference rectangle (with a specific size and aspect ratio) assigned to each spatial position in a detector's output grid. Instead of regressing absolute bbox coordinates (hard), the network predicts small *offsets* relative to the anchor — much easier to learn. Faster R-CNN uses 9 anchors per location; YOLO clusters them from training data via K-means. ([walkthrough](#5g-guided)) |
| **Augmentation** | Random label-preserving image transformations (horizontal flip, rotation, crop, brightness/contrast jitter, MixUp/CutMix) applied to training images on the fly. Each epoch the network sees slightly different versions of each image, learning *invariance* and effectively multiplying your training set. The cheapest single accuracy booster for small datasets; never apply to val/test. ([walkthrough](#2g-guided)) |
| **Backbone** | The convolutional portion of a pretrained model — typically all layers up to but not including the final classifier head. Get one with `keras.applications.<Name>(weights='imagenet', include_top=False)`. Holds 99% of the parameters and the ImageNet knowledge; in transfer learning you keep the backbone and replace the head. ([walkthrough](#3g-guided)) |
| **BatchNorm** | Per-batch, per-channel normalisation layer that subtracts the batch mean, divides by the batch standard deviation, and then applies a learnable scale + shift (`γ·x + β`). Hugely stabilises training and allows larger learning rates. At inference time it uses running averages instead of the current batch's stats — forgetting `training=False` is a classic stealth bug. ([walkthrough](#2g-guided)) |
| **Bounding box (bbox)** | The axis-aligned rectangle around an object, described by 4 numbers. Two formats dominate: corner `(x1, y1, x2, y2)` (PASCAL VOC, COCO, torchvision) and centre `(xc, yc, w, h)` (YOLO, often normalised to [0, 1]). Mixing formats silently produces wrong IoU/NMS results. ([walkthrough](#5g-guided)) |
| **Computer Vision** | The subfield of AI that takes images or video as input and produces a useful answer — a label, a bounding box, a mask, a generated image, an embedding. Modern CV is built on deep neural networks (CNNs and Vision Transformers) trained end-to-end on labelled data, replacing the hand-engineered features of classical CV. ([walkthrough](#1g-guided)) |
| **Channel** | One 2-D plane of an image tensor. Grayscale images have 1 channel; RGB has 3 (red, green, blue); satellite imagery can have 10+ channels. The third dim in an `(H, W, C)` tensor. Conv kernels operate per channel and sum across them. ([walkthrough](#1g-guided)) |
| **Conditional GAN (cGAN)** | A GAN where both G and D receive an extra input — typically a class label or text embedding — letting you control *what* gets generated ("a cat" vs "a dog"). Generalises to pix2pix (image-to-image), text-to-image, and CycleGAN. D must see the condition too, or G has no reason to honour it. ([walkthrough](#9g-guided)) |
| **Conv2D / Convolution** | The core operation of a CNN: slide a small learnable kernel over the input, computing a dot product at every position to produce one output number per position. The collection of outputs is a feature map. Captures locality (only looks at neighbours), weight sharing (same kernel everywhere), and compositionality (stacked layers build complex features). ([walkthrough](#1g-guided)) |
| **Conv2DTranspose** | Learnable upsampling layer (sometimes mis-called "deconvolution"). Produces a larger spatial output by applying a learnable kernel — used in GAN generators and segmentation decoders. Can produce visible checkerboard artefacts when `kernel_size % stride != 0`; safer alternative is `UpSampling2D + Conv2D`. ([walkthrough](#7g-guided)) |
| **Contrastive loss** | Pair-based metric-learning loss: pull same-identity pairs together (`d²`), push different-identity pairs apart up to a margin (`max(0, m − d)²`). Simpler than triplet but generally produces less sharp embeddings. ([walkthrough](#8g-guided)) |
| **Cosine similarity** | `(x · y) / (‖x‖ · ‖y‖)` — the cosine of the angle between two vectors, ignoring magnitude. Range `[-1, 1]`. After L2-normalisation, equivalent to (negative) Euclidean distance for ranking. The default similarity metric for image and text embeddings. ([walkthrough](#4g-guided)) |
| **DCGAN** | "Deep Convolutional GAN" — a 2016 set of architectural and hyperparameter rules that made GAN training stable for the first time. Recipe: strided convs (no pooling), BN everywhere except G output and D input, LeakyReLU(0.2) in D, ReLU + tanh in G, Adam(2e-4, β1=0.5). Memorise the recipe. ([walkthrough](#9g-guided)) |
| **Dice coefficient / loss** | Overlap-based segmentation metric: `2|A∩B| / (|A| + |B|)`. Equivalent to F1 score for masks. As a *loss* (`1 − Dice`), it's the standard choice for imbalanced binary masks where per-pixel cross-entropy is dominated by background pixels. ([walkthrough](#7g-guided)) |
| **Dilated / atrous convolution** | A convolution with kernel taps spaced apart by a dilation factor. Expands the receptive field *without* downsampling — keeps spatial resolution high while seeing global context. The signature trick of DeepLab segmentation. ([walkthrough](#7g-guided)) |
| **Discriminator (GAN)** | The "police" half of a GAN — a CNN that takes an image and outputs the probability it's real. Trained with binary cross-entropy against label 1 for real, 0 for G's fakes. Discarded at deployment time. ([walkthrough](#9g-guided)) |
| **Dropout** | Regularisation layer that randomly zeros a fraction `p` of activations during training only (a no-op at inference). Forces the network to spread information across redundant neurons; acts as an implicit ensemble of sub-networks. Typical rates: 0.5 for Dense, 0.1–0.25 for Conv. ([walkthrough](#2g-guided)) |
| **Embedding** | A fixed-length dense vector that represents an input (image, word, audio clip) in a way where "similar inputs → close vectors." For CV, usually the output of the penultimate layer of a pretrained CNN. Foundation of similarity search, clustering, dedup, and verification. ([walkthrough](#4g-guided)) |
| **Encoder-decoder** | An architectural pattern where the network first downsamples the input into a small bottleneck representation (encoder), then upsamples it back to the input size (decoder). U-Net, FCN, and segmentation networks in general use this shape, often with skip connections to preserve spatial detail. ([walkthrough](#7g-guided)) |
| **Euclidean distance** | `‖x − y‖₂ = √(Σ (x_i − y_i)²)` — straight-line distance between two vectors. Range `[0, ∞)`. After L2-normalisation, ranks neighbours identically to cosine. ([walkthrough](#4g-guided)) |
| **Face verification** | The task of deciding whether two face images depict the same person. Pipeline: detect → align → embed (via Siamese / FaceNet / ArcFace) → measure distance → threshold. The canonical Siamese-network application; behind every face-unlock system. ([walkthrough](#8g-guided)) |
| **FAISS** | Facebook's library for billion-scale approximate nearest-neighbour search. Supports inverted indices (IVF), product quantisation (PQ), and HNSW graphs; CPU and GPU. The standard ANN backend for production image-retrieval systems. ([walkthrough](#4g-guided)) |
| **Feature map** | The 2-D output of a single convolutional filter — a heat-map of "where this pattern matched." A Conv2D layer with `Cout` filters produces `Cout` stacked feature maps as output. Carries the network's internal representation between layers. ([walkthrough](#1g-guided)) |
| **FID (Fréchet Inception Distance)** | Standard GAN evaluation metric. Compute Inception-V3 features for many real and many generated images, fit Gaussians to each, and compute the Fréchet distance between the two Gaussians. Lower = closer distributions = better GAN. Catches mode collapse. ([walkthrough](#9g-guided)) |
| **Fine-tuning** | Transfer-learning recipe: after feature extraction, unfreeze the top few backbone layers and continue training with a 10× smaller LR (typically `1e-5`). Gains another 2–10% accuracy. Risk: too-large LR causes catastrophic forgetting of ImageNet features. ([walkthrough](#3g-guided)) |
| **Flatten** | Reshape layer that collapses a `(B, H, W, C)` tensor into `(B, H·W·C)`. No parameters; just rearranges memory. Bridges the conv backbone to a Dense classifier head. Modern alternative: `GlobalAveragePooling2D`. ([walkthrough](#1g-guided)) |
| **Focal Loss** | Variant of cross-entropy: `−α(1 − p)^γ log(p)`. The `(1 − p)^γ` factor down-weights easy examples (high predicted probability) so gradient focuses on hard, misclassified ones. Solves single-stage detection's class imbalance; the key innovation of RetinaNet. ([walkthrough](#6g-guided)) |
| **FPN (Feature Pyramid Network)** | A top-down pathway that combines features from multiple backbone scales — high-resolution shallow features get semantic context from deeper layers. Used in modern YOLO, RetinaNet, Mask R-CNN. Fixed small-object detection. ([walkthrough](#6g-guided)) |
| **Generator (GAN)** | The "counterfeiter" half of a GAN — a reverse-CNN that maps a random noise vector `z` to a fake image. Strided transposed convs + BN + ReLU + tanh output is the DCGAN recipe. The only network you keep at inference. ([walkthrough](#9g-guided)) |
| **GlobalAveragePooling2D (GAP)** | Pooling layer that averages each feature map down to a single number. Replaces `Flatten + Dense(huge)` at the end of a CNN — slashes the classifier head's parameter count by ~100×. The default modern choice. ([walkthrough](#2g-guided)) |
| **Grid cell (YOLO)** | One `(i, j)` position in YOLO's `S × S` output grid. Responsible for any object whose centre falls inside it; predicts `B` anchor boxes with offsets, objectness, and class probabilities. ([walkthrough](#6g-guided)) |
| **Hard negative / mining** | A "hard negative" in metric learning is one where `d(a, n) < d(a, p)` — already misclassified. Hard-negative mining is the strategy of intentionally picking informative negatives during training, since random sampling yields mostly easy negatives that produce zero gradient. FaceNet uses *semi-hard* negatives for stability. ([walkthrough](#8g-guided)) |
| **ImageNet** | The canonical CV pretraining dataset — 1.28M images across 1 000 classes (ILSVRC-2012 subset). Almost every pretrained CNN started from training on ImageNet; "ImageNet-pretrained" is the implicit default for transfer learning. ([walkthrough](#3g-guided)) |
| **Image as tensor** | An image inside the computer is a 3-D array of pixel intensities: `(H, W, C)` in Keras/TF (channels-last) or `(C, H, W)` in PyTorch (channels-first). Pixel values are 0–255 uint8 raw, or 0–1 float after rescaling. Batched data adds a leading axis. ([walkthrough](#1g-guided)) |
| **Instance segmentation** | Per-pixel class label *plus* instance ID — different cats get different IDs in the output mask. Required when you need to count or track individual objects (cell biology, retail, AR). Canonical model: Mask R-CNN. ([walkthrough](#7g-guided)) |
| **IoU (Intersection over Union)** | `area(A ∩ B) / area(A ∪ B)` — overlap quality between two regions (boxes for detection, masks for segmentation). Range `[0, 1]`; 0.5 is the standard "match" threshold in detection. Drives NMS and mAP. ([walkthrough](#5g-guided)) |
| **Kernel / filter** | The small `(kH, kW, Cin)` array of learnable weights used in one convolution. After training, each filter has tuned itself to detect a specific pattern (edge, texture, object part). `Cout` filters → `Cout` feature maps. ([walkthrough](#1g-guided)) |
| **Latent vector (`z`)** | The random input to a GAN's generator — typically 100–256 numbers sampled from a standard Gaussian. Coordinates in *latent space*. Linear interpolation between two `z`s usually produces smooth morphs between generated images. ([walkthrough](#9g-guided)) |
| **LeakyReLU** | Activation `f(x) = x if x > 0 else 0.2·x`. Unlike plain ReLU, the negative slope keeps a small gradient flowing for negative inputs — fixes the "dead neuron" problem and is the standard activation in GAN discriminators. ([walkthrough](#9g-guided)) |
| **L1 / L2 regularization** | Loss penalties that discourage large weights. L1 adds `λ Σ |w|` (drives weights *exactly* to 0 → sparse). L2 adds `λ Σ w²` (smooth shrinkage toward 0 but not to 0; the default for CNNs). Also called weight decay. ([walkthrough](#2g-guided)) |
| **L2 normalisation** | Divide each embedding by its L2 norm so it lies on the unit sphere (length 1). Removes magnitude bias and makes cosine similarity equivalent to Euclidean distance for ranking. Always normalise before NN search on image embeddings. ([walkthrough](#4g-guided)) |
| **mAP (mean Average Precision)** | The standard object-detection evaluation metric. For each class, compute the area under the precision-recall curve (= AP), then average across classes. Reported at a specific IoU threshold (PASCAL@0.5) or averaged across `[0.5, 0.55, …, 0.95]` (COCO). ([walkthrough](#5g-guided)) |
| **Margin (α)** | Required gap between positive and negative pair distances in contrastive/triplet loss. Without a margin, embeddings collapse to a point. Typical values: 0.2 for unit-norm cosine, 1.0 for raw L2. ([walkthrough](#8g-guided)) |
| **Mask R-CNN** | Faster R-CNN + a small per-ROI mask head (an FCN producing `28×28` masks) + ROI Align (bilinear, no quantisation). The canonical *instance* segmentation model — gives you bboxes, classes, and per-instance masks in one pass. ([walkthrough](#7g-guided)) |
| **MaxPooling2D** | Fixed downsampling layer that takes the maximum value in each pooling window (default `2×2, stride=2` halves spatial dims). No learnable parameters. Provides mild translation invariance and shrinks memory; modern alternative is strided conv. ([walkthrough](#1g-guided)) |
| **Mode collapse** | GAN failure mode where the generator produces only a few "safe" outputs that consistently fool D, ignoring most of the data distribution. Loss curves don't reveal it — you must inspect sample diversity. Fix with minibatch discrimination, WGAN-GP, spectral normalisation. ([walkthrough](#9g-guided)) |
| **Nearest-neighbour search** | Algorithm "given a query embedding, return the `k` stored points closest to it." Brute force is `O(N·D)` per query; production systems use approximate NN (Annoy, FAISS) for sub-linear time. The retrieval workhorse. ([walkthrough](#4g-guided)) |
| **NMS (Non-Maximum Suppression)** | Greedy post-processing for detectors: sort boxes by confidence, keep the highest, drop all overlaps above an IoU threshold (e.g., 0.5), repeat. Deduplicates the many overlapping boxes detectors emit per object. Run per-class to allow legitimate cross-class overlaps. ([walkthrough](#5g-guided)) |
| **Objectness score** | Per-anchor scalar predicted by single-stage detectors saying "is there an object centred here?" Defined as `P(object) × IoU(pred, GT)` — combines presence with bbox accuracy. Multiplied with class probs to get the final confidence. ([walkthrough](#6g-guided)) |
| **One-shot learning** | Recognising a new class from a *single* labelled example. Standard classifiers can't do it (need many examples per class); Siamese networks can (learn a similarity function, then enrol new identities by storing one embedding). ([walkthrough](#8g-guided)) |
| **Padding** | Extra (usually zero) pixels added around the input image so the kernel can sit at the edges. `'same'` padding adds enough zeros to preserve output spatial size at stride 1; `'valid'` adds none and the output shrinks by `(K−1)`. ([walkthrough](#1g-guided)) |
| **PCA** | Linear dimensionality reduction technique. Projects embeddings onto the top `k` directions of maximum variance. Typical for embeddings: 2048 → 150 dims with <0.5% retrieval accuracy loss and ~13× speedup. ([walkthrough](#4g-guided)) |
| **Pixel accuracy** | The fraction of pixels classified correctly in a segmentation output. *Misleading* on imbalanced masks — a model that predicts "background" everywhere scores 95%+ on portrait data. Use mIoU or Dice instead. ([walkthrough](#7g-guided)) |
| **Pooling** | Fixed (non-learnable) downsampling — MaxPool keeps the maximum of each window, AvgPool the mean. Default `pool_size=2, strides=2` halves spatial dims. ([walkthrough](#1g-guided)) |
| **Pre-trained model** | A neural network whose weights are already learned on some large dataset (almost always ImageNet for CV). Auto-downloaded by Keras / PyTorch on first use. The starting point of every transfer-learning project. ([walkthrough](#3g-guided)) |
| **Receptive field** | The region of input pixels that influences a single output activation deep in the network. Grows as you stack convs and pools. A deep ResNet has a receptive field covering most of the image; a shallow first-conv-layer has a 3×3 receptive field. |
| **Region proposal** | A candidate bounding box generated *before* classification in a two-stage detector — "something interesting might be here." Generated by Selective Search (R-CNN) or RPN (Faster R-CNN+). ([walkthrough](#5g-guided)) |
| **ReLU** | The default activation in deep CV: `f(x) = max(0, x)`. Non-saturating for positives (gradient = 1), trivial to compute, and free of the vanishing-gradient problem that plagued sigmoid/tanh in early deep nets. ([walkthrough](#1g-guided)) |
| **Residual block** | A ResNet building block with a skip connection: `out = F(x) + x`. The identity shortcut lets gradients flow through deep networks without vanishing — the reason 50–1000+ layer networks are trainable. ([walkthrough](#3g-guided)) |
| **ROI Align / ROI Pool** | Layers that crop a fixed-size feature patch (typically `7×7`) from variable-size proposals against the backbone's feature map. RoIPool quantises positions; RoIAlign uses bilinear interpolation (no quantisation) — required for pixel-precise masks in Mask R-CNN. ([walkthrough](#5g-guided)) |
| **RPN (Region Proposal Network)** | A small CNN inside Faster R-CNN that learns to generate region proposals end-to-end. Replaces the hand-crafted Selective Search of R-CNN, making the whole detector trainable as one network and ~10× faster. ([walkthrough](#5g-guided)) |
| **Semantic segmentation** | Per-pixel class labelling with no instance distinction — all cat pixels share the same label, you can't tell two cats apart. Canonical models: U-Net, FCN, DeepLab. Metric: mean IoU. ([walkthrough](#7g-guided)) |
| **Siamese network** | An architecture made of two (or more) identical sub-networks with **shared weights**, used to compare inputs via embedding distance. Trained with contrastive or triplet loss. The canonical tool for one-shot / verification tasks (face, signature, fingerprint). ([walkthrough](#8g-guided)) |
| **Skip connection** | A direct shortcut path from an earlier layer to a later one — either by *addition* (ResNet: `out = F(x) + x`) or *concatenation* (U-Net: `concat([encoder_features, decoder_features])`). Preserves information and gradients through deep networks. ([walkthrough](#7g-guided)) |
| **Softmax** | The final activation in a classifier: converts a vector of arbitrary "logits" into a probability distribution that sums to 1. Always paired with cross-entropy loss. |
| **SSD (Single-Shot Detector)** | Single-stage detector with multi-scale prediction — heads attached to several feature-map resolutions, so small objects come from shallow (high-res) maps and large objects from deep maps. SSD-MobileNet is the staple lightweight detector on mobile / edge. ([walkthrough](#6g-guided)) |
| **Stride** | The step size (in pixels) the convolutional kernel takes as it slides across the input. Default 1 (touch every pixel). Stride 2 skips every other position, roughly halving the output spatial size — a cheap learnable downsampler. ([walkthrough](#1g-guided)) |
| **StyleGAN** | State-of-the-art face GAN (2018–2021) introducing a mapping network from `z` to a disentangled `w` space, adaptive instance normalisation, and per-layer noise injection. Famous for photorealism and attribute control (interpolate `w` to age a face, change hair colour, etc.). ([walkthrough](#9g-guided)) |
| **t-SNE / UMAP** | Non-linear dimensionality reduction methods that project high-dim embeddings to 2-D or 3-D for *visualisation*. They distort distances dramatically — useful for inspecting clusters but **never use for retrieval**. ([walkthrough](#4g-guided)) |
| **Transfer learning** | The practice of reusing a model trained on a huge dataset (ImageNet) as the starting point for a new, related task. Freeze the pretrained backbone, attach a new task head, train the head, optionally fine-tune top layers. The default approach for any CV task with limited labelled data. ([walkthrough](#3g-guided)) |
| **Transposed convolution** | Learnable upsampling layer (`Conv2DTranspose`). Effectively the gradient operation of a strided conv run forward. Used in GAN generators and segmentation decoders. Can produce checkerboard artefacts when `kernel_size % stride != 0`. ([walkthrough](#7g-guided)) |
| **Triplet loss** | Metric-learning loss on triples `(anchor, positive, negative)`: `max(0, d(a,p)² − d(a,n)² + α)`. Enforces a *relative* ordering — anchor's positive must be closer than its negative by at least margin α. FaceNet's central loss; needs hard-negative mining to train usefully. ([walkthrough](#8g-guided)) |
| **U-Net** | Symmetric encoder-decoder segmentation architecture with skip connections at every level (concatenation, not addition). The canonical semantic-segmentation network, originally for biomedical microscopy; also the backbone of diffusion-model denoisers. ([walkthrough](#7g-guided)) |
| **Underfitting** | The model fails to fit even the training set — both train and val accuracy are low. Opposite of overfitting. Fix by increasing model capacity, training longer, or reducing regularisation — *not* by adding more dropout/augmentation. ([walkthrough](#2g-guided)) |
| **Upsampling (vs transposed conv)** | `UpSampling2D` is fixed (nearest or bilinear) interpolation with no learnable parameters; `Conv2DTranspose` is learnable upsampling. Inside U-Net decoders, `UpSampling2D + Conv2D` is cleaner (no checkerboard); GAN generators usually prefer `Conv2DTranspose`. ([walkthrough](#7g-guided)) |
| **Vanishing gradient** | When gradients become so small flowing back through many layers that early-layer weights stop updating. Plagued early deep nets using sigmoid/tanh. Fixed by ReLU, BatchNorm, and residual/skip connections. |
| **VGG** | 2014 CNN architecture by Oxford's VGG group; uniform stacks of 3×3 convolutions with 16 or 19 layers. Simple and historically important but parameter-heavy (138M / 144M); mostly used as a reference / teaching example now. ([walkthrough](#3g-guided)) |
| **WGAN** | Wasserstein GAN — replaces the JS-divergence loss of vanilla GANs with the **Wasserstein (Earth Mover's) distance**. Smoother gradients, less mode collapse. The "critic" (D) must be 1-Lipschitz; enforced via weight clipping (WGAN) or gradient penalty (WGAN-GP). ([walkthrough](#9g-guided)) |
| **YOLO** | "You Only Look Once" — the canonical single-stage detector. Divides the image into an `S × S` grid; each cell predicts a few anchor bboxes with offsets, objectness, and class probabilities, all in one forward pass. `ultralytics.YOLO('yolov5s.pt')(img)` is the modern one-liner. ([walkthrough](#6g-guided)) |

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
