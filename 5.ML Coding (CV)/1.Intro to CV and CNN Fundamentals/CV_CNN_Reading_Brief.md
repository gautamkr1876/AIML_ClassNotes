<a id="top"></a>
# CV & CNN — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~25 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`CV_CNN_Jargon_Card.md`](./CV_CNN_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `Intro_to_CV_and_CNN_Fundamentals (1).ipynb` in this folder.

---

## 🎯 30-second TL;DR

On image data, a small **CNN** beats a big **MLP**. The notebook proves this with a clothing-image classifier:

- **MLP** (vanilla neural network) → **~27% test accuracy**
- **CNN** (convolutional neural network) → **~51% test accuracy** (nearly double)

The reason is three properties of images that CNNs exploit and MLPs ignore:

1. **Locality** — pixels near each other are related; far-apart pixels usually aren't.
2. **Stationarity** — the same pattern can appear anywhere in an image.
3. **Compositionality** — complex shapes are built from simpler ones (edges → shapes → objects).

The notebook calls this the **screwdriver-vs-hammer** analogy: image data is a screw; a CNN is the screwdriver; a vanilla MLP is the hammer. You *can* hammer a screw — it just doesn't work well.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **What computer vision is** — classification, object detection, OCR, image generation; the challenges (different sizes, lighting, occlusion).
2. **Why a vanilla MLP fails on images** — flattening destroys spatial structure; parameter count explodes.
3. **Inspiration from the human visual cortex** — vision is **hierarchical**: low-level features (edges) → mid-level (shapes) → high-level (objects).
4. **CNN building blocks** — Convolution → Padding → Stride → Pooling → Flatten → Dense.
5. **🔥 Headline experiment** — train MLP and CNN on the **same clothing dataset**, watch CNN win.
6. **Convolution mechanics** — a manual 3×3 kernel sliding across a 6×6 image; the **Sobel filter** as a real example.
7. **RGB / multi-channel convolution** — why a kernel must match the input's depth (3×3 on RGB is actually 3×3×3).
8. **Padding and stride formulas** — how output shape depends on input, kernel size, padding, and stride.
9. **Pooling (max vs average)** — a shrinking step with **zero learnable parameters**.
10. **Three formulas to memorise** — conv output shape, conv param count, pool output shape.

---

## 🧠 The big idea — three properties of images

CNNs work because they **bake in** what we know about images. MLPs treat every pixel as an unrelated input variable. CNNs treat the image as an image.

Analogy — **spotting your friend in a photo.** You don't memorise pixel positions. You look for **edges** (where light meets dark), then **shapes** (eyes, nose, mouth), then **a face**. And you'd recognise your friend whether they're in the top-left or the bottom-right of the photo. That's exactly the three properties:

- **Locality** → your eye next to your eyebrow is meaningful; your eye next to a far-away tree isn't. CNNs only look at small neighbourhoods at a time.
- **Stationarity** → a face is a face, whether top-left or bottom-right. CNNs use the **same kernel everywhere** (parameter sharing), so a "face detector" works at any position.
- **Compositionality** → simple things stack into complex things. Stacked CNN layers grow their **receptive field** so deep layers see entire faces, even though each layer only does 3×3.

---

## 📖 Core concept primers

Five primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, and a worked example tied to the notebook.

### 1. Convolution + Kernel

> **🪜 Mental model:** a **stencil** sliding across the image. At each spot, the stencil produces one number summarising the patch underneath.

A **kernel** (also called a **filter**) is a small grid of numbers — usually 3×3. **Convolution** is the act of sliding it across the image. At each position, the kernel's numbers are multiplied by the underlying pixel values and the products are summed. The result is **one** number per position — and the whole sweep produces a new "image" called a **feature map**.

**Tiny worked example.** Input is a 6×6 grayscale image (so depth = 1). Kernel is 3×3.

```
Input (6×6)              Kernel (3×3)
[ . . . . . . ]          [ -1  0  1 ]
[ . . . . . . ]          [ -2  0  2 ]   ← Sobel vertical-edge detector
[ . . . . . . ]          [ -1  0  1 ]
[ . . . . . . ]
[ . . . . . . ]
[ . . . . . . ]
```

The kernel sits at position (0,0), multiplies its 9 numbers with the underlying 9 pixels, sums → output[0][0]. Slides one pixel right, repeats. Without padding, the output is 4×4 (6 - 3 + 1 = 4). With `padding="same"`, the output stays 6×6 because zeros are added around the edges.

**Why it matters in this notebook.** The CNN's first layer (Cell 56) has 16 such kernels: `Conv2D(filters=16, kernel_size=3, padding="same")`. Each kernel produces one feature map. **In a CNN, the kernel values are learned** — the network discovers which patterns are useful (edges, textures, etc.) on its own.

### 2. Stride & Padding

> **🪜 Mental model:** **stride** = how big a step the stencil takes. **Padding** = a zero-border so the stencil can reach the edges.

**Stride** is how many pixels the kernel jumps each step. Stride 1 = slide one pixel at a time. Stride 2 = skip every other pixel (output is roughly half the size).

**Padding** is a border of zeros added around the input before the kernel slides on it. Without padding, the output shrinks every layer (a 6×6 input with a 3×3 kernel gives a 4×4 output). With `padding="same"`, enough zeros are added so the output keeps the same height and width as the input.

**The formula** for the output side length:

```
o = (n + 2p - f) / s  +  1
```

In words: **the output side length = (input side + twice the padding − filter size) divided by the stride, plus one.**

- `n` = input side length (e.g., 6 if input is 6×6)
- `p` = padding (number of zeros on each side; 0 if no padding)
- `f` = filter side (e.g., 3 for a 3×3 kernel)
- `s` = stride

**Worked example.** `n=6, f=3, p=0, s=1`: `(6 + 0 − 3)/1 + 1 = 4`. So output is 4×4. ✅
With `padding="same"`: `p=1`, output is `(6 + 2 − 3)/1 + 1 = 6` — same as input. ✅

### 3. Pooling (Max / Average)

> **🪜 Mental model:** a shrinking step that keeps the strongest signal and throws away exact position.

**Pooling** scans the image with a small window (usually 2×2) and replaces each window with a single value: the **max** (max pooling) or the **mean** (average pooling). It **has no learnable parameters** — it's just a fixed rule.

**Why it exists.** Two reasons. First, it halves the data size (a 128×128 feature map becomes 64×64 after 2×2 max pool), making the network smaller and faster. Second, it gives the network **translation invariance** — if a feature shifts by one pixel, the max value in the patch is usually the same, so the output barely changes. That's exactly **compositionality** ("the model doesn't care exactly where the feature is, just that it's there").

**Pooling output formula:** `o = floor((n − f) / s) + 1`. By default, Keras uses `pool_size=2, stride=2`, so a 128×128 input becomes 64×64.

**Max pool vs avg pool.** Max pool is used **between** Conv layers (keeps the most-activated signal). Average pool is sometimes used **at the very end**, before the Dense layers, to summarise each feature map into one number.

### 4. Feature Map & Channels (RGB)

> **🪜 Mental model:** a kernel must be **as deep as its input**. RGB image (depth 3) → kernel is 3×3×3, not just 3×3.

A **feature map** is the output of one kernel sweeping across the input. One Conv2D layer with `filters=16` produces **16 feature maps**, stacked into a depth-16 output (this depth is the new "channel" axis for the next layer).

**Channels** are the depth of the data. An input RGB image has 3 channels (Red, Green, Blue). A 3×3 kernel applied to an RGB image is actually **3×3×3** — it has 9×3 = 27 numbers and looks at all three colour channels at once. The output is still a single 2D feature map, because the three depth-slices are summed.

**Why it matters in this notebook.** The first Conv2D layer takes a `(128, 128, 3)` RGB image. Each of its 16 kernels is `3×3×3 = 27` weights + 1 bias = **28 params per kernel**, ×16 kernels = **448 params** for that layer. Then MaxPool halves the spatial size → output is `(64, 64, 16)`.

### 5. Parameter Sharing & Sparse Connectivity (the locality argument)

> **🪜 Mental model:** Instead of "every pixel talks to every neuron" (MLP), it's "every small neighbourhood is checked by the same kernel" (CNN).

These are the two design tricks that make CNNs work.

- **Sparse connectivity** — Each output value only depends on a small patch (e.g., 3×3) of the input, not the entire input. This bakes in **locality**.
- **Parameter sharing** — The same kernel is used at every position. So the parameter count is **independent of input size**. This bakes in **stationarity**.

**Why it matters in this notebook.** This is the answer to "why does CNN win." It's not because CNNs have more capacity — they often have **less**. It's because the parameters they have are spent on the right things.

### 6. Why MLP fails on images (the parameter-explosion argument)

> **🪜 Mental model:** flattening a picture into a 1D row is like cutting a photo into pixel-confetti — every spatial relationship is gone.

When you feed an image to an MLP, the first step is to **flatten** it into a 1D vector. For a `128×128×3` image, that's `49,152` numbers. Then a Dense layer with, say, 1024 neurons needs `49,152 × 1024 + 1024 ≈ 50 million` parameters — just for the first layer.

**The notebook's bigger example.** For a `1280×720` image fed into a Dense(128) layer:
`1280 × 720 = 921,600` input pixels → `921,600 × 128 = 117,964,800` parameters in just the first dense layer. Infeasible.

And after all that — flattening **destroys spatial relationships**. Pixel (0,0) and pixel (0,1) are neighbours in the image but they become indices 0 and 1 in the flat vector; the model has no way to know they were related. That's why the MLP only reaches 27% accuracy in your notebook.

---

## 🔥 The headline experiment — at a glance

The notebook trains both models for the same number of epochs on the same clothing dataset and compares.

| | **MLP** (Vanilla NN) | **CNN** |
|---|---|---|
| Input handling | Flattens 128×128×3 → 49,152 numbers | Keeps 2D structure |
| Architecture | Flatten → Dense(1024) → Dense(256) → Dense(10) | Conv2D(16, 3×3) → MaxPool → Flatten → Dense(256) → Dense(10) |
| Activations | ReLU + final softmax | ReLU + final softmax |
| Optimizer / loss | Adam + sparse categorical cross-entropy | Adam + sparse categorical cross-entropy |
| **Test accuracy** | **~27%** | **~51%** (nearly 2× better) |
| Why | Treats pixels independently; flatten destroys structure | Exploits locality + parameter sharing + compositionality |

**Headline takeaway** (the notebook says it best): *"Where we have screw aka image data, using a specialist tool like screw-driver (CNNs) instead of a hammer (vanilla NN) proves to be more effective."*

---

## 🧮 Three formulas to memorise

### 1. Conv output shape

```
o = (n + 2p − f) / s  +  1
```

**In words:** output side = (input side + twice padding − filter size) ÷ stride + 1.
Symbols: `n` input side, `p` padding, `f` filter side, `s` stride.

### 2. Conv layer parameter count

```
params = ((m × n × d) + 1) × k
```

**In words:** each filter has `m×n×d` weights (filter height × filter width × **input depth**) plus 1 bias; multiply by `k` filters.
**Worked example from the notebook (Cell 92):** for a 3×3 kernel on an RGB (depth 3) input with 48 filters: `((3×3×3) + 1) × 48 = 28 × 48 = 1,334` params. (This is a teaching example in Part 2 — **not** the same as the headline-experiment CNN, which uses 16 filters.)

### 3. Pooling output shape

```
o = floor((n − f) / s)  +  1
```

**In words:** output side = floor of (input side − pool window size) ÷ stride, plus one. Note: **zero learnable parameters** — pooling is a rule, not something the network learns.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–15** | CV intro, applications (OCR, classification, generation), dataset loading | **Skim** — ~5 min. Just absorb the vocabulary; the Jargon Card has it all. |
| **16–45** | Data preprocessing → MLP training → MLP evaluation (gets ~27%) | **Read normally** — ~10 min. Note the accuracy. |
| **46–69** | CNN training → CNN evaluation (gets ~51%) → screwdriver-vs-hammer takeaway | **FOCUS — this is the headline** — ~15 min. |
| **70–90** | Why MLP fails; CNN's three special features (locality, stationarity, compositionality); convolution mechanics with Sobel filter | **Read carefully** — ~15 min. The "why" of everything. |
| **91–115** | RGB / multi-channel convolution, padding & stride formulas, pooling | **Reference** — ~10 min. Skim the prose, focus on the formula examples. |

**Total target read time for the notebook itself:** ~55 min. Add this brief's ~25 min and you're at **~80 min**, vs. the current **90–120+ min**.

---

## ✅ Walk-away checklist

After reading the notebook, you should be able to say in your own words:

- [ ] **Why an MLP fails on images** — flattening destroys spatial structure, and the parameter count explodes.
- [ ] **What a kernel does** — slides across the image, multiplies-and-sums at each position, produces a feature map.
- [ ] **What stride and padding control** — stride sets the step size; padding decides whether the output shrinks.
- [ ] **Why pooling has zero learnable parameters** — it's a fixed rule (max or mean), not something the network learns.
- [ ] **The three properties of images CNNs exploit** — locality, stationarity, compositionality. And which CNN feature encodes each (sparse connectivity / parameter sharing / stacked layers + pooling).
- [ ] **How to count parameters in a conv layer** — `((m × n × d) + 1) × k`. And why this is **independent of input image size** (unlike a Dense layer).

If any of these feel shaky after the notebook, come back to the relevant primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Your input is a `28×28` grayscale image. You apply a `Conv2D(filters=8, kernel_size=3, padding="valid", stride=1)`. What is the output shape?
2. Same input, but with `padding="same"`. What is the output shape now?
3. A Conv2D layer with **32 filters**, **5×5 kernel**, on an **RGB input**. How many learnable parameters does it have?
4. Why does max pooling have **zero learnable parameters**?
5. In one sentence: why does the CNN beat the MLP in this notebook even though, in raw terms, an MLP has more parameters?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **Output: 26×26×8.** Using `o = (n + 2p − f)/s + 1 = (28 + 0 − 3)/1 + 1 = 26`. Depth = number of filters = 8.
2. **Output: 28×28×8.** `padding="same"` means enough padding is added so the output side equals the input side. Depth still = 8.
3. **2,432 parameters.** `((5 × 5 × 3) + 1) × 32 = (75 + 1) × 32 = 76 × 32 = 2,432`. The "× 3" comes from RGB input depth.
4. **Max pooling is a fixed rule** — pick the maximum value in each window. There are no weights to learn; the operation is the same on every input. (This is different from Conv2D, where the kernel values are learned.)
5. **CNN wins because it bakes in the three properties of images** (locality, stationarity, compositionality) via sparse connectivity, parameter sharing, and stacked layers + pooling. MLPs spend their parameters on relationships that don't exist (every pixel ↔ every neuron). CNNs spend their parameters on relationships that **do** exist (small local patches, repeated everywhere).

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./CV_CNN_Jargon_Card.md)
