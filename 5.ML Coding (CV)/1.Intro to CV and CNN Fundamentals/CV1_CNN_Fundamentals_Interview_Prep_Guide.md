<a id="top"></a>
# CV Notebook 1 — Intro to CV & CNN Fundamentals (Deep Dive)

> Per-notebook companion to the master guide. For the cross-notebook synthesis, cheat sheet, glossary and drill, see [`../CV_Revision_Guide.md` §1](../CV_Revision_Guide.md#1-module1). **This deep-dive is standalone** — every concept the notebook touches has its own Concept Definition Template entry below; you should never have to leave this file to understand a term.

## What this notebook actually demonstrates

A side-by-side proof that **a tiny CNN beats a massive MLP** on image classification:
- **Dataset:** Clothing-Small (10 classes — t-shirt, shirt, jacket, dress, skirt, shorts, pants, longsleeve, shoes, hat). 3,068 train / 341 val / 372 test images. Imbalanced (t-shirt has 795 images, hat has 123).
- **Baseline MLP:** `Flatten → Dense(1024) → Dense(256) → Dense(10)` → **~50.6M parameters, ~36% test accuracy**.
- **Tiny CNN:** `Conv2D(16, 3) → MaxPool → Flatten → Dense(256) → Dense(10)` → **~16.8M parameters, ~50% test accuracy** (+40% relative improvement with 67% fewer parameters).

The whole notebook is the canonical "specialization beats brute force" lesson — a model whose architecture matches the data's structure (CNN for images) crushes a brute-force fully-connected one.

## 🪜 Mental anchors for this notebook

- **Tea-room analogy** — A Python list scatters ingredients across the room (pointer-chase); a NumPy/Keras ndarray image lines them up on one shelf (contiguous memory). *Why it matters here:* every image in this notebook is an `(H, W, C)` ndarray — same trick that made NumPy 100× faster than lists makes tensor ops fast on GPU.
- **CNN inductive biases** — locality + stationarity + compositionality. Three free assumptions about images that CNNs bake into their architecture; MLPs lack them. *Why it matters here:* this is the headline lesson — a 16.8M-param CNN beats a 50.6M-param MLP because it has these biases for free.
- **Sliding stamp** — a convolution is a small grid of weights "stamped" onto every patch of the image; at each position you multiply-and-sum to get one output pixel. *Why it matters here:* every Conv2D cell in the notebook is doing exactly this.
- **Output-shape formula** — `O = (N + 2P − F) / S + 1`. *Why it matters here:* the quiz cells ask you to compute it on the spot, and you need it to predict `(128, 128, 3) → (64, 64, 16)` after one Conv+Pool.

[🔝 Back to top](#top)

<a id="walkthrough"></a>
## 📖 Concept walkthroughs

> Every concept this notebook touches gets a full Concept Definition Template entry. No link-only blocks. A beginner should be able to read top-to-bottom and walk away knowing *what / why / how / where / related* for every term, with the notebook's specific shapes, numbers, and code substituted in.

### Image as tensor

> **🪜 Mental model:** *A photo is a stack of numbers.* Height × Width × Channels — each cell is one pixel's intensity in one colour.

**What it is.** Inside a deep-learning framework, an image is a **3-dimensional array** (called a **tensor** when it can live on a GPU). For this notebook, every image is resized to shape `(128, 128, 3)` — **128 rows × 128 columns × 3 colour channels** (red, green, blue). Each entry is a number representing that pixel's intensity in that channel: an integer in `[0, 255]` before normalisation, a float in `[0, 1]` after. A *batch* of `B` images adds a leading axis, giving `(B, 128, 128, 3)` — Keras/TensorFlow's **channels-last** convention; PyTorch uses channels-first `(B, 3, 128, 128)`.

**Why it matters.** Every CV model expects a tensor of a specific shape and value range. Mismatches here are the #1 silent bug — the model trains but produces nonsense because pixels are 0–255 when the model expected 0–1, or because the layout is channels-first when the model expected channels-last. Half of CV interviews open with "what's the shape of an RGB image batch?" — get the axis order wrong and the interviewer assumes you've never trained a model end-to-end.

**How it works.** When `tf.keras.utils.image_dataset_from_directory` loads a JPEG, it decompresses the file into a NumPy array, then converts it to a `tf.Tensor`. The notebook's `layers.Resizing(128, 128)` enforces a uniform spatial size (raw images can have any resolution). After preprocessing, a batch from `train_ds` has shape `(32, 128, 128, 3)` because the default batch size is 32 and each image is 128×128 RGB.

**Where it's used.**
- **Cell 2** of this notebook: `image_dataset_from_directory(...)` produces `(image_batch, label_batch)` where `image_batch.shape == (32, 256, 256, 3)` by default.
- **Cell 3**: `layers.Resizing(128, 128)` and `layers.Rescaling(1./255)` reshape the tensor and rescale values to `[0, 1]`.
- **Cells 4–5**: both MLP and CNN consume tensors of shape `(B, 128, 128, 3)`.

**Related terms.**
- **Tensor** — multidimensional array that can live on a GPU.
- **Channel** — one colour plane (R, G, or B for RGB images).
- **Batch** — multiple images stacked along a new leading axis for parallel processing.
- **Channels-last (HWC) vs channels-first (CHW)** — Keras default vs PyTorch default; same data, transposed layout.
- **`uint8` vs `float32`** — the raw pixel `dtype` (0–255 integers) vs the model-ready `dtype` (0–1 floats).

```python
# A single batch in this notebook after preprocessing
# train_ds yields tuples of shape (32, 128, 128, 3) and (32,)
for x, y in train_ds.take(1):
    print(x.shape, x.dtype, x.numpy().min(), x.numpy().max())
# → (32, 128, 128, 3) float32 0.0 1.0
```

**Gotcha.** OpenCV reads images as **BGR**, not RGB. If you mix `cv2.imread` with a Keras pipeline that expects RGB, your network produces subtly wrong predictions for months. (Keras's `image_dataset_from_directory` reads RGB directly, so this notebook is safe — but it's the most common silent CV bug in the wild.)

### Pixel range — `[0, 255]` → `[0, 1]` (Rescaling)

> **🪜 Mental model:** *Squeeze big numbers into small ones the network likes.* Raw pixels (0–255) are loud; gradient descent prefers a polite range (0–1 or −1–1).

**What it is.** Raw image files store each pixel as an 8-bit unsigned integer in `[0, 255]`. Neural networks train more stably when inputs sit in a small, centred range — typically `[0, 1]` or `[-1, 1]`. **Rescaling** is the one-line conversion that maps the integer range to the model's expected float range. In this notebook: `layers.Rescaling(1./255)` multiplies every pixel by `1/255`, so 0 stays 0 and 255 becomes 1.0. Some pretrained models (Module 3) use model-specific rescaling instead (e.g., subtract ImageNet mean), but for from-scratch training, divide-by-255 is the default.

**Why it matters.** Large input magnitudes interact badly with weight initialisation and learning rates. If pixels are 0–255, the first layer's pre-activations are also 0–255-scale, which means: (1) the activation function saturates immediately, (2) gradients explode or vanish, (3) the learning rate that worked for `[0, 1]` inputs is now 255× too aggressive. Rescaling is the cheapest possible fix — one multiplication, zero parameters.

**How it works.**
1. `layers.Rescaling(scale, offset)` is a no-parameter Keras layer that computes `output = input * scale + offset`.
2. With `scale=1./255`, the operation maps `[0, 255] → [0, 1]`. With `scale=1./127.5, offset=-1`, it maps to `[-1, 1]`.
3. Because it's a layer (not a preprocessing function), it lives *inside* the model graph. The saved model can therefore take raw 0–255 inputs at deployment — the caller doesn't need to pre-divide.

**Where it's used.**
- **Cell 3** of this notebook: `preprocess = keras.Sequential([Resizing(128, 128), Rescaling(1./255)])` — chained with resizing so a single `train_ds.map(preprocess)` does both.
- In transfer learning (Module 3), each pretrained backbone has its own `preprocess_input` function that does scale+mean-subtraction; you'd skip `Rescaling(1./255)` in that case.

**Related terms.**
- **Normalisation** — broader term covering rescaling, mean-subtraction, and z-score scaling.
- **Standardisation** (z-score) — `(x − μ) / σ` per channel; used in PyTorch's `transforms.Normalize`.
- **`preprocess_input`** — model-specific rescaling for pretrained backbones (covered in Module 3).
- **Quiz Q1** — the notebook's first multiple-choice cell asks "which layer rescales to `[0, 1]`?" → answer is `Rescaling`.

```python
preprocess = keras.Sequential([
    layers.Resizing(128, 128),       # uniform spatial size
    layers.Rescaling(1./255),        # 0–255 → 0–1
])
train_ds = train_data.map(lambda x, y: (preprocess(x), y))
```

**Gotcha.** Place `Rescaling` **inside the model** (as a Keras layer), not outside as a NumPy operation. Otherwise the deployed model expects pre-divided inputs and your serving code silently breaks if the caller forgets the division.

### Why MLP fails on images (the notebook's headline lesson)

> **🪜 Mental model:** *Show a friend a photo cut into 49,152 random scraps.* That's what `Flatten` does — every spatial relationship is destroyed before the network can use it.

**What it is.** A **Multi-Layer Perceptron (MLP)** is the classical fully-connected neural network: every input feature wires to every neuron in the next layer, no spatial structure assumed. To apply an MLP to an image you must first **flatten** the `(128, 128, 3)` tensor into a 1-D vector of `128 × 128 × 3 = 49,152` numbers — losing all information about which pixels were neighbours. The notebook's baseline MLP does exactly this: `Flatten → Dense(1024) → Dense(256) → Dense(10)`.

**Why it matters.** This is the notebook's *thesis*. The MLP racks up **~50.6M parameters** (because `49,152 × 1,024 ≈ 50.3M` weights in the first Dense layer alone), trains for 10 epochs, and lands at **~36% test accuracy**. The tiny CNN — one Conv2D + Pool + Dense — uses **~16.8M parameters (67% fewer)** and reaches **~50% test accuracy**. The lesson "*specialisation beats brute force*" is exactly the inductive-bias argument: a model whose architecture matches the data's structure wins, no matter how big the alternative is.

**How it works (three reasons MLPs lose on images).**
1. **No spatial structure.** After `Flatten`, pixel `(0, 0)` and pixel `(127, 127)` look equally adjacent to the MLP — both are just two of the 49,152 input numbers. The MLP would have to *learn* that pixel 0 and pixel 1 are neighbours from data alone. A CNN, by contrast, only ever looks at small `k × k` neighbourhoods.
2. **Parameter explosion.** Every input pixel connects to every neuron in the first hidden layer. With 49,152 inputs and 1,024 hidden units, you have ~50M weights *before* the model has learned anything useful — and they're all *position-specific* (one weight per pixel location).
3. **No translation invariance.** Shift a shirt 5 pixels to the right and every MLP weight needs to relearn it — it's an entirely new vector to the network. A CNN reuses the same filter at every position (weight sharing), so a shift is "the same image" to it.

**Where it's used.** Nowhere — for images. MLPs are still fine for tabular data (rows of features with no spatial layout). For images, MLPs are a teaching foil: the easiest baseline to *beat*, used precisely to motivate CNNs. This notebook's whole structure (train MLP → fail → introduce CNN → win) is the canonical CV teaching arc. In FAANG interviews, "why is a CNN better than an MLP for images" is the most common opening question — and the right answer names the three biases below.

**Related terms.**
- **CNN (Convolutional Neural Network)** — the architecture that solves all three MLP problems (next entries).
- **Inductive bias** — assumptions baked into an architecture; CNNs have three (locality, stationarity, compositionality) that MLPs lack.
- **Weight sharing** — using the same filter at every spatial position; the trick that gives translation invariance.
- **Dense layer** — the building block of an MLP (also called fully-connected or FC layer).
- **Flatten** — the operation that destroys spatial structure to feed an MLP.

```python
mlp = keras.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),    # 49,152 inputs
    layers.Dense(1024, activation='relu'),        # 50.3M params here alone
    layers.Dense(256,  activation='relu'),
    layers.Dense(10,   activation='softmax'),
])
# Total: ~50.6M params, ~36% test accuracy
```

**Gotcha.** A "bigger MLP" is not the fix. Doubling the hidden layer to 2,048 just doubles the parameters without addressing the three structural problems. The fix is to *change the architecture* to one that respects images' structure — i.e., use a CNN.

### Convolution (the operation)

> **🪜 Mental model:** *Sliding stamp.* A small grid of weights is pressed onto every patch of the image; at each position you multiply-and-sum to produce one output number.

**What it is.** A **convolution** in deep learning is a *sliding dot product*: take a small `kH × kW` patch of the input, multiply element-wise by a learnable **kernel** (same size), sum all the products into a single number — this is one output pixel. Slide the kernel across the whole image, computing one number per position, to produce an output 2-D grid called a **feature map**. For multi-channel inputs (RGB), the kernel has matching `Cin` channels, and you sum the products across all channels too.

**Why it matters.** Convolution is the *single most important* operation in deep learning for vision. It encodes three biases that match how images actually work: (a) **locality** (you only look at neighbours), (b) **stationarity** (same kernel used everywhere, so a pattern in the corner is detected the same way as one in the centre), (c) **compositionality** (stacking conv layers builds edges → textures → object parts → objects). All three are free assumptions that an MLP has to learn from data — which it can't, on 3,068 training images.

**How it works.**
1. Place the `kH × kW × Cin` kernel over the top-left patch of the input.
2. Compute the **dot product**: `Σ kernel[i, j, c] * input[i, j, c]`. For a 3×3 RGB kernel that's `3 × 3 × 3 = 27` multiplications and one sum.
3. Add a learnable **bias** scalar.
4. Slide the kernel by `stride` pixels and repeat.
5. The collection of outputs forms one feature map. If you have `Cout` kernels, you get `Cout` feature maps stacked along the channel axis — a `(H_out, W_out, Cout)` tensor.

**Where it's used.**
- **Cell 5** of this notebook: `layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(128, 128, 3))` — 16 filters of size 3×3 sliding over the input.
- Every CNN architecture (LeNet, VGG, ResNet, EfficientNet) is mostly stacks of Conv2D.
- In interviews, "explain a convolution in 30 seconds" is a near-guaranteed opener — say "sliding dot-product over a small window, with weight sharing across positions."

**Related terms.**
- **Kernel / filter** — the learnable weight grid (next entry).
- **Feature map** — the 2-D output of one filter.
- **Stride** — step size when sliding.
- **Padding** — zeros added at the edges so the kernel can sit there too.
- **Weight sharing** — the property that the same filter is applied at every position.
- **Cross-correlation** — what conv layers *actually* compute (math purists distinguish this from "true convolution" which flips the kernel; deep-learning ignores the distinction).

```python
# The notebook's single conv layer
layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
              activation='relu', input_shape=(128, 128, 3))
# Input (128, 128, 3) → Output (128, 128, 16) — 16 feature maps.
```

**Gotcha.** A 3×3 conv on a 3-channel input has `3 · 3 · 3 = 27` weights *per filter*, **not 9**. Forgetting the input-channel dimension is the classic "I thought Conv was 2D" mistake. With 16 filters and one bias each, the layer's total parameter count is `(3 · 3 · 3 + 1) · 16 = 448`.

### Kernel / filter

> **🪜 Mental model:** *A tiny pattern detector.* Each filter learns to fire when its specific pattern (an edge, a curve, a stripe, a colour blob) appears in a patch.

**What it is.** A **kernel** (also called a **filter**) is the small array of learnable weights used in one convolution. Its shape is `(kH, kW, Cin)` — typically `3 × 3` or `5 × 5` spatially, with `Cin` matching the input's number of channels. A single Conv2D layer has `Cout` filters; each filter scans the whole input and produces one feature map. In this notebook, `Conv2D(16, 3)` means **16 filters, each of shape `(3, 3, 3)`** — `3 × 3` spatial, 3 channels deep because the input is RGB.

**Why it matters.** Filters are *what the network learns*. After training, low-level filters often look like edge or colour detectors; deeper-layer filters fire on textures, then object parts, then whole objects. Understanding "filter = learned pattern detector" is the intuition that makes CNNs make sense. In transfer learning (Module 3), the filters of a pretrained network are the literal thing you reuse — you're inheriting a library of pattern detectors that took millions of GPU-hours to learn.

**How it works.** Weights start random (typically He or Glorot initialisation). Gradient descent nudges them to minimise the loss. Over training, each filter ends up tuned to a specific pattern that the data wanted detected. You can visualise filters from a trained network — early VGG layers reveal Gabor-like edge filters arising spontaneously, without anyone hand-designing them.

**Where it's used.**
- **Specified by `kernel_size`** in `Conv2D`. Almost always odd (3, 5, 7) so the kernel has a single integer centre pixel.
- **Cell 5** of this notebook: `Conv2D(16, 3, ...)` configures 16 trainable `3 × 3 × 3` kernels.
- The pretrained filters of ImageNet models (Module 3) are what you reuse in transfer learning.

**Related terms.**
- **Convolution** — the operation that uses the filter.
- **Feature map** — what a single filter outputs.
- **Weight sharing** — the same filter is applied at every position (this is *why* a CNN has fewer params than an MLP).
- **`Cout`** — the number of filters in a Conv2D layer; controls how many parallel pattern detectors the layer learns.

```python
# A single Conv2D layer's filters
conv = layers.Conv2D(filters=16, kernel_size=3, padding='same')
# conv.kernel.shape after build → (3, 3, 3, 16): kH × kW × Cin × Cout
```

**Gotcha.** Even-sized kernels (2×2, 4×4) have no integer centre pixel and produce subtle alignment artefacts when stacked. Always use odd kernels in practice (3, 5, 7).

### Feature map

> **🪜 Mental model:** *A heat-map of where one pattern appeared.* Bright spots mean "this filter's pattern matched here strongly"; dark spots mean "this filter saw nothing of interest here."

**What it is.** A **feature map** (or *activation map*) is the 2-D output of one filter sliding over an input. Its dimensions `(H_out, W_out)` are determined by the input size, kernel size, stride, and padding (the output-shape formula below). One Conv2D layer with `Cout` filters produces `Cout` stacked feature maps, giving an output tensor of shape `(H_out, W_out, Cout)`. In this notebook, after `Conv2D(16, 3, padding='same')` on the `(128, 128, 3)` input, the output is `(128, 128, 16)` — 16 feature maps, each 128×128.

**Why it matters.** Feature maps are how information flows through a CNN — every intermediate "image" the network sees is a stack of feature maps. Their spatial dimensions shrink as you go deeper (because of pooling/stride), but the channel count grows (more filters detecting more patterns). The final feature maps of a pretrained backbone are the **embeddings** used in transfer learning (Module 3) and similarity search (Module 4).

**How it works.** Given input `(H, W, Cin)` and a `Conv2D(Cout, kH, kW)` layer with stride `S` and padding `P`, output shape is `(H_out, W_out, Cout)` where `H_out = (H + 2P − kH) / S + 1`. Each value in a feature map is a single filter's response (post-activation, typically ReLU). The 16 feature maps in this notebook are 16 different "views" of the same image — one heat-map per learned pattern.

**Where it's used.**
- **Cell 5** of this notebook produces a `(128, 128, 16)` feature map after Conv2D.
- After `MaxPooling2D()`, the feature map becomes `(64, 64, 16)`.
- **`model.summary()`** in the notebook shows the feature-map shape at every layer — your sanity-check tool.

**Related terms.**
- **Activation map** — synonym for feature map.
- **Channel** — one feature map = one channel of the output tensor.
- **Receptive field** — the input region that influences a single output activation; expands as you stack layers.
- **Embedding** — what you get when you reduce a feature map to a single vector (e.g., via Global Average Pooling).

**Gotcha.** Don't confuse "number of feature maps" (= output channels = `Cout`) with the spatial size of each map (= `H_out × W_out`). A `(64, 64, 16)` tensor has 16 feature maps, *each of which* is 64×64 = 4,096 pixels.

### Stride

> **🪜 Mental model:** *How big a step the stamp takes.* Stride 1 = touch every pixel; stride 2 = skip every other one and halve the output.

**What it is.** The **stride** is the step size (in pixels) the kernel moves between consecutive positions when sliding over the input. Default stride is 1. Stride 2 means the kernel jumps two pixels at a time, producing an output roughly half the input's spatial size. You can have different strides per axis (`strides=(2, 2)`), but they're almost always equal.

**Why it matters.** Stride is the cheap way to **downsample** inside a Conv2D — fewer output positions = fewer ops in the next layer. In this notebook, the Conv layer uses the default `strides=1` (so the conv itself doesn't downsample), and the `MaxPooling2D()` immediately after does the downsampling with its own default stride of 2. Modern architectures (ResNet, EfficientNet) often replace MaxPool with strided convolutions because strided convs are *learnable* downsamplers, while pooling is fixed.

**How it works.** With stride `S`, the kernel's anchor moves by `S` pixels per step. Output spatial size is `(H + 2P − kH) / S + 1` (integer division — fractions are floored). For this notebook's Conv: `(128 + 2·1 − 3) / 1 + 1 = 128`. For the subsequent MaxPool: `(128 + 0 − 2) / 2 + 1 = 64`.

**Where it's used.**
- **Cell 5** of this notebook: `Conv2D(..., strides=1)` (implicit default) and `MaxPooling2D()` with implicit `strides=2`.
- `Conv2D(strides=2)` in any downsampling block — common in ResNet stems.
- Detection backbones (Module 6) use strided convs to build the multi-scale feature pyramid.

**Related terms.**
- **Pooling** — the alternative way to downsample, with no learnable parameters.
- **Dilation (atrous)** — sibling concept that *spaces out* the kernel taps instead of stepping; expands receptive field without downsampling.
- **`'same'` padding interaction** — with `'same'` padding *and* stride `S > 1`, output size becomes `⌈N / S⌉`, not `N`.

**Gotcha.** A non-integer `(N + 2P − F) / S + 1` is silently floored — your output shape may be smaller than you expected. Always sanity-check with `model.summary()`.

### Padding (`'valid'` vs `'same'`)

> **🪜 Mental model:** *A photo frame of zeros.* Padding adds zero pixels around the image so the kernel can sit at the edges and edge pixels participate in the convolution.

**What it is.** **Padding** is the practice of adding extra (usually zero) pixels around the input before convolving. Two common Keras modes:
- **`'valid'`** — no padding (`P = 0`). Output shrinks by `(kH − 1)` per layer when stride is 1.
- **`'same'`** — Keras computes `P` so the output has the same spatial size as the input *when stride is 1*. For a 3×3 kernel that means `P = 1` on each side.

PyTorch takes a numeric `padding=P` directly (number of zeros per side).

**Why it matters.** Without padding, every conv layer shrinks the image — stack 10 layers of `'valid'` 3×3 convs and you've lost 20 pixels of edge information. `'same'` padding lets you stack as many layers as you want while keeping the spatial size constant, and lets edge pixels participate in convolutions instead of being ignored. The notebook uses `padding='same'` for its single Conv2D so the output stays `(128, 128, 16)`.

**How it works.** For kernel size `kH` and `'same'` padding with stride 1, Keras adds `P = (kH − 1) / 2` zeros on each side. The convolution then sees an effectively `(N + 2P) × (M + 2M)` input, and the formula `O = (N + 2P − kH) / S + 1` gives `O = N` exactly. For `stride > 1`, Keras may pad asymmetrically and the output becomes `⌈N / S⌉`.

**Where it's used.**
- **Cell 5** of this notebook: `Conv2D(16, 3, padding='same', ...)` → output stays 128×128.
- `'same'` is the default in modern CNNs (U-Net, ResNet variants).
- `'valid'` shows up in older nets (LeNet) and in classifier heads where shrinking is desired.
- The notebook's **Quiz Q3** asks which statements about padding are true: `'same'` preserves dimensions ✓, `'valid'` drops non-fitting regions ✓.

**Related terms.**
- **Stride** — interacts with padding to determine output size.
- **Output-shape formula** — see next entry.
- **Zero-padding vs reflection padding** — the latter mirrors edge pixels instead of using zeros; sometimes better for image generation.

```python
# Notebook: Conv2D(16, 3, padding='same') on (128, 128, 3) input
# N=128, F=3, P=1 (added by 'same'), S=1
# O = (128 + 2·1 − 3) / 1 + 1 = 128  ✓ — output stays 128×128
```

**Gotcha.** `'same'` does **not** guarantee `O = N` when `S > 1` — it preserves the size only at stride 1. With stride 2 and `'same'`, the output becomes `⌈N / 2⌉`. Always run the formula explicitly when strides are non-trivial.

### Output-shape formula `O = (N + 2P − F) / S + 1` (memorise this)

> **🪜 Mental model:** *Tape-measure rule for convolutions.* Given the input width, kernel size, padding, and stride, count how many kernel positions fit along one dimension.

**What it is.** A purely arithmetic formula that tells you the spatial size of a Conv2D's or MaxPool2D's output:
- `O` = output height (or width).
- `N` = input height (or width).
- `F` = filter / pool size (`kernel_size` or `pool_size`).
- `P` = padding (per side, in pixels — 0 for `'valid'`, `(F−1)/2` for `'same'` at stride 1).
- `S` = stride (step size).

Read it in words: *"(input length + extra padding on both sides − one kernel) divided by step, then add one for the starting position."* Integer division — fractions are floored.

**Why it matters.** Output shape governs everything downstream: memory budget, parameter count of the next layer, whether your Flatten + Dense will explode. The notebook's **Quiz Q4** (`12 × 12` input, `3 × 3` filter, stride 1, no padding → `10 × 10`) is the canonical interview drill — get the answer wrong in a coding round and the interviewer will assume you've never trained a CNN end-to-end.

**How it works (symbol-by-symbol translation).**
1. Start with the input length `N` along one spatial axis.
2. Add `2P` because padding adds `P` zeros on the left and `P` on the right — that's `2P` extra positions the kernel can stand on.
3. Subtract `F` because the kernel takes up `F` positions and can't extend past the end.
4. Divide by `S` to count how many strides fit in the remaining space.
5. Add `1` because the kernel's starting position itself contributes one output, even before any stride.
6. Floor the result (integer division) — if the kernel can't quite fit at the end, those leftover pixels are silently dropped.

**Where it's used.**
- **Every `Conv2D` and `MaxPool2D` call** — the formula governs both.
- `model.summary()` in Keras shows the output shape per layer — verify against your manual calculation.
- **Quiz Q4** of this notebook drills exactly this formula.
- FAANG whiteboard rounds: "draw an architecture, what's the shape after this layer?" — the formula is the answer.

**Related terms.**
- **`'same'` padding** — the Keras shortcut that picks `P = (F−1)/2` to make `O = N` *at stride 1*.
- **`'valid'` padding** — `P = 0`; output shrinks every layer.
- **Receptive field** — the input region influencing one output pixel; *related but distinct* from output shape.
- **`tf.keras.Model.summary()`** — the sanity-check tool that prints `Output Shape` for every layer.

```python
# Notebook example: input (128, 128, 3), Conv2D(16, 3, padding='same')
# N=128, F=3, P=1 (for 'same'), S=1 → O = (128 + 2 − 3) / 1 + 1 = 128. ✓
# Then MaxPooling2D() defaults: F=2, P=0, S=2 → O = (128 + 0 − 2) / 2 + 1 = 64. ✓
```

**Gotcha.** `'same'` padding **does not** guarantee `O = N` when `S > 1` — it preserves the size only at stride 1. Run the formula explicitly when strides are non-trivial.

### CNN inductive biases (the *real* reason CNNs win)

> **🪜 Mental model:** *Three free assumptions about how images work.* Locality, stationarity, compositionality — bake them into the architecture and you stop having to learn them from data.

**What it is.** An **inductive bias** is a built-in assumption an architecture makes about its input — assumptions that the data *should* obey. CNNs encode three biases that match natural images:
- **Locality** — pixels close to each other are correlated; far-away pixels are mostly independent. So convolutions only mix pixels within a small `k × k` window.
- **Stationarity (translation invariance)** — the same visual pattern can appear anywhere; the same filter should detect it wherever it appears. So the filter's weights are **shared** across all positions (weight sharing).
- **Compositionality** — complex patterns are built from simple ones (edges → textures → object parts → objects). So we **stack** conv layers, each building on the previous one's outputs.

**Why it matters.** These three biases are *why* a 16.8M-param CNN beats a 50.6M-param MLP on this notebook's Clothing-Small dataset. The MLP has to learn from data that pixel (12, 34) is related to pixel (13, 34) — but with only 3,068 training images, it never gets enough evidence. The CNN doesn't need to *learn* that — locality is hard-coded into its 3×3 kernel size. Same for stationarity (weight sharing) and compositionality (stacked layers). The fewer assumptions a model has to learn from data, the less data it needs, and the better it generalises.

**How it works (in this notebook's CNN).**
1. **Locality** is set by the kernel size: `Conv2D(16, 3, ...)` only ever looks at a `3 × 3 × 3` neighbourhood (27 pixels including channels) to produce one output value. The MLP looks at all 49,152 at once — no locality.
2. **Stationarity** comes for free in convolution — the same 16 filters are slid across every position. The MLP has separate weights for every `(row, col)`, which means it must relearn every shifted variant of a shape.
3. **Compositionality** comes from stacking: a single Conv layer captures low-level features (edges); a stack captures parts, then objects. The notebook only has one Conv + one Pool (not much compositionality), but even that beats the MLP — and Module 2 will stack five conv blocks.

**Where it's used.** Every CNN architecture (LeNet, VGG, ResNet, EfficientNet) makes the same three assumptions. Vision Transformers (ViTs) *relax* compositionality and locality, which is why they need 100× more data to match a CNN. In interviews, "why do CNNs beat MLPs on images" is asking *exactly* this question — and the right answer names all three biases.

**Related terms.**
- **Weight sharing** — the mechanism that gives translation invariance / stationarity.
- **Translation equivariance** — what convolution gives you; output shifts when input shifts. Combined with pooling, this becomes (approximate) translation **invariance**.
- **Receptive field** — what locality buys you locally; expands as you stack layers (compositionality).
- **Vision Transformer (ViT)** — a competing architecture with weaker inductive biases; needs much more data.

**Gotcha.** "CNN = good for images" is the lazy answer. The interviewer wants the three biases by name — locality, stationarity, compositionality — and a sentence each on *why* they help. Memorise this trio.

### Pooling (max vs average)

> **🪜 Mental model:** *Shrink-and-keep-the-loudest.* Take a `2 × 2` window, output one number (the max or the average), then move on. Halves the spatial size each time.

**What it is.** **Pooling** is a fixed, non-learnable downsampling operation. The two common types:
- **MaxPooling2D** — outputs the *largest* value in each pooling window.
- **AveragePooling2D** — outputs the *mean* of each window.

This notebook uses `layers.MaxPooling2D()` with all defaults: `pool_size=2, strides=2, padding='valid'`. That means a 2×2 window slides with stride 2 (non-overlapping), halving both spatial dimensions. Pooling has **zero learnable parameters** — it's deterministic given the input.

**Why it matters.** Pooling does three useful things at once: (1) reduces spatial size so deeper layers run faster and use less memory, (2) introduces small **translation invariance** (a tiny shift of the input → roughly the same pooled output, because the max is unchanged), (3) acts as a coarse regulariser by discarding fine-grained position information the network doesn't need. In this notebook, the single MaxPool turns `(128, 128, 16) → (64, 64, 16)` — a 4× reduction in spatial volume.

**How it works.** Slide a non-overlapping `2×2` window across each feature map. MaxPool keeps `max(window)`; AvgPool keeps `mean(window)`. There's no backprop through the unselected positions in MaxPool — the gradient flows only through the position that was the max. AvgPool spreads gradient equally across the four positions.

**Where it's used.**
- **Cell 5** of this notebook: `layers.MaxPooling2D()` immediately after the Conv2D layer.
- Classical CNNs (VGG, LeNet) pool after every conv block. Module 2 uses one MaxPool per conv block, five times.
- Modern ResNets often replace mid-network pooling with strided convs and use **Global Average Pooling** (Module 2) at the end.
- **Quiz Q6** of this notebook asks "what's the role of pooling and what does it sacrifice?" — exactly this entry.

**Related terms.**
- **Strided convolution** — learnable alternative to pooling.
- **Global Average Pooling (GAP)** — pools the *whole* feature map to one number per channel; covered in Module 2.
- **MaxUnpool / unpool** — the inverse, used in some decoders (U-Net).

```python
# Notebook usage — default pool_size=2, strides=2
layers.MaxPooling2D()
# (128, 128, 16) → (64, 64, 16) — halves spatial dims, channels unchanged
```

**Gotcha.** MaxPool destroys information — once a value is discarded it's gone. Don't pool too aggressively early in the network or you'll lose detail you need later. Also: don't use pooling on tasks that need precise localisation (segmentation, detection) — it throws away the exact pixel position.

### ReLU activation

> **🪜 Mental model:** *Half-wave rectifier.* Negative numbers → 0; positives pass through unchanged. The cheapest non-linearity that doesn't kill gradients.

**What it is.** **ReLU** ("Rectified Linear Unit") is the non-linear activation function `f(x) = max(0, x)` — applied element-wise to a Conv or Dense layer's output. ReLU is the default activation in modern CV — almost every Conv2D layer is followed by a ReLU, and in this notebook the `activation='relu'` argument bakes it into the layer itself (no separate `Activation` layer needed).

**Why it matters.** Without a non-linearity between layers, stacking Conv or Dense layers is mathematically equivalent to a single linear layer (no extra expressive power, no matter how many layers you stack). ReLU adds non-linearity *cheaply* (one comparison) and — crucially — has a *non-saturating* gradient for positive inputs (gradient = 1), which fixes the **vanishing gradient** problem that plagued earlier networks using sigmoid or tanh.

**How it works.**
1. **Forward pass:** output equals input if positive, else 0.
2. **Backward pass:** gradient equals 1 if the original input was positive, else 0.
3. **No parameters** — it's just a `max(0, x)` operation, no weights to learn.

**Where it's used.**
- **Cell 4 (MLP)** and **Cell 5 (CNN)** of this notebook: both Dense and Conv layers use `activation='relu'`.
- Every Conv2D in modern CNNs (LeNet, VGG, ResNet, EfficientNet) defaults to ReLU.
- Hidden layers in MLPs.
- Output layers don't use ReLU — they use softmax for classification, tanh/linear for regression.

**Related terms.**
- **Sigmoid** — the older non-linearity; saturates and vanishes the gradient. Avoid in hidden layers.
- **LeakyReLU** — `f(x) = x if x > 0 else 0.01·x`; fixes the **dead-neuron** problem (a ReLU whose input is permanently negative and so never updates).
- **GELU / Swish** — smoother modern alternatives, mainly used in transformers.
- **Vanishing gradient** — the problem ReLU was invented to solve.

```python
# Notebook usage — built into the layer
layers.Conv2D(16, 3, padding='same', activation='relu')
# Equivalent to:
# layers.Conv2D(16, 3, padding='same'); layers.ReLU()
```

**Gotcha.** **Dead ReLU:** if a neuron's weights drift to where its input is always negative, the gradient is permanently 0 and the neuron never learns again. Cures: He initialisation, reasonable LR, or use LeakyReLU. Not usually a problem in small models like this notebook's, but bites in deeper nets.

### Flatten

> **🪜 Mental model:** *Squash 2-D feature maps into a 1-D row.* Collapses spatial structure so a Dense layer can consume the result.

**What it is.** A **Flatten** layer reshapes a tensor `(B, H, W, C)` into `(B, H · W · C)` — keeping the batch dimension, collapsing everything else into a single vector per example. It has **no parameters**; it's just a memory rearrangement (or a view). In this notebook, after the MaxPool produces `(B, 64, 64, 16)`, Flatten turns it into `(B, 65,536)` — a vector of 65,536 numbers per image, ready to be fed to a Dense layer.

**Why it matters.** Conv layers preserve spatial structure; **Dense (fully-connected)** layers don't — they expect a flat vector. So whenever you transition from the convolutional part of a network to the classifier head, you need a Flatten (or its modern replacement, `GlobalAveragePooling2D` — see Module 2). The Flatten itself is free, but the *Dense layer after it* is where most of the CNN's parameters live: `Dense(256)` on top of a 65,536-vector adds `65,536 × 256 + 256 ≈ 16.78M` parameters — that's where the notebook's CNN gets its 16.8M total.

**How it works.** Just a reshape: `tf.reshape(x, [batch_size, -1])`. No math, no parameters, no learning. The order in which spatial dimensions are flattened is `(H, W, C)` row-major (Keras), so position `(i, j, c)` of the feature map maps to index `i * W * C + j * C + c` of the vector.

**Where it's used.**
- **Cell 5** of this notebook: after `MaxPooling2D`, `layers.Flatten()` turns `(64, 64, 16) → (65,536,)`.
- At the boundary between the conv backbone and the dense classifier head in any classical CNN.
- Increasingly replaced by `GlobalAveragePooling2D` (Module 2), which averages each feature map to one number — producing 16 numbers instead of 65,536, slashing the next Dense layer's parameter count by ~4,000×.

**Related terms.**
- **Dense / Fully-connected layer** — what consumes the flat vector.
- **GlobalAveragePooling2D** — modern alternative; covered in Module 2.
- **Reshape** — the general operation Flatten is a special case of.

```python
# Notebook: feature map shape transition
# After MaxPool: (B, 64, 64, 16)
layers.Flatten()
# After Flatten: (B, 65_536) — ready for Dense
```

**Gotcha.** After Flatten, the vector size depends on the feature map shape. For this notebook's `(64, 64, 16)` feature map → 65,536 numbers; a `Dense(256)` on top adds 16.78M parameters — most of the CNN's weights. Replacing Flatten with `GlobalAveragePooling2D` (which would output 16 numbers) drops that to just `16 × 256 + 256 ≈ 4,400` parameters. This is why Module 2 makes the swap.

### Dense (Fully-Connected) layer

> **🪜 Mental model:** *Every input talks to every output.* Each output unit is a learned weighted sum of *all* input units, plus a bias.

**What it is.** A **Dense** layer (also called "fully-connected", "FC", or "linear") maps a vector of `Nin` numbers to a vector of `Nout` numbers via `y = W · x + b`, where `W` is a learnable `(Nout × Nin)` weight matrix and `b` is a length-`Nout` bias vector. Typically followed by a non-linearity (ReLU in hidden layers, softmax at the output of a classifier). This notebook uses `Dense(256, activation='relu')` as a hidden layer and `Dense(10, activation='softmax')` as the final classifier.

**Why it matters.** Dense layers can mix information from *anywhere* in the feature vector — useful for the final classification step where the network needs to combine all the high-level features into a per-class score. But they explode in parameter count on raw image inputs: one Dense layer on a flattened `128 × 128 × 3` image (49,152 inputs) with 1,024 units is `49,152 × 1,024 + 1,024 ≈ 50.3M` parameters. That's why MLPs are bad on raw images and why CNNs put Dense layers *only at the end*, after the conv backbone has reduced the dimensionality.

**How it works.** A single matrix multiply plus a bias-add plus an activation. The forward pass: `y = activation(W @ x + b)`. Backpropagation is straightforward: gradient w.r.t. weights is `δ · xᵀ`; gradient w.r.t. input is `Wᵀ · δ`. The parameter count is `Nin × Nout + Nout` (the trailing `+ Nout` is one bias per output unit).

**Where it's used.**
- **Cell 4 (MLP)** of this notebook: three Dense layers — `Dense(1024) → Dense(256) → Dense(10)`. All 50.6M params live in these three layers.
- **Cell 5 (CNN)** of this notebook: `Dense(256, relu) → Dense(10, softmax)` as the classifier head atop the Flatten.
- The final 1–2 layers of every classifier in every modern CNN.
- The entirety of an MLP (Multi-Layer Perceptron).

**Related terms.**
- **MLP** — a stack of only Dense layers; the original 1980s neural net.
- **Logit** — the raw pre-softmax output of the final Dense layer (10 numbers in this notebook).
- **Softmax** — turns logits into a probability distribution (next entry).
- **Linear layer** — PyTorch's name for the same thing (`nn.Linear`).

```python
# Notebook usage
layers.Dense(256, activation='relu')     # hidden layer
layers.Dense(10,  activation='softmax')  # final classifier — 10 clothing classes
```

**Gotcha.** Dense layers throw away spatial structure entirely — they're why MLPs are bad for raw images and CNNs are good. Put Dense layers *only after* a Conv backbone has condensed spatial information; never feed raw image pixels to a big Dense layer.

### Softmax activation

> **🪜 Mental model:** *Vote-share calculator.* Turn 10 raw scores into 10 probabilities that sum to 1 — the bigger the score, the bigger the share.

**What it is.** **Softmax** is the activation function applied to the last layer of a multi-class classifier. It takes a vector of `K` raw scores (called **logits**) and returns a vector of `K` probabilities that sum to 1, via `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`. In this notebook, `Dense(10, activation='softmax')` produces a 10-dimensional probability vector over the 10 clothing classes — e.g., `[0.02, 0.85, 0.01, ..., 0.03]` meaning the model is 85% sure the image is class 1 (shirt).

**Why it matters.** Softmax converts arbitrary real-valued logits into a valid probability distribution, which is what the cross-entropy loss expects. It also *exaggerates* the largest logit (because of the exponential), making the model's "best guess" stand out clearly. The argmax of softmax outputs is the predicted class label.

**How it works.**
1. **Exponentiate** every logit: `e^{z_i}`. Negative logits become small positive numbers; large positive logits become huge.
2. **Sum** all exponentials: `Z = Σ_j e^{z_j}`.
3. **Divide** each exponential by the sum: `p_i = e^{z_i} / Z`. Now all `p_i` are in `(0, 1)` and they sum to 1.
4. For inference, take `argmax(p)` to pick the predicted class.

**Where it's used.**
- **Cell 5** of this notebook: `Dense(10, activation='softmax')` as the final classifier.
- **Cell 6**: the predicted probabilities from `model.predict(test_ds)` come out as a `(N, 10)` matrix of softmax outputs; `tf.argmax(y_pred, axis=1)` picks the predicted class.
- Every multi-class classifier in deep learning.

**Related terms.**
- **Logit** — the raw pre-softmax score (one per class).
- **Sigmoid** — softmax's sibling for binary or multi-label classification (each output independent).
- **Cross-entropy loss** — pairs with softmax; measures how far the predicted distribution is from the true one-hot label.
- **`sparse_categorical_crossentropy`** — the loss this notebook uses; assumes integer labels and softmax outputs.

```python
# Final classifier head — 10 classes
layers.Dense(10, activation='softmax')
# Output shape per example: (10,) — probabilities summing to 1
```

**Gotcha.** If labels are one-hot encoded vectors (`[0, 1, 0, 0, ...]`) you must use `categorical_crossentropy`; if they're integer indices (`1`) you must use `sparse_categorical_crossentropy`. Mixing them up causes silent training failures or shape errors. This notebook uses integer labels (from `image_dataset_from_directory`), hence `sparse_categorical_crossentropy`.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Setup & imports
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np, matplotlib.pyplot as plt
```

### 2. Loading the Clothing-Small dataset
```python
train_data = tf.keras.utils.image_dataset_from_directory("path/train", shuffle=True)
val_data   = tf.keras.utils.image_dataset_from_directory("path/val",   shuffle=False)
test_data  = tf.keras.utils.image_dataset_from_directory("path/test",  shuffle=False)
```
Each dataset is a `tf.data.Dataset` yielding `(image_batch, label_batch)`. Default `batch_size=32`, default `image_size=(256, 256)`. The class index is determined by alphabetical folder-name order — verify with `train_data.class_names`.

### 3. Preprocessing — resize + rescale to `[0, 1]`
```python
preprocess = keras.Sequential([
    layers.Resizing(128, 128),     # uniform input size
    layers.Rescaling(1./255),      # map 0–255 → 0–1
])
train_ds = train_data.map(lambda x, y: (preprocess(x), y))
val_ds   = val_data  .map(lambda x, y: (preprocess(x), y))
test_ds  = test_data .map(lambda x, y: (preprocess(x), y))
```
Both transformations are Keras layers, so once compiled into a saved model they apply automatically — no out-of-band preprocessing required at deployment.

### 4. Baseline MLP (the bad approach — show why CNNs win)
```python
mlp = keras.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),    # 49,152 inputs!
    layers.Dense(1024, activation='relu'),
    layers.Dense(256,  activation='relu'),
    layers.Dense(10,   activation='softmax'),
])
mlp.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
mlp.fit(train_ds, validation_data=val_ds, epochs=10)
# ~50.6M params, ~36% test accuracy
```

**Why this is bad:**
1. Flattening destroys spatial structure.
2. Every input → every neuron in the next layer = parameter explosion.
3. No translation invariance — a shifted shirt is "a new image" to an MLP.

### 5. Simple CNN (the good approach)
```python
cnn = keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu',
                  input_shape=(128, 128, 3)),     # → (128, 128, 16)
    layers.MaxPooling2D(),                         # → (64, 64, 16)
    layers.Flatten(),                              # → (65_536,)
    layers.Dense(256, activation='relu'),
    layers.Dense(10,  activation='softmax'),
])
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(train_ds, validation_data=val_ds, epochs=10)
# ~16.8M params, ~50% test accuracy
```

### 6. Evaluation & confusion matrix
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = cnn.predict(test_ds)
preds = tf.argmax(y_pred, axis=1).numpy()
y_true = np.concatenate([y.numpy() for _, y in test_ds])
print(accuracy_score(y_true, preds))
print(classification_report(y_true, preds))
```

**Plain English:** `model.predict` runs the trained CNN on every test batch and returns a `(N, 10)` matrix of class probabilities. `argmax(axis=1)` collapses each row to the index of its highest-probability class. We then stitch the true labels out of the dataset (one batch at a time) and compare — `classification_report` is the per-class precision/recall/F1 table you need because the classes are imbalanced.

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `tf.keras.utils.image_dataset_from_directory` | Folder structure must be `path/class_name/*.jpg` |
| `layers.Resizing(h, w)` | Resize inside the model graph |
| `layers.Rescaling(scale, offset)` | `1./255` is standard for `[0, 1]` |
| `layers.Conv2D(filters, kernel_size, strides, padding, activation)` | The CNN workhorse |
| `layers.MaxPooling2D(pool_size, strides)` | Defaults: `pool_size=2, strides=2` |
| `layers.Flatten()` | Required before `Dense` |
| `layers.Dense(units, activation)` | `'softmax'` on the output for classification |
| `model.compile(optimizer, loss, metrics)` | `'sparse_categorical_crossentropy'` for integer labels |
| `model.fit(train_ds, validation_data, epochs)` | Returns a `History` object |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **Imbalanced classes** — t-shirt has 795 train images, hat has 123. Don't trust raw accuracy alone; look at per-class precision/recall from `classification_report`.
2. **`image_dataset_from_directory` is alphabetical** — the class index is determined by folder-name sort order. Print `train_data.class_names` to verify.
3. **`sparse_categorical_crossentropy` requires integer labels.** If you one-hot encoded by accident, switch to `categorical_crossentropy`.
4. **`Dense` after `Flatten` has the bulk of the parameters** — `Flatten((64, 64, 16)) → Dense(256)` is `65,536 × 256 ≈ 17M`. This is why Module 2 swaps in `GlobalAveragePooling2D`.
5. **Resize *and* rescale, in that order, inside the model.** Putting these as `keras.layers` (rather than dataset-side preprocessing) lets the saved model accept raw 0–255 input — no out-of-band preprocessing step for the caller.
6. **Don't shuffle the val/test set.** `image_dataset_from_directory(..., shuffle=False)` for val/test is mandatory if you want labels and predictions to line up when building the confusion matrix.
7. **A `3 × 3` Conv on RGB has 27 weights per filter, not 9.** Forgetting `Cin` in the parameter count is the canonical interview slip.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

> ≥ 10 questions, ≥ 5 sourced and cited from canonical interview banks. Every concept in the 📖 walkthrough section above has at least one Q&A item below.

**Q1. Which layer rescales input values to `[0, 1]`?** *(notebook quiz cell, original)*
→ **`layers.Rescaling(1./255)`** — placed inside the model so the saved network accepts raw 0–255 input.

**Q2. Which is FALSE about MLPs for images?** *(notebook quiz cell, original)*
→ **"MLPs have fewer params than CNNs"** is false — it's the opposite. The notebook's MLP has ~50.6M params, the CNN has ~16.8M.

**Q3. True statements about padding?** *(notebook quiz cell, original)*
→ (a) Use `'valid'` if edge regions aren't useful — true. (c) `'same'` preserves dimensions (at stride 1) — true. (d) `'valid'` drops non-fitting regions — true.

**Q4. `12 × 12` input, `3 × 3` filter, stride 1, no padding — output size?** *(notebook quiz cell, original)*
→ `O = (12 + 0 − 3) / 1 + 1 = 10`. **Output: 10 × 10.**

**Q5. Why does a CNN need far fewer parameters than an MLP for the same image task?** *(adapted from `andrewekhalel/MLQuestions`, "convolutional neural network")*
→ **Weight sharing + local connectivity.** One filter's weights are reused across every spatial position (parameter sharing), and each output looks only at a small `k × k` neighbourhood (not the full image). Compared to a Dense layer that wires every input pixel to every output neuron, this slashes the parameter count by orders of magnitude — and the saved capacity goes into learning *more useful* features.

**Q6. What's the role of pooling in a CNN, and what does it sacrifice?** *(adapted from `chiphuyen/ml-interviews-book`, vision chapter)*
→ Pooling **downsamples** feature maps — `MaxPool2D()` with defaults halves H and W — which (a) reduces compute and memory in deeper layers, and (b) buys a small amount of local translation invariance (a tiny shift of the input → roughly the same max). It sacrifices fine spatial detail (the exact pixel position of a feature), which is fine for classification but problematic for tasks that need precise localisation (segmentation, detection).

**Q7. Translation invariance — what is it, and how does a CNN achieve it?** *(adapted from `alexeygrigorev/data-science-interviews`, deep-learning section)*
→ **Translation invariance** = the output prediction doesn't change when the input is shifted. A CNN gets it from **weight sharing** (the same filter is applied at every position, so a shifted pattern produces a shifted feature map — *equivariance*) plus pooling (which collapses local position information into the max/avg). An MLP has no such property — every pixel position has its own dedicated weights, so a shift is a totally new input.

**Q8. A `224 × 224 × 3` input passes through `Conv2D(64, 7, strides=2, padding='same')`. What's the output shape and parameter count?** *(common FAANG CV question; matches ResNet's stem)*
→ With `'same'` padding *and* stride 2, the formula gives `O = ⌈N / S⌉ = 112`. **Output shape: `(112, 112, 64)`.** Param count: `(7 × 7 × 3 + 1) × 64 = 9,472`.

**Q9. Why ReLU and not sigmoid for hidden layers?** *(adapted from `alexeygrigorev/data-science-interviews`)*
→ Sigmoid's derivative is at most 0.25 and goes to 0 for `|x|` large → in a deep stack, gradients **vanish** and the lower layers stop learning. ReLU's derivative is 1 for positive inputs → no vanishing in the positive regime, training stays healthy. The cost is the **dead-ReLU** failure mode (a neuron whose input is permanently negative never updates), cured by proper initialisation or LeakyReLU.

**Q10. Why are odd kernel sizes (3, 5, 7) preferred over even ones?** *(adapted from `andrewekhalel/MLQuestions`)*
→ Odd kernels have a single **integer centre pixel**, so the output of each conv is naturally aligned with the input. Even kernels (2×2, 4×4) centre at `(0.5, 0.5)` and produce subtle alignment artefacts when stacked — the output drifts by half a pixel each layer.

**Q11. In this notebook, after `Conv2D(16, 3, padding='same')` then `MaxPooling2D()`, what is the output shape?** *(notebook-specific, original)*
→ Conv: `(128 + 2 − 3)/1 + 1 = 128` → `(128, 128, 16)`. MaxPool: `(128 + 0 − 2)/2 + 1 = 64` → **`(64, 64, 16)`**.

**Q12. What's the parameter count of this notebook's single Conv2D layer, and where do the CNN's remaining 16.78M parameters live?** *(notebook-specific, original)*
→ Conv2D: `(3 × 3 × 3 + 1) × 16 = 448` params — peanuts. The other ~16.78M live in `Dense(256)` after `Flatten`: `65,536 × 256 + 256 = 16,777,472`. This is *exactly* why Module 2 replaces Flatten with `GlobalAveragePooling2D`, dropping this Dense to `16 × 256 + 256 ≈ 4,400` params.

**Q13. Why doesn't this notebook use `softmax` in the hidden Dense layers, only the final one?** *(common FAANG question, original)*
→ Softmax collapses its input into a probability distribution (sums to 1). Applying that inside a hidden layer would arbitrarily restrict the representations the layer can produce — most of the dynamic range is squashed. Hidden layers use ReLU (or similar) for representational power; softmax is reserved for the *final* layer where you want a probability distribution over classes.

[🔝 Back to top](#top)

## 🪞 Extra ladder for this notebook — Conv vs MLP parameter math

**Basic** — params in a Dense layer.
```python
# Dense(256) on a (49,152,) input
# 49_152 × 256 + 256 = 12,583,168 params
```

**Intermediate** — params in a Conv2D layer.
```python
# Conv2D(16, 3) on (128, 128, 3)
# (3 × 3 × 3 + 1) × 16 = 448 params
```

**Advanced** — the same Conv2D produces a `(128, 128, 16)` feature map (with `'same'` padding). The Conv itself has 448 params; it's the Flatten + Dense *after* pooling that blows up to 16.78M. Replace `Flatten + Dense(10)` with `GlobalAveragePooling2D + Dense(10)` and you're down to ~170 params total for the head — 100,000× fewer than the MLP equivalent. *That* is the CNN advantage, distilled.

[🔝 Back to top](#top)

## What comes next

This notebook leaves you with **~50% test acc and a ~40% train/val gap** (textbook overfitting). [Notebook 2 →](../2.Tackling%20Overfitting%20in%20CNN/) tackles that gap with deeper conv stacks, dropout, BatchNorm, augmentation, and weight decay — bringing test accuracy to ~78%.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
