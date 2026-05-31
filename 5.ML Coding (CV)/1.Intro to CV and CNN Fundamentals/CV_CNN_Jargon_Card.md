<a id="top"></a>
# CV & CNN Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading the notebook, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`CV_CNN_Reading_Brief.md`](./CV_CNN_Reading_Brief.md) FIRST. This card is just the dictionary.

---

## A

**Accuracy** — The fraction of predictions a model got right, expressed as a percentage. Your notebook's MLP gets ~27% test accuracy; the CNN gets ~51%. Accuracy alone can be misleading on imbalanced data (use a confusion matrix to dig deeper).

**Activation function** — A small rule applied between layers (like ReLU, sigmoid, softmax) that lets the network learn curved patterns, not just straight-line ones. Without it, a deep network is mathematically the same as one layer — useless.

**Adam (optimizer)** — The algorithm that updates the network's weights during training. Think of it as a smart hill-climber that auto-tunes its step size. It's the safe default and what your notebook uses.

**ANN (Artificial Neural Network)** — Generic umbrella term for any neural network. In this notebook it's used interchangeably with **MLP** (a "vanilla" or fully-connected network).

**Average Pooling** — Same idea as Max Pooling, but takes the **average** of each patch instead of the max. Typically used near the end of a CNN, just before the dense layers, to summarise features.

## B

**Batch** — A small group of training samples (e.g., 32 images) processed together in one step. Bigger batches = faster but more memory. All images in a batch must have the **same shape**, which is why the notebook resizes to 128×128.

**Bias** — A single extra number added inside a neuron, on top of the weighted inputs. Lets the neuron shift its output up or down. Every Dense/Conv layer has biases unless you explicitly turn them off.

## C

**Channels (RGB)** — The "depth" of an image. A color image has **3 channels** (Red, Green, Blue) stacked on top of each other; a grayscale image has 1. A 3×3 kernel applied to an RGB image is actually 3×3×3 — it sees all three channels at once.

**CNN (Convolutional Neural Network)** — A neural network designed for images. Instead of treating each pixel as a separate input (like an MLP), it scans the image with small filters (kernels) and learns which patterns matter. The whole point of this notebook.

**Compositionality** — One of the three image properties CNNs exploit. Complex shapes are built from simpler ones: edges → corners → eyes → face. Pooling and stacked layers naturally encode this hierarchy.

**Confusion Matrix** — A grid showing predicted-class vs true-class counts. The diagonal = correct predictions; off-diagonal = mistakes (and **which** class got confused for which). Used in your notebook after both MLP and CNN training.

**Conv2D** — Keras's name for a 2D convolutional layer. The arguments matter: `filters=` (how many kernels), `kernel_size=` (e.g., 3 for a 3×3 kernel), `stride=` (how many pixels to jump), `padding=` ("same" or "valid").

**Convolution** — The core operation of a CNN. A small grid (kernel) slides across the image. At each spot it multiplies its numbers by the underlying pixels, sums them up, and outputs one number. The result is a new image called a **feature map**.

**Cross-entropy (loss)** — The "wrongness score" a classifier tries to minimise during training. Low cross-entropy = predicted probabilities match the true class. Used together with softmax at the output. Your notebook uses `sparse_categorical_crossentropy` (same idea, just expects integer labels instead of one-hot vectors).

## D

**Dense layer (a.k.a. Fully Connected layer)** — A regular neural-network layer where every input is connected to every neuron. Lots of parameters. CNNs use a few Dense layers at the very end for classification, after convolution has done the heavy lifting.

**DNN (Deep Neural Network)** — Any neural network with more than one hidden layer. CNN, MLP, RNN are all DNNs once they have depth.

## E

**Edge Detection** — Finding boundaries between light and dark regions in an image. A classic computer-vision task — and a special hand-designed kernel (the **Sobel filter**) does this without any learning. Your notebook uses it to show what a kernel "sees."

**Epoch** — One full pass through the entire training dataset. 10 epochs = the network sees every training image 10 times. More epochs = more learning, but also risk of **overfitting**.

## F

**Feature Map** — The output you get after a kernel slides over the whole image. It's a new "image" where each pixel says "how strongly was the kernel's pattern present here?" One filter produces one feature map.

**Filter** — Another word for **Kernel**. The two terms are used interchangeably in the notebook.

**Flatten** — Converting a 2D image (or feature map) into a 1D row of numbers. A CNN's last step before the Dense layers — Dense layers expect 1D input. Example: a 64×64×16 feature map flattens into a vector of length 16,384.

**Fully Connected layer** — Same as **Dense layer**.

## H

**Hidden Layer** — Any layer between the input and the output. Your notebook's MLP has two hidden layers (sizes 1024 and 256); its CNN has one (size 256).

## K

**Kernel** — A small grid of numbers (e.g., 3×3) that slides across an image during convolution. Different kernels detect different things — edges, corners, textures. In a CNN, the kernel values are **learned** during training (you don't pick them by hand). Same thing as a **filter**.

## L

**Locality** — One of the three image properties CNNs exploit. Pixels near each other are related (your eye and your eyebrow are next to each other); pixels far apart usually aren't. CNNs only look at small neighbourhoods at a time, which is exactly the right inductive bias for images.

**Loss** — The number the network is trying to make as small as possible during training. Lower loss = better predictions. The specific loss function depends on the task — cross-entropy for classification, mean-squared error for regression, etc.

## M

**Max Pooling** — A shrinking step. Looks at small patches (usually 2×2) of an image and keeps only the **biggest** value. Halves the size of the data and has **zero learnable parameters** — it's a rule, not a learned operation. Makes the network robust to tiny shifts in feature position.

**MLP (Multi-Layer Perceptron)** — The "vanilla" neural network. Every neuron in one layer is connected to every neuron in the next (fully connected). Works fine for tabular data but **fails on images** because it ignores that nearby pixels are related — exactly what your notebook demonstrates (27% accuracy).

## N

**Normalization (Rescaling)** — Scaling pixel values from the 0–255 range down to 0–1 (divide by 255). Helps the network train faster and more stably. Your notebook does this in the preprocessing step.

## O

**OCR (Optical Character Recognition)** — Reading printed or handwritten text from an image. One of the computer vision applications listed in the notebook intro.

**Overfitting** — When a network memorises the training data but fails on new data. Sign: training accuracy keeps rising while validation accuracy stalls or drops. Caused by too many parameters, too little data, or too many epochs.

## P

**Padding** — Adding a border of zeros around the image before the kernel slides on it. Without padding ("valid"), the output shrinks every layer. With `padding="same"`, the output stays the same size as the input. Your notebook's CNN uses `padding="same"`.

**Parameter (a.k.a. Weight)** — A number inside the network that gets **learned** during training. More parameters = more capacity but more memory and more overfitting risk. A 48-filter 3×3 conv layer on an RGB input has `((3×3×3)+1)×48 = 1,334` learnable parameters — the notebook walks through exactly this calculation.

**Parameter Sharing** — A key CNN trick: the same kernel slides across the entire image, so the **same weights** detect a feature wherever it appears. This dramatically reduces parameter count compared to a Dense layer and is how CNNs encode **stationarity**.

**Pooling** — Umbrella term for the shrinking step. See **Max Pooling** and **Average Pooling**. The formula for output size is `floor((n - f) / s) + 1` where `n` = input side, `f` = pool window size, `s` = stride.

## R

**Receptive Field** — The patch of the original image that ends up influencing one output pixel deep in the network. Early layers see tiny patches (3×3); deeper layers see much larger regions because patches stack on top of each other. This is **compositionality** in action.

**ReLU (Rectified Linear Unit)** — A one-line rule applied between layers: "if the number is negative, replace it with zero; otherwise keep it." The default activation function for CNN hidden layers. Cheap, fast, works well.

**RGB** — The three colour channels of a standard image: Red, Green, Blue. See **Channels**.

## S

**Sigmoid** — An activation function that squashes any number into the 0–1 range. Used at the output for **binary** classification. For multi-class classification (10 fashion classes in your notebook), you use **softmax** instead.

**Sobel Filter** — A famous hand-designed 3×3 kernel that detects edges (horizontal or vertical) without any learning. Used in your notebook (via `cv2.Sobel`) to show concretely what a kernel "sees" before learned kernels are introduced.

**Softmax** — An activation function applied at the very end of a multi-class classifier. Takes a row of numbers and squashes them into probabilities that **add up to 1**. Example: `[2.1, 0.3, 5.5]` → `[0.03, 0.005, 0.96]` ("96% sure it's class 3").

**Sparse Connectivity (a.k.a. Sparse Connections)** — A CNN feature: each output neuron only looks at a small patch of the previous layer (not the whole thing). This encodes **locality** and slashes the parameter count vs. an MLP.

**Stationarity** — One of the three image properties CNNs exploit. The same pattern can appear anywhere in an image (a cat's whisker is a whisker whether it's at the top-left or bottom-right). CNNs handle this via **parameter sharing**.

**Stride** — How many pixels the kernel jumps each step. Stride 1 = slide one pixel at a time (slow, detailed output). Stride 2 = skip every other pixel (output is roughly half the size). Your notebook's Conv2D uses stride 1.

## T

**Test Accuracy** — Accuracy measured on data the model has **never seen during training**. The real measure of how well it generalises. Your notebook's headline numbers (27% MLP, 51% CNN) are test accuracies.

---

[🔝 Back to top](#top)
