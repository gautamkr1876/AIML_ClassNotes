<a id="top"></a>
# CV Notebook 1 — Intro to CV & CNN Fundamentals (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §1](../CV_Revision_Guide.md#1-module1).

## What this notebook actually demonstrates

A side-by-side proof that **a tiny CNN beats a massive MLP** on image classification:
- **Dataset:** Clothing-Small (10 classes — t-shirt, shirt, jacket, dress, skirt, shorts, pants, longsleeve, shoes, hat). 3,068 train / 341 val / 372 test. Imbalanced (t-shirt 795, hat 123).
- **Baseline MLP:** flatten + 2 Dense → **50.6M params, ~36% test acc**.
- **Tiny CNN:** one Conv2D + Pool + Dense → **16.8M params, ~50% test acc** (+40% relative improvement with 67% fewer params).

The whole notebook is the canonical "specialization beats brute force" lesson.

## 🪜 Mental anchors for this notebook

- **Tea-room analogy** — Python lists scatter ingredients across the room (pointer-chase); NumPy / ndarray images line them up on one shelf.
- **CNN inductive biases** — locality + stationarity + compositionality. These three biases buy you 100× fewer params and better generalization on image data.
- **Output-shape formula** — `O = (N + 2P − F) / S + 1`. Drill it.

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
Each dataset is a `tf.data.Dataset` yielding `(image_batch, label_batch)`. Default `batch_size=32`, default `image_size=(256,256)`.

### 3. Preprocessing — resize + rescale to [0,1]
```python
preprocess = keras.Sequential([
    layers.Resizing(128, 128),     # uniform input size
    layers.Rescaling(1./255),      # map 0-255 → 0-1
])
train_ds = train_data.map(lambda x, y: (preprocess(x), y))
val_ds   = val_data  .map(lambda x, y: (preprocess(x), y))
test_ds  = test_data .map(lambda x, y: (preprocess(x), y))
```

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
2. Every input → every neuron in next layer = parameter explosion.
3. No translation invariance — shifted shirt is "a new image" to an MLP.

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

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `tf.keras.utils.image_dataset_from_directory` | Folder structure must be `path/class_name/*.jpg` |
| `layers.Resizing(h, w)` | Resize inside the model graph |
| `layers.Rescaling(scale, offset)` | `1./255` is standard for `[0, 1]` |
| `layers.Conv2D(filters, kernel_size, strides, padding, activation)` | The CNN workhorse |
| `layers.MaxPooling2D(pool_size, strides)` | Defaults: `pool_size=2, strides=2` |
| `layers.Flatten()` | Required before `Dense` |
| `layers.Dense(units, activation)` | `'softmax'` on output for classification |
| `model.compile(optimizer, loss, metrics)` | `'sparse_categorical_crossentropy'` for integer labels |
| `model.fit(train_ds, validation_data, epochs)` | Returns a `History` object |

## ⚠️ Notebook-specific gotchas

1. **Imbalanced classes** — t-shirt has 795 train images, hat has 123. Don't trust raw accuracy alone; look at per-class precision/recall.
2. **`image_dataset_from_directory` is alphabetical** — the class index is determined by folder name sort order. Print `train_data.class_names` to verify.
3. **`sparse_categorical_crossentropy` requires integer labels.** If you one-hot encoded by accident, switch to `categorical_crossentropy`.
4. **`Dense` after `Flatten` has the bulk of the parameters** — `Flatten((64,64,16)) → Dense(256)` is `65_536 × 256 ≈ 17M`. This is why Module 2 swaps in `GlobalAveragePooling2D`.

## 🎯 Notebook quiz cells (verbatim — drill these)

**Q1.** Which layer rescales input values to `[0, 1]`?
→ **(c) Rescaling layer** (`Rescaling(1./255)`).

**Q2.** Which is FALSE about MLPs for images?
→ **(b)** "MLPs have fewer params than CNNs" — actually the opposite.

**Q3.** True statements about padding?
→ **(a, c, d):**
- (a) Use `'valid'` if edges aren't useful.
- (c) `'same'` preserves dimensions.
- (d) `'valid'` drops non-fitting regions.

**Q4.** `12 × 12` input, `3 × 3` filter, stride 1, no padding → output size?
→ `(12 - 3 + 0) / 1 + 1 = ` **10 × 10**.

## 🪞 Extra ladder for this notebook — Conv vs MLP param math

**Basic** — params in a Dense layer.
`Dense(256)` on a `(49,152,)` input: `49_152 × 256 + 256 = 12,583,168`.

**Intermediate** — params in a Conv2D layer.
`Conv2D(16, 3)` on `(128, 128, 3)`: `(3 × 3 × 3 + 1) × 16 = 448`.

**Advanced** — the same Conv2D produces a `(128, 128, 16)` feature map (with `'same'` padding). The Conv itself has 448 params; the Flatten + Dense after pooling is what blows up. Replace the Dense after Flatten with `GlobalAveragePooling2D + Dense(10)` and you're down to ~6K params total — 8000× fewer than the MLP equivalent. *That* is the CNN advantage, distilled.

## What comes next

This notebook leaves you with **50% test acc and a 40% train/val gap** (overfitting). [Notebook 2 →](../2.Tackling%20Overfitting%20in%20CNN/) tackles that with dropout, BatchNorm, augmentation, and weight decay.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
