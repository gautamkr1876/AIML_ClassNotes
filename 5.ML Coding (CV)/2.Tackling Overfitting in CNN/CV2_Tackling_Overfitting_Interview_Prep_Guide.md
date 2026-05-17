<a id="top"></a>
# CV Notebook 2 — Tackling Overfitting in CNNs (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §2](../CV_Revision_Guide.md#2-module2).

## What this notebook actually demonstrates

The baseline CNN from Notebook 1 hit **51.6% test acc with a 40-point train/val gap** — textbook overfitting. This notebook iteratively applies the regularization toolkit and tracks accuracy + gap at each step:

| Iteration | Technique Added | Train | Val | Test | Gap (Train − Val) |
|---|---|---|---|---|---|
| 0. Baseline (single Conv) | — | 99.9% | 59.8% | 51.6% | **40.1%** |
| 1. Deeper conv blocks | Architecture | 66.9% | 59.2% | 68.6% | 7.6% |
| 2. + Dropout + BatchNorm | Regularization | ~88% | 67.7% | 75.3% | 20.4% |
| 3. + L2 weight decay + LR scheduler | Weight decay | ~85% | ~78% | 77.4% | ~7% |
| 4. + Data augmentation | Augmentation | ~87% | 80%+ | **78.0%** | ~7% |

**Test accuracy: 51.6% → 78.0%** with the gap closing from 40% to ~7%. Same data, same dataset.

## 🪜 Mental anchors for this notebook

- **Overfitting = memorization.** Train ↑, val plateaus → memorize.
- **Dropout = implicit ensemble** of sub-networks.
- **BatchNorm = layer-input rescaling** with learned shift/scale.
- **Augmentation = virtual data multiplication** — same content, varied appearance.

## 🧠 Key cell-by-cell walkthrough

### 1. Deeper conv stack (Iteration 1)
```python
model = keras.Sequential([
    layers.Conv2D(16,  3, padding='same', activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32,  3, padding='same', activation='relu'),  layers.MaxPooling2D(),
    layers.Conv2D(64,  3, padding='same', activation='relu'),  layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),  layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),  layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),           # avoid huge Flatten + Dense
    layers.Dense(256, activation='relu'),
    layers.Dense(10,  activation='softmax'),
])
```
Five Conv blocks with increasing filters (16→256). `GlobalAveragePooling2D()` replaces `Flatten`, slashing the final-dense params.

### 2. Add BatchNorm + Dropout (Iteration 2)
```python
def block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)         # after activation
    x = layers.MaxPooling2D()(x)
    return x

# Dropout in the dense head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(10, activation='softmax')(x)
```

### 3. Add L2 weight decay (Iteration 3)
```python
layers.Conv2D(32, 3, padding='same',
              kernel_regularizer=regularizers.l2(1e-3))
```
Apply `l2(1e-3)` to *every* Conv and Dense `kernel_regularizer`. Adds `λ Σ w²` to the loss — gentle, smooth shrinkage.

### 4. Callbacks: EarlyStopping + ReduceLROnPlateau
```python
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=5
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, min_delta=0.001,
        restore_best_weights=True,
    ),
]
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

### 5. Data augmentation (Iteration 4)
```python
augment = keras.Sequential([
    layers.Resizing(156, 156),                 # over-resize then crop
    layers.RandomCrop(128, 128),
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.Rescaling(1.0/255),
])

# Train-only augmentation
train_ds = train_data.map(
    lambda x, y: (augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Val/test get only the rescaling
val_ds = val_data.map(lambda x, y: (layers.Rescaling(1./255)(x), y))
```

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `layers.BatchNormalization()` | Per-batch normalization with learnable γ, β |
| `layers.Dropout(rate)` | Stochastic activation zeroing, train-only |
| `regularizers.l1(λ)` / `l2(λ)` | Weight-decay penalties as `kernel_regularizer=` |
| `layers.RandomFlip / Rotation / Crop / Brightness / Contrast / Translation` | Built-in augmentation layers |
| `layers.GlobalAveragePooling2D()` | Replaces `Flatten + huge Dense` |
| `keras.callbacks.EarlyStopping(monitor, patience, min_delta, restore_best_weights)` | Auto-halt + restore best |
| `keras.callbacks.ReduceLROnPlateau(monitor, factor, patience)` | LR decay on plateau |

## ⚠️ Notebook-specific gotchas

1. **Augmentation must be train-only.** Apply to val/test and your metrics become noise.
2. **`BatchNormalization` at inference uses running stats** — call `model(x, training=False)` explicitly if needed.
3. **`Dropout` is disabled at inference** automatically. Don't manually scale activations.
4. **`EarlyStopping(restore_best_weights=True)` is the safe default** — without it, you get the last-epoch weights (often worse than the peak).
5. **`ReduceLROnPlateau` interacts with `EarlyStopping`** — give LR-reduce a smaller patience (5) than early-stop patience (10), so LR gets to act before training stops.
6. **L2 = `λ Σ w²`, NOT just `Σ w²`.** `kernel_regularizer=l2(1e-3)` — the `λ` is `1e-3`.
7. **Don't `Dropout` inside Conv blocks at 0.5** — that's too aggressive for spatial features. Use 0.1–0.25 there; reserve 0.5 for dense heads.

## 🎯 Notebook quiz cells (verbatim)

**Q1.** Why does Batch Normalization help?
→ Normalizes layer inputs per minibatch, stabilizes training, allows larger LRs without divergence.

**Q2.** What is dropout, and why is it required?
→ Reduces co-adaptation by stochastically zeroing activations — implicit ensemble of sub-networks.

**Q3.** How does BatchNorm behave at inference?
→ Uses **running mean/variance** (EMA) computed during training, not the current batch.

**Q4.** Where do you place Dropout vs BatchNorm?
→ Dropout: between Dense layers (0.5), or post-MaxPool (0.1–0.25).
→ BatchNorm: typically after activation (some teams do before; pick one).

**Q5.** L1 vs L2 — practical difference?
→ L1 produces *sparse* weights (feature selection). L2 produces smooth shrinkage.

## 🪞 Extra ladder — diagnosing train vs val gap

**Basic** — if train acc ≫ val acc → overfitting → add regularization.

**Intermediate** — if both low → underfitting → bigger model or longer training.

**Advanced** — flat train AND flat val AND non-zero gap → data leak, label noise, or a frozen layer you forgot to unfreeze. *Look at sample predictions*, not just numbers.

## What comes next

This notebook gets you from 51% → 78% test acc *training from scratch*. [Notebook 3 →](../3.Transfer%20learning%201/) hits 79% on a 10-class landmarks dataset with **only 737 training images** by using a pretrained backbone.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
