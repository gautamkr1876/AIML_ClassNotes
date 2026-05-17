<a id="top"></a>
# CV Notebook 3 — Transfer Learning (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §3](../CV_Revision_Guide.md#3-module3).

## What this notebook actually demonstrates

Recognize **10 famous landmarks** (Niagara Falls, Eiffel Tower, Great Wall, Machu Picchu, Gateway of India, etc.) from a **737-image training set**. With so little data, two approaches:

| Approach | Test Accuracy | Trainable Params |
|---|---|---|
| VGG16 trained **from scratch** | **11.6%** | 14.96M |
| VGG16 + ImageNet weights + frozen conv + new classifier head | **79.1%** | **0.25M** |

**6.8× test-accuracy improvement** with **60× fewer trainable params** by *not* starting from scratch. The canonical lesson: someone else trained the hard bits — reuse them.

## 🪜 Mental anchor for this notebook

**Layer hierarchy:** early CNN layers learn generic visual features (edges, textures, colors); deeper layers learn task-specific patterns. The generic stuff transfers freely between any image task → freeze it. Retrain only the classifier head.

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset
**Famous landmarks** (10 classes):
- Niagara Falls, Golden Gate Bridge, Kantanagar Temple, Eiffel Tower, Washington Monument, Hanging Temple, Forth Bridge, Great Wall of China, Machu Picchu, Gateway of India
- Train: 737 images (~70/class). Val: 155 (~15/class). Test: 43 (~4/class).
- Pretty balanced; resized to `(224, 224, 3)` to match VGG.

### 2. Load + resize
```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "landmarks/train", image_size=(224, 224), batch_size=32,
)
val_ds   = tf.keras.utils.image_dataset_from_directory(
    "landmarks/val",   image_size=(224, 224), batch_size=32,
)
test_ds  = tf.keras.utils.image_dataset_from_directory(
    "landmarks/test",  image_size=(224, 224), batch_size=32,
)

preprocess = keras.Sequential([layers.Rescaling(1./255)])
train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
```

### 3. Approach A — from scratch (the bad approach for small data)
```python
vgg_scratch = tf.keras.applications.VGG16(
    weights=None,                # ← random init, not pretrained
    include_top=True,
    classes=10,
    input_shape=(224, 224, 3),
)
vgg_scratch.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
vgg_scratch.fit(train_ds, validation_data=val_ds, epochs=5)
# → test accuracy ~12% (barely above random for 10 classes)
```

### 4. Approach B — transfer learning (the right approach for small data)
```python
# Step 1: load pretrained backbone, drop original classifier
backbone = tf.keras.applications.VGG16(
    weights='imagenet',          # ImageNet weights
    include_top=False,           # remove 1000-class FC head
    input_shape=(224, 224, 3),
)

# Step 2: freeze the backbone
backbone.trainable = False

# Step 3: attach a new classifier head
model = keras.Sequential([
    backbone,                            # 14.71M frozen params
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),  # 0.25M trainable
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
# → test accuracy ~79% in 5 epochs
```

### 5. Inspecting the architecture
```python
model.summary()
# vgg16 (Functional)             14,714,688 (non-trainable)
# flatten                                  0
# dense                              250,890 (trainable)  ← only this is learned
# Total params: 14,965,578
# Trainable params:    250,890
```

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))` | Pretrained VGG16 with classifier removed |
| `weights='imagenet'` | Loads pretrained weights |
| `weights=None` | Random init (no transfer) |
| `include_top=False` | Drops original 1000-class FC head |
| `backbone.trainable = False` | Freezes ALL layers in the backbone |
| `for layer in backbone.layers[:-4]: layer.trainable = False` | Per-layer fine-grained freezing |

### Pretrained backbones available in `tf.keras.applications`
| Model | Params | ImageNet Top-1 | Input |
|---|---|---|---|
| VGG16 | 138M | 74.4% | 224 |
| VGG19 | 144M | 74.5% | 224 |
| ResNet50 | 25.6M | 76.0% | 224 |
| ResNet101 | 44.7M | 77.4% | 224 |
| InceptionV3 | 23.9M | 78.8% | 299 |
| Xception | 22.9M | 79.0% | 299 |
| EfficientNetB0 | 5.3M | 77.1% | 224 |
| MobileNetV2 | 3.5M | 71.8% | 224 |

## ⚠️ Notebook-specific gotchas

1. **`input_shape` must match the backbone's training size** — VGG/ResNet expect 224×224, Inception/Xception expect 299×299. Wrong size → silent shape error or shrinking output.
2. **Each backbone family has its own `preprocess_input`.** For VGG it's `tf.keras.applications.vgg16.preprocess_input` (subtracts ImageNet RGB mean, doesn't divide by 255). For ResNet: same idea but different mean. For EfficientNet/MobileNet: actually expects `[0, 255]` integers! Mixing up preprocessing → much worse accuracy.
3. **`backbone.trainable = False` freezes layers** but does NOT freeze BatchNorm's running stats. Call `backbone(x, training=False)` at inference too.
4. **Fine-tuning order matters.** Always train the head *first* (with backbone frozen), then unfreeze top blocks with a 10× smaller LR (`1e-5`). Reversing the order destroys ImageNet features.
5. **`Flatten()` after VGG16 produces a huge feature vector** (`7 × 7 × 512 = 25,088`). Consider `GlobalAveragePooling2D()` instead → 512 features, much fewer head params, often equal accuracy.

## 🎯 Notebook quiz cells (verbatim)

**Q1.** Why freeze the pretrained convolutional layers?
→ Preserves ImageNet features (universal low-level edges, textures); prevents overfitting on small datasets.

**Q2.** How many parameters are trainable in transfer learning?
→ Only the new classifier head — **250,890 params** (vs 14.7M total).

**Q3.** Why are 3×3 filters used instead of larger ones in VGG?
→ Two stacked 3×3 (`18` params) cover the same receptive field as one 5×5 (`25` params) — fewer params + extra non-linearity.

**Q4.** What is top-k accuracy and why is it useful?
→ Score 1 if true label is in top-k predictions. Useful when class boundaries are ambiguous (husky vs malamute).

**Q5.** Why does transfer learning outperform from-scratch on small datasets?
→ ImageNet weights encode universal visual features learned from 14M images; fine-tuning needs far less data and far fewer epochs.

## 🪞 Extra ladder — feature extraction → fine-tuning

**Basic** — feature extraction: freeze everything, train head.
```python
backbone.trainable = False
model.fit(train_ds, epochs=5)
```

**Intermediate** — fine-tune top 4 layers after the head converges, with 10× smaller LR.
```python
backbone.trainable = True
for layer in backbone.layers[:-4]: layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)   # smaller LR!
model.fit(train_ds, epochs=10)
```

**Advanced** — **differential learning rates** per stage:
```python
# PyTorch idiom: parameter groups with different LRs
optim = torch.optim.AdamW([
    {'params': backbone.head.parameters(),     'lr': 1e-3},   # new layers, high LR
    {'params': backbone.layer4.parameters(),   'lr': 1e-4},   # last block, medium
    {'params': backbone.layer3.parameters(),   'lr': 1e-5},   # block before, low
])
```

## What comes next

This notebook used VGG16 *for classification*. [Notebook 4 →](../4.Image%20Similarity%20%3A%20Understanding%20Embeddings/) uses **ResNet-50** the same way — but instead of classifying, extracts the penultimate-layer **embedding** and uses it for reverse image search.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
