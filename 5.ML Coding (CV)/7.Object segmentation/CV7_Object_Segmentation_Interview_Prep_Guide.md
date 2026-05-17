<a id="top"></a>
# CV Notebook 7 — Object Segmentation (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §7](../CV_Revision_Guide.md#7-module7).

## What this notebook actually demonstrates

The **Google-Pixel portrait-mode** use case: classify every pixel as person / background, then blur the background. Builds two segmentation architectures and compares them:

| Architecture | Skip connections | Edges | Dice |
|---|---|---|---|
| FCN-8 (with VGG-16 encoder) | Few, late-stage (Pool3, Pool4 added) | **Coarse** | High |
| **U-Net** | **Dense, at every level** | **Sharp** | High |

Then introduces **Mask R-CNN** (instance segmentation) and **DeepLab v3** (atrous convolutions) conceptually.

Dataset: **18,698 portrait images, 128×128 RGB** with binary person-vs-background masks.

## 🪜 Mental anchor for this notebook

**Segmentation = classification per pixel.** Output is the same shape as the input, but each pixel carries a class label. The architectural challenge: encoder downsamples (good at "what"), decoder upsamples (good at "where"). **Skip connections** bridge them so boundaries stay sharp.

Picture the U-shape: information flows down the encoder, hits the bottleneck, flows back up the decoder, with side-channels at every level carrying spatial detail forward.

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset prep — masks to one-hot
```python
images = np.load('img_uint8.npy')         # (18698, 128, 128, 3)
masks  = np.load('msk_uint8.npy')         # (18698, 128, 128, 1) 0/255
masks[masks > 0] = 1                       # binarize
Y = tf.keras.utils.to_categorical(masks)  # (18698, 128, 128, 2) one-hot
```

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
**Result:** Strong Dice but blurry boundaries — skips happen too late.

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

### 5. Training
```python
model = unet()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[dice_coefficient])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=16)
```

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

## ⚠️ Notebook-specific gotchas

1. **`Conv2DTranspose` produces checkerboard artifacts** when `kernel_size % strides ≠ 0`. U-Net avoids them by using `UpSampling2D + Conv2D` instead.
2. **Pixel accuracy is misleading on imbalanced masks.** A "predict background everywhere" model scores 95%+ on portraits where only ~30% of pixels are person. Use **mean IoU** or **Dice** as the metric.
3. **Dice loss + BCE/CE is the standard combo.** Pure Dice has unstable gradients early in training; pure CE gets dominated by majority class. The two together are the de-facto pattern for medical/portrait segmentation.
4. **Class index 0 is background in most pretrained segmentation models.** Verify before computing IoU.
5. **Mask R-CNN outputs masks at low resolution** (28×28 per ROI). You must upsample them to bounding-box size at display time.
6. **`to_categorical(masks)` requires integer-valued masks** in `[0, num_classes)`. If your masks are 0/255 binary, divide by 255 first.

## 🎯 Notebook quiz cells (verbatim)

**Q1.** Why is Dice loss better than cross-entropy for segmentation?
→ Cross-entropy treats pixels independently and gets dominated by the majority class. Dice measures global mask overlap → robust to imbalance.

**Q2.** Transposed convolution vs UpSampling2D — when which?
→ Transposed conv has learnable parameters → can adapt, but checkerboard artifacts if kernel & stride mismatch. UpSampling2D is fixed (cheap, no artifacts). U-Net uses UpSampling for efficiency.

**Q3.** How do skip connections improve segmentation?
→ Encoder destroys spatial detail; decoder restores it. Skip connections inject fine-grained spatial info from the encoder directly into the decoder at matching resolutions → sharp edges.

**Q4.** Mask R-CNN vs U-Net?
→ Mask R-CNN is **instance-aware** (RPN finds objects, then segments each one). U-Net is **semantic** only (one mask, no instance distinction). For "count and segment each car separately" → Mask R-CNN.

**Q5.** Why does FCN produce coarser edges than U-Net?
→ FCN's decoder has only a few upsampling stages with late skip additions → information lost by the time we upsample. U-Net's symmetric structure with skip concatenation at every level preserves much more detail.

## 🪞 Extra ladder — segmentation evaluation

**Basic** — pixel accuracy: `(correct pixels) / (total pixels)`. Misleading on imbalance.

**Intermediate** — **mean IoU** across classes: `mean_c [ IoU_c ]`. The standard PASCAL VOC metric.

**Advanced** — **boundary metrics** like the Boundary F-measure for edge-quality evaluation. Or **panoptic quality** (PQ) for panoptic segmentation: combines instance recognition with semantic accuracy.

## What comes next

This notebook covered semantic + instance segmentation. [Notebook 8 →](../8.Siamese%20network/) shifts gears entirely: **metric learning** with Siamese networks for **one-shot verification** (faces, signatures).

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
