<a id="top"></a>
# CV Notebook 3 — Transfer Learning (Deep Dive)

> Per-notebook companion to the master guide. For the cross-notebook synthesis, cheat sheet, glossary and drill, see [`../CV_Revision_Guide.md` §3](../CV_Revision_Guide.md#3-module3). **This deep-dive is standalone** — every concept the notebook touches has its own Concept Definition Template entry below; you should never have to leave this file to understand a term.

## What this notebook actually demonstrates

Recognise **10 famous landmarks** (Niagara Falls, Eiffel Tower, Great Wall, Machu Picchu, Gateway of India, Golden Gate Bridge, Kantanagar Temple, Washington Monument, Hanging Temple, Forth Bridge) from a **737-image training set**. With so little data, two approaches:

| Approach | Test Accuracy | Trainable Params |
|---|---|---|
| VGG16 trained **from scratch** | **11.6%** | 14.96M |
| VGG16 + ImageNet weights + frozen conv + new classifier head | **79.1%** | **0.25M** |

**6.8× test-accuracy improvement** with **60× fewer trainable params** by *not* starting from scratch. The canonical lesson: **someone else trained the hard bits on ImageNet — reuse them.**

## 🪜 Mental anchors for this notebook

- **Layer hierarchy** — early CNN layers learn generic visual features (edges, textures, colours); deeper layers learn task-specific patterns. The generic stuff transfers freely between any image task → freeze it. Retrain only the classifier head. *Why it matters here:* this is the principle that lets a 737-image dataset reach 79% accuracy — almost all the "vision" comes from someone else's ImageNet training.
- **Borrow the eyes, learn a new tongue** — the pretrained backbone is the "eyes" (frozen); the new classifier head is the "tongue" (trained from scratch on landmarks). *Why it matters here:* this is the two-line summary of the notebook's transfer-learning recipe.
- **Big LR for new layers, tiny LR for borrowed ones** — when you eventually unfreeze backbone layers for fine-tuning, the LR must drop by ~10× (e.g. `1e-3 → 1e-5`). *Why it matters here:* the advanced ladder shows the differential-LR pattern explicitly.

[🔝 Back to top](#top)

<a id="walkthrough"></a>
## 📖 Concept walkthroughs

> Every concept this notebook touches gets a full Concept Definition Template entry below. No link-only blocks. A beginner reading top-to-bottom should walk away with the *what / why / how / where / related* for every term, with the notebook's specific 737-image dataset, VGG16 backbone, and parameter counts substituted in.

### Transfer learning

> **🪜 Mental model:** *Don't relearn what someone else already mastered.* Start with weights that already know edges, textures, and object parts — only train the last few layers for your specific task.

**What it is.** **Transfer learning** is the practice of taking a neural network already trained on a large dataset and reusing its learned weights as the starting point for a new, related task. In CV that almost always means: start with a CNN pretrained on **ImageNet** (1.28M images, 1,000 classes), throw away its 1,000-class classifier head, attach a new head for your task, and train (with most of the network frozen). The pretrained backbone has already learned generic visual features (edges, textures, parts) that transfer to almost any image task. In this notebook, the source model is VGG-16, the target task is 10-class landmark classification, and the result is **79.1% test accuracy on 737 training images** — vs **11.6% if you train the same VGG16 from random init**.

**Why it matters.** Training a CNN from scratch needs ~10⁵–10⁶ labelled images plus days of GPU time. With transfer learning you can hit 80%+ accuracy on a custom task with **5,000 labelled images and 30 minutes of training** — or, as this notebook shows, even on **737 images**. It's the most impactful trick in practical CV — and one of the most common interview topics ("you have 1,000 images for a new task — what's your approach?"). Below ~10k labelled images, transfer learning is essentially mandatory.

**How it works.**
1. Pick a pretrained backbone (e.g., `tf.keras.applications.VGG16(weights='imagenet', include_top=False)`).
2. **Freeze** the backbone weights (so they don't change during training): `backbone.trainable = False`.
3. Attach a fresh classifier head matched to your task (e.g., `Flatten → Dense(10, softmax)` for 10 classes).
4. Train only the head with a normal LR (`1e-3` Adam default) until it converges.
5. *Optional fine-tuning step (later):* unfreeze the top conv blocks, train everything with a **10× smaller** LR (`1e-5`) to gently adapt the borrowed features to your domain.

**Where it's used.** Almost every real-world CV project. This notebook uses it to go from 12% test accuracy (from-scratch VGG) to 79% (pretrained VGG + frozen backbone + new head) on a 10-class landmarks task. In FAANG interviews, "fine-tune a pretrained model" is the default answer to any small-data CV problem.

**Related terms.**
- **Pretrained model** — the source model whose weights you reuse (next entry).
- **Backbone** — the convolutional body, minus the classifier.
- **Feature extraction** — variant where the backbone is fully frozen.
- **Fine-tuning** — variant where top layers of the backbone are unfrozen and retrained gently.
- **Catastrophic forgetting** — what happens if you fine-tune with too large an LR — the model forgets ImageNet knowledge.

```python
# The notebook's transfer-learning recipe in 4 lines
backbone = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                       input_shape=(224, 224, 3))
backbone.trainable = False                          # freeze
model = keras.Sequential([backbone, layers.Flatten(),
                          layers.Dense(10, activation='softmax')])
```

**Gotcha.** Domain mismatch matters — ImageNet → medical X-rays transfers poorly because ImageNet has no X-rays. Use a domain-specific pretrained model (CheXNet for X-rays) when one exists. For natural images, the landmarks task in this notebook, or pets, food, products, etc., ImageNet is the right source.

### Pre-trained model

> **🪜 Mental model:** *Borrow someone's gym progress.* They put in the millions of reps; you get to start strong.

**What it is.** A **pre-trained model** is a neural network whose weights have already been learned on some (usually huge) dataset. In CV the canonical pretrained source is **ImageNet** (1,000 classes, 1.28M images), but you can also pretrain on JFT-300M (Google's billion-image private set), or on self-supervised tasks (SimCLR, MAE) where labels aren't needed. Pretrained weights are typically packaged as a file (50–500 MB) that the framework downloads once and caches. In this notebook, `tf.keras.applications.VGG16(weights='imagenet', ...)` triggers an automatic download (~58 MB) of weights learned on **ImageNet ILSVRC-2012**; subsequent runs load from `~/.keras/models/`.

**Why it matters.** Pretrained weights encode an enormous amount of visual knowledge — they're the closest thing CV has to a "starter kit." Without them, anyone working on a small custom dataset would be stuck. The notebook is the proof: 737 images is not enough to train a 14.96M-param VGG16 from scratch (you get 11.6% — barely above the 10% random baseline for 10 classes). With pretrained weights, the same 737 images get you to 79.1%.

**How it works.** The original team trained the model on ImageNet for weeks of GPU-time. The resulting weights file is uploaded to a model zoo (TensorFlow Hub, PyTorch Hub, HuggingFace). You download it once; the framework caches it locally. Loading the model + weights takes a few seconds after the first download. From that point, you can use the model exactly like one you'd trained yourself — it just *already knows things*.

**Where it's used.**
- **Cell 4** of this notebook: `VGG16(weights='imagenet', include_top=False)` is the source pretrained model.
- As the starting point for almost every transfer-learning task.
- Available in `keras.applications` (VGG, ResNet, EfficientNet, MobileNet, ...), `torchvision.models`, and HuggingFace `transformers` (for NLP/CV transformers).

**Related terms.**
- **ImageNet** — the dataset most CV models are pre-trained on (next entry).
- **Model zoo** — collection of pretrained models (TF Hub, PyTorch Hub, HuggingFace).
- **Self-supervised pretraining** — no labels needed; SimCLR, BYOL, MAE.
- **`weights='imagenet'` vs `weights=None`** — pretrained vs random init; the notebook tries both.

**Gotcha.** Each pretrained model expects its **own preprocessing**. VGG subtracts the ImageNet mean (in BGR order!); ResNet uses a different mean; EfficientNet expects integer pixels in `[0, 255]` and does its own normalisation. Use `<model>.preprocess_input` — never just `/255` blindly. This notebook simplifies by using `Rescaling(1./255)` (works adequately with VGG since the network is robust enough), but in production you'd use `tf.keras.applications.vgg16.preprocess_input`.

### Backbone

> **🪜 Mental model:** *The skeleton without the hat.* All the conv layers, but without the final classifier — ready for you to attach a new head.

**What it is.** The **backbone** of a CNN is the convolutional portion — typically all layers up to (but not including) the final classifier head. In code this is what you get from `tf.keras.applications.VGG16(weights='imagenet', include_top=False)` — the `include_top=False` flag strips VGG16's original 1,000-class FC head, leaving just the convolutional body. The backbone's output for a `(224, 224, 3)` input is a feature map of shape `(7, 7, 512)` — a stack of 512 feature maps at 7×7 resolution.

**Why it matters.** The backbone is where 99% of the parameters and ImageNet knowledge live. VGG16's backbone has **14.71M parameters** (out of 14.96M total in the transfer-learning model). The classifier head is a couple of Dense layers — trivial to retrain. So when adapting to a new task, you keep the backbone frozen and replace the head. This split — frozen backbone, trainable head — is what makes transfer learning work on small data.

**How it works.** Conceptually identical to a regular CNN — Conv → ReLU → Pool blocks stacked into a deep funnel. The specific shape depends on the architecture (16 layers for VGG-16, 50 for ResNet-50). You can call the backbone like a function on input: `features = backbone(x, training=False)` produces the feature map. In Keras you usually wrap it in a `Sequential` model and let the framework manage the forward pass.

**Where it's used.**
- **Cell 4** of this notebook: `backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
- As the frozen feature extractor in transfer learning.
- As the embedding extractor in image similarity (Module 4, where the backbone's output is the embedding).
- As the feature provider for detection heads (Modules 5–6) and segmentation decoders (Module 7).

**Related terms.**
- **Head** — the task-specific layers atop the backbone (classifier, detector, mask head).
- **Stem** — the very first conv block of a backbone (often distinct, e.g., 7×7 conv in ResNet).
- **`include_top=False`** — Keras flag that strips the classifier.
- **Embedding** — what a backbone produces when used as a feature extractor.

```python
# Notebook usage
backbone = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,                      # strip the 1000-class head
    input_shape=(224, 224, 3),
)
# Output shape: (7, 7, 512) for 224×224 input
```

**Gotcha.** Different backbones have different output shapes and channel counts. VGG16 → `(7, 7, 512)`; ResNet-50 → `(7, 7, 2048)`; EfficientNetB0 → `(7, 7, 1280)`. Always run `backbone.summary()` and verify the output shape before attaching a head. Mismatching the head's input dimension to the backbone's output is a frequent source of silent shape bugs.

### Classifier head

> **🪜 Mental model:** *The "tongue" you train.* The small stack of layers on top of the backbone that maps generic features → your specific class labels.

**What it is.** The **classifier head** (or just "head") is the set of layers on top of the backbone that produce the final task-specific output — typically a `Flatten` (or `GlobalAveragePooling2D`) followed by one or two `Dense` layers ending in a softmax. In this notebook the head is `Flatten → Dense(10, softmax)` — minimal, just enough to convert the `(7, 7, 512)` feature map into 10 class probabilities. Parameter count: `(7 × 7 × 512) × 10 + 10 = 250,890` — versus the backbone's 14.71M frozen weights.

**Why it matters.** The head is the *only* trainable part of a feature-extraction model. Its job is to learn the *task-specific* combination of pretrained features. Because it's small (250k params here), it converges in 5–20 epochs even on a 737-image dataset — fast and safe. Replace it with something more sophisticated (e.g., `GlobalAveragePooling2D → Dense(256, relu) → Dropout → Dense(10, softmax)`) when you need more capacity, but start minimal.

**How it works.**
1. The backbone produces a feature map of shape `(7, 7, 512)`.
2. `Flatten()` reshapes it to a single vector of `7 × 7 × 512 = 25,088` numbers.
3. `Dense(10, activation='softmax')` maps that vector to 10 class probabilities via `softmax(W @ x + b)`.
4. During `fit`, gradients flow only into the head's `W` and `b` — the backbone's frozen weights aren't touched.

**Where it's used.**
- **Cell 4** of this notebook: `Flatten + Dense(10, softmax)` atop the frozen VGG16 backbone.
- Every transfer-learning project; the head is what you customise.
- Variants: `GlobalAveragePooling2D → Dense` (fewer params), `Flatten → Dense(256, relu) → Dropout → Dense(num_classes)` (more capacity), etc.

**Related terms.**
- **Backbone** — what the head sits on top of.
- **`include_top=False`** — the Keras flag that removes the original 1,000-class head, making room for yours.
- **GlobalAveragePooling2D** — modern alternative to `Flatten` that produces a 512-dim vector instead of a 25,088-dim one — fewer head params.
- **Softmax** — the final activation that turns logits into class probabilities.

```python
model = keras.Sequential([
    backbone,                                       # 14.71M frozen
    layers.Flatten(),                               # (7,7,512) → (25_088,)
    layers.Dense(10, activation='softmax'),         # 250,890 trainable
])
```

**Gotcha.** **`Flatten()` after VGG16 produces a huge feature vector** (`7 × 7 × 512 = 25,088`). For a single Dense(10) layer atop it, that's still only 250k params — fine. But if you add a `Dense(256)` in the middle, you'd get `25,088 × 256 ≈ 6.4M` params just in that layer. Consider `GlobalAveragePooling2D()` instead — output is 512 features, and `Dense(256)` atop it is only `131k` params.

### ImageNet

> **🪜 Mental model:** *The CV pretraining dataset.* 1.28M labelled images over 1,000 categories — what almost every pretrained CNN started from.

**What it is.** **ImageNet** is a large image dataset organised by WordNet noun synonyms. The version used in CV pretraining is **ILSVRC-2012**: ~1.28M training images, 50,000 validation, 100,000 test, distributed across 1,000 fine-grained classes (specific dog breeds, mushroom types, household objects, vehicles, etc.). "ImageNet-pretrained" almost always means trained on this 1,000-class subset. When the notebook calls `VGG16(weights='imagenet')`, the weights it downloads are the ones that won (or near-won) the ILSVRC-2014 competition.

**Why it matters.** ImageNet's diversity (1,000 classes covering animals, objects, scenes) makes a model pretrained on it a great general-purpose visual feature extractor. Features that distinguish 'husky' from 'malamute' are precisely the kind that generalise to a custom landmark or product-classification task. **In this notebook,** the reason VGG16-with-ImageNet hits 79% on landmarks (a domain ImageNet doesn't directly cover!) is that ImageNet's edges, textures, and shape detectors transfer cleanly to natural images of any kind.

**How it works.** The ILSVRC competition (2010–2017) drove the field — AlexNet (2012) was the breakout, ResNet (2015) reached human-level on the test set. Every architecture in `keras.applications` was originally trained on this 1,000-class ImageNet. The pretrained weights file simply encodes the result of that training.

**Where it's used.**
- As the source domain for almost all CV transfer learning.
- **Cell 4** of this notebook: the `'imagenet'` string maps to a specific file of weights that VGG16 trained on ILSVRC-2012.
- State-of-the-art top-1 accuracies on ImageNet-1K (~88% for modern ConvNeXt/ViT) are still the standard benchmark.

**Related terms.**
- **ILSVRC** — the official competition / benchmark (2010–2017).
- **Top-1 / top-5 accuracy** — fraction of test images where the correct class is the top prediction / among top 5.
- **JFT-300M** — Google's private 300M-image dataset used for newer pretraining.
- **CheXNet** — example of a *domain-specific* pretrained model (chest X-rays).

**Gotcha.** ImageNet skews heavily toward animals, objects, and natural scenes. For specialised domains (medical X-rays, satellite imagery, fashion close-ups) you may want a domain-specific pretrained model. For this notebook's landmarks (natural photographs), ImageNet is a great match.

### Feature extraction — freeze backbone, train head

> **🪜 Mental model:** *Use the pretrained eyes; learn only a new tongue.* The backbone is frozen — it transforms images into features just like it learned to on ImageNet. The new head is the only part that actually trains for landmarks.

**What it is.** **Feature extraction** is the simpler of the two transfer-learning recipes. You load a pretrained backbone, set `backbone.trainable = False` (which freezes every weight inside it), and attach a small new classifier head on top. During `model.fit`, only the new head's weights update; the backbone passes images through unchanged. Effectively the backbone is acting as a **fixed feature extractor** — it consumes raw images and outputs a feature vector, and you train a small classifier on top of those vectors.

**Why it matters.** This is the notebook's **Approach B** — the one that reaches 79.1% test accuracy with only 250k trainable params. It's fast (no backprop through the 14.71M-param backbone), it's safe (zero risk of destroying ImageNet features), and it's the right starting point any time your dataset is small (1k–10k images). On 737 training images, *anything else* would either overfit or under-use ImageNet's knowledge — feature extraction is the sweet spot.

**How it works.**
1. **Load the pretrained backbone** with `include_top=False` and the matching `input_shape` (`(224, 224, 3)` for VGG).
2. **Freeze it:** `backbone.trainable = False`. This sets all 14.71M weights to non-trainable; gradient descent ignores them.
3. **Stack a new head** on top: `Flatten()` → `Dense(num_classes, activation='softmax')`. (Optionally insert `GlobalAveragePooling2D` instead of `Flatten` — see the gotchas.)
4. **Compile** with a normal learning rate (`Adam(1e-3)`, the default) — only the new head is training, and it needs a fast LR to converge in few epochs.
5. **Train** for a handful of epochs (5–20). Watch val loss; usually it plateaus quickly because the head is small.

**Where it's used.**
- **Cell 4** of this notebook — the headline "Approach B" experiment.
- As the default first step of *any* transfer-learning project.
- In Keras: `tf.keras.applications.<Backbone>(weights='imagenet', include_top=False)`.
- In PyTorch: `torchvision.models.<backbone>(weights=...).requires_grad_(False)` plus a new head.
- In `fastai`: it's literally the default behaviour of `Learner.fit_one_cycle` before you call `unfreeze()`.

**Related terms.**
- **Fine-tuning** — the *next* step where you unfreeze top backbone layers (next entry).
- **`backbone.trainable = False`** — the Keras one-liner that does the freezing.
- **Embedding** — the backbone's output, used as input to the new head (and also for similarity in Module 4).
- **Frozen vs trainable params** — what `model.summary()` shows; `Non-trainable params: 14,714,688` is the signature of a frozen backbone.

```python
backbone = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                        input_shape=(224, 224, 3))
backbone.trainable = False                       # ← freeze (key line)
model = keras.Sequential([
    backbone,
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),      # ← only this trains
])
```

**Gotcha.** `backbone.trainable = False` freezes the layer **weights** but does **not** stop BatchNorm layers from updating their running statistics. To keep BN's running mean/variance fixed at ImageNet's values, either (a) call the backbone with `backbone(x, training=False)`, or (b) freeze before compile. Mismatch between training-mode BN stats and inference-mode causes mysterious accuracy gaps in transfer learning.

### Fine-tuning — unfreeze top, train with small LR

> **🪜 Mental model:** *Adjust the tail of the funnel, gently.* Unfreeze only the last few conv blocks (the most task-specific ones) and train them with a tiny LR so the ImageNet features aren't destroyed.

**What it is.** **Fine-tuning** is the second-stage transfer-learning recipe, layered on top of feature extraction. After the new head has converged, you unfreeze the top few backbone layers (typically the last 1–4 conv blocks) and continue training with a **10× smaller learning rate** (e.g., `1e-5` instead of `1e-3`). The bottom layers (which learned universal features — edges, textures) stay frozen; only the top, more task-specific layers adapt to your data. The notebook's "Extra ladder" section walks through this exact pattern; the headline experiment in cell 4 stops at feature extraction.

**Why it matters.** Fine-tuning typically gains another **2–10% accuracy** over feature extraction alone — useful when you have enough data (~5,000+ images) and need every accuracy point. But it's risky: too-large a learning rate can **catastrophically forget** the pretrained ImageNet features in a single epoch, leaving you worse off than feature extraction. The 10× LR drop and the "always-after-feature-extraction" ordering are the safety rails that make this technique work.

**How it works.**
1. **First, finish feature extraction.** Don't unfreeze until the new head has converged — at that point its gradients are small and safe to backprop through the backbone.
2. **Unfreeze the top N layers.** Typically: `backbone.trainable = True`, then `for layer in backbone.layers[:-N]: layer.trainable = False`. For VGG16, N=4 (the last conv block) is a common choice.
3. **Re-compile** with a smaller LR: `model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)`. **Re-compile is mandatory** — changing the optimiser without recompiling silently keeps the old LR.
4. **Train for a few more epochs.** Watch val loss carefully — if it stops improving or gets worse, you've over-fine-tuned; back off (lower LR or fewer unfrozen layers).
5. **Optionally use discriminative LRs** — even smaller LR on the layers closer to the bottom (e.g., `1e-6` for early conv blocks, `1e-5` for the last block, `1e-4` for the head).

**Where it's used.**
- When you have enough data and feature extraction has plateaued.
- Modern recipes (PyTorch Lightning, `fastai`'s `learner.fine_tune`) automate the unfreezing schedule and discriminative LRs.
- In FAANG ML rounds, "you've done feature extraction and got 78% — what next?" expects fine-tuning as the answer.

**Related terms.**
- **Feature extraction** — the *first* step (backbone fully frozen).
- **Catastrophic forgetting** — losing pretrained knowledge from too-large fine-tuning LR; the failure mode this whole recipe is designed to avoid.
- **Discriminative learning rates** — different LRs per depth; smaller at the bottom, larger at the top.
- **Layer-wise unfreezing** — a gradual schedule that unfreezes one block at a time over several epoch ranges (ULMFiT-style).

```python
# After feature extraction has converged:
backbone.trainable = True
for layer in backbone.layers[:-4]:
    layer.trainable = False                              # unfreeze only last 4 layers
model.compile(optimizer=keras.optimizers.Adam(1e-5),     # 10× smaller LR
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

**Gotcha.** **Always do feature extraction first, then fine-tune.** Unfreezing on day one — when the head is still randomly initialised — back-propagates large gradients through the backbone and destroys the pretrained weights. This is the #1 transfer-learning mistake.

### Learning-rate strategy for fine-tuning (why `1e-5`, not `1e-3`)

> **🪜 Mental model:** *The bigger the step, the more damage to existing knowledge.* The new head needs big LR to learn fast from scratch; the pretrained backbone needs tiny LR so each gradient step nudges, doesn't shove.

**What it is.** The rule that during fine-tuning, the learning rate for the unfrozen backbone layers must be **roughly 10× smaller** than the LR you'd use to train from scratch. Typical numbers: scratch / head LR = `1e-3` (Adam default); fine-tune LR = `1e-5`. When using a single optimiser for both the head and the unfrozen backbone, the simplest pattern is to use the small LR (`1e-5`) for everything during the fine-tune phase — the head has already converged at `1e-3`, so a small LR won't hurt it much, and the backbone needs the small LR to stay safe.

**Why it matters.** Each gradient descent step is `w_new = w_old − lr · gradient`. If `gradient` is normal-sized and `lr` is large, the update wipes out the existing weight value. For random init that's fine — you're trying to wipe out random noise. But for **pretrained weights** that already encode years of ImageNet training, a large `lr` quickly destroys that knowledge — the phenomenon called **catastrophic forgetting**. The 10× LR drop is the safety margin that keeps each step in the "nudge" regime, not the "shove" regime.

**How it works.**
1. **Feature extraction stage:** `model.compile(optimizer=Adam(1e-3))`. Train the new head until val loss plateaus.
2. **Fine-tuning stage:** unfreeze the top N backbone layers. **Re-compile** with `Adam(1e-5)`. Train for a few more epochs.
3. **Optional further decay:** add `ReduceLROnPlateau(factor=0.3, patience=3)` to gently shrink LR further if val loss plateaus.
4. **Optional discriminative LRs:** apply different LRs per depth (PyTorch parameter groups, or `fastai`'s `slice(1e-5, 1e-3)`). Most projects don't need this.

**Where it's used.**
- Every fine-tuning workflow in modern CV.
- `keras.optimizers.Adam(1e-5)` is the standard fine-tune LR — memorise it.
- The same rule applies in NLP (BERT fine-tuning typically uses `2e-5` to `5e-5`).

**Related terms.**
- **Catastrophic forgetting** — what too-large LR causes; the failure mode this rule exists to prevent.
- **Discriminative LRs** — per-layer LRs; smaller at the bottom, larger at the top of the network.
- **Warmup** — gradually increase LR over the first few hundred steps; helps stabilise even very small LRs.
- **`AdamW`** — variant of Adam that decouples weight decay from gradient updates; preferred for fine-tuning in modern code.

**Gotcha.** **Re-compile after switching LRs.** If you change `optimizer=Adam(1e-5)` without calling `model.compile(...)` again, Keras silently keeps the old optimiser (with the old LR). Always `compile` immediately after changing the LR or unfreezing layers.

### Catastrophic forgetting

> **🪜 Mental model:** *Erase the textbook while studying for the new test.* If you train the borrowed backbone too aggressively, it forgets everything it learned on ImageNet — leaving you with random weights again.

**What it is.** **Catastrophic forgetting** is the phenomenon where fine-tuning a pretrained model on a new task destroys the original learned knowledge — large gradient steps overwrite ImageNet features with noise from the new (small) task. The result: the model's accuracy *drops* during fine-tuning, often below what feature extraction alone achieved. It's not a bug in any layer; it's a consequence of choosing the wrong LR or unfreezing too early.

**Why it matters.** Catastrophic forgetting is the #1 transfer-learning failure mode. You start with a great pretrained model, run fine-tuning, and discover the accuracy got *worse*. The whole point of transfer learning is the borrowed knowledge — if you destroy it, you're back to training from scratch, which (as this notebook's Approach A shows) gives you 11.6% on 737 images.

**How it works.** The gradient of the loss w.r.t. each weight is roughly proportional to (1) how far the model's current prediction is from the target and (2) the LR. Early in fine-tuning, the model's predictions are still bad on the new task (because the head was random), so gradients are large. A large LR multiplies those gradients into huge weight updates that overwrite the pretrained values.

**Where it's used (as a concept to be aware of):**
- Every fine-tuning workflow — the whole "feature extraction first, then fine-tune with `1e-5`" recipe exists to prevent this.
- In NLP fine-tuning (BERT, GPT) the same dynamic applies; LRs of `2e-5` to `5e-5` are the canonical safe range.

**Related terms.**
- **Fine-tuning** — the procedure that risks catastrophic forgetting if done wrong.
- **Discriminative learning rates** — per-layer LRs; the canonical defence.
- **Warmup** — gradual LR increase from a tiny value; another defence.
- **Layer-wise unfreezing** — only unfreeze one block at a time; another defence.

**Gotcha.** Two protective rules together: (a) **always do feature extraction first** (so the head's gradients are small before they backprop into the backbone), and (b) **use `1e-5` for fine-tuning**, not `1e-3`. Skip either rule and you risk catastrophic forgetting.

### VGG-16 — the specific backbone this notebook uses

> **🪜 Mental model:** *The "uniform stacks of 3×3 convs" CNN.* No fancy tricks — just lots of small filters and max-pooling, hence "**V**ery deep **G**roup → VGG."

**What it is.** **VGG-16** (Simonyan & Zisserman, 2014) is a 16-layer-deep CNN built out of uniform stacks of 3×3 convolutions interleaved with 2×2 max-pooling. The architecture is: `Conv(64) → Conv(64) → Pool → Conv(128) → Conv(128) → Pool → Conv(256) × 3 → Pool → Conv(512) × 3 → Pool → Conv(512) × 3 → Pool → FC(4096) → FC(4096) → FC(1000)`. With the 1,000-class head, it's 138M parameters total; with `include_top=False` (this notebook), the backbone alone is 14.71M parameters.

**Why it matters.** VGG is mostly of historical importance now (slow, huge), but its simplicity makes it the canonical teaching backbone — easy to inspect, easy to attach a new head to, and the layer hierarchy is easy to understand. In production you'd reach for ResNet-50 or EfficientNetB0 instead (better accuracy, far fewer parameters). For *this notebook*, VGG is chosen for clarity, not speed.

**How it works.** Every conv layer is 3×3 with `padding='same'` and ReLU activation. Pooling layers halve the spatial size after each block. The result is a feature-map funnel: input `(224, 224, 3) → (112, 112, 64) → (56, 56, 128) → (28, 28, 256) → (14, 14, 512) → (7, 7, 512)`. The classifier head then collapses that `(7, 7, 512)` into class scores.

**Where it's used.**
- **Cell 4** of this notebook: `VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`.
- Reference / teaching backbone in many courses.
- Style transfer (the original VGG-style-transfer paper uses VGG-19's feature maps).
- Rarely used in production today — superseded by ResNet, EfficientNet, ConvNeXt.

**Related terms.**
- **ResNet** — the modern default; residual connections enable much deeper networks at fewer params.
- **EfficientNet** — best accuracy/parameter tradeoff; uses compound scaling.
- **MobileNet** — designed for phones; depthwise-separable convs.
- **`include_top=False`** — drops the original 1,000-class FC head.

**Specific facts for this notebook:**
- Input size: **`(224, 224, 3)`** (fixed — the pretrained weights are tuned for this).
- Output (with `include_top=False`): **`(7, 7, 512)`**.
- Total VGG16 params: **138M**; **14.71M** after dropping the FC head.
- New head with `Flatten + Dense(10)`: `(7 × 7 × 512) × 10 + 10 = 250,890` trainable params.

**Gotcha.** VGG (and ResNet) expect 224×224 input; Inception and Xception expect 299×299; EfficientNet variants expect 224 → 600 depending on the variant. Resizing your data to the wrong size silently breaks the pretrained weights — the network still runs but accuracy plummets.

### Top-k accuracy

> **🪜 Mental model:** *Allowed a few guesses.* Top-1 = "is the model's #1 guess correct?"; top-5 = "is the correct class anywhere in the model's top 5 guesses?"

**What it is.** **Top-k accuracy** is the fraction of test examples for which the true label is among the model's top `k` predictions. For a 10-class classifier, **top-1 accuracy** is what you usually mean by "accuracy" — the model's single best guess matches the label. **Top-5 accuracy** is more lenient — the correct class is in the top 5 highest-probability outputs. The notebook's quiz Q4 asks "what is top-k accuracy and why is it useful?"

**Why it matters.** Top-5 is the historical ImageNet metric (because some ImageNet classes are genuinely hard to distinguish — husky vs malamute, multiple types of mushroom). Top-1 is stricter and what you'd report for a deployment context. For this notebook's 10-class landmarks, top-1 is the natural metric (only 10 classes, so top-5 would be ~50% by random guessing).

**How it works.** For each test image: (1) compute the model's `K`-dim probability output; (2) sort the probabilities descending; (3) check whether the true label is in the top `k` of the sorted list; (4) average across all test images.

**Where it's used.**
- **ImageNet benchmark** — top-5 is the historical metric; modern reporting also includes top-1.
- Multi-class problems with ambiguous boundaries (fine-grained classification).
- Recommendation systems (top-k retrieval accuracy).

**Related terms.**
- **Top-1 accuracy** — the strict version (same as "accuracy" for a hard classifier).
- **Per-class accuracy** — the breakdown when classes are imbalanced.
- **Confusion matrix** — finer-grained view of where the model confuses classes.

**Gotcha.** Top-k is more lenient than top-1, so don't compare a top-5 number to someone else's top-1 number. Always state which `k` you're using.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset
**Famous landmarks** (10 classes):
- Niagara Falls, Golden Gate Bridge, Kantanagar Temple, Eiffel Tower, Washington Monument, Hanging Temple, Forth Bridge, Great Wall of China, Machu Picchu, Gateway of India.
- Train: 737 images (~70 / class). Val: 155 (~15 / class). Test: 43 (~4 / class).
- Pretty balanced; resized to `(224, 224, 3)` to match VGG's expected input.

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
# Total params:        14,965,578
# Trainable params:       250,890
# Non-trainable params: 14,714,688
```

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))` | Pretrained VGG16 with classifier removed |
| `weights='imagenet'` | Loads pretrained weights |
| `weights=None` | Random init (no transfer) |
| `include_top=False` | Drops original 1000-class FC head |
| `backbone.trainable = False` | Freezes ALL layers in the backbone |
| `for layer in backbone.layers[:-4]: layer.trainable = False` | Per-layer fine-grained freezing (for fine-tuning) |

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

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **`input_shape` must match the backbone's training size** — VGG/ResNet expect 224×224, Inception/Xception expect 299×299. Wrong size → silent shape error or shrinking output.
2. **Each backbone family has its own `preprocess_input`.** For VGG it's `tf.keras.applications.vgg16.preprocess_input` (subtracts ImageNet RGB mean, doesn't divide by 255). For ResNet: same idea but different mean. For EfficientNet/MobileNet: actually expects `[0, 255]` integers! Mixing up preprocessing → much worse accuracy.
3. **`backbone.trainable = False` freezes layers** but does NOT freeze BatchNorm's running stats. Call `backbone(x, training=False)` at inference too, or freeze before compile.
4. **Fine-tuning order matters.** Always train the head *first* (with backbone frozen), then unfreeze top blocks with a 10× smaller LR (`1e-5`). Reversing the order destroys ImageNet features.
5. **`Flatten()` after VGG16 produces a huge feature vector** (`7 × 7 × 512 = 25,088`). Consider `GlobalAveragePooling2D()` instead → 512 features, much fewer head params, often equal accuracy.
6. **Re-compile when switching from feature extraction to fine-tuning.** Changing `optimizer=Adam(1e-5)` without calling `model.compile(...)` again is silently ignored — Keras keeps the old LR.
7. **VGG's huge parameter count is mostly in its FC head**, which we strip with `include_top=False`. The 14.71M backbone is much smaller than the full 138M model.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

> ≥ 10 questions, ≥ 5 sourced and cited from canonical interview banks. Every concept in the 📖 walkthrough section above has at least one Q&A item below.

**Q1. Why freeze the pretrained convolutional layers?** *(notebook quiz cell, original)*
→ Three reasons: (a) **preserves ImageNet features** (universal low-level edges, textures); (b) **prevents overfitting** on small datasets like this notebook's 737 images; (c) **far faster training** — no backprop through 14.71M backbone weights.

**Q2. How many parameters are trainable in this notebook's transfer-learning model?** *(notebook quiz cell, original)*
→ Only the new classifier head — **250,890 params** (`(7 × 7 × 512) × 10 + 10`). The backbone's 14,714,688 params are frozen.

**Q3. Why are 3×3 filters used instead of larger ones in VGG?** *(notebook quiz cell, original)*
→ Two stacked 3×3 convs (`2 × 9 = 18` params per channel pair) cover the same **receptive field** as one 5×5 conv (`25` params) — fewer params *and* an extra ReLU non-linearity per stack → richer representations with less computation.

**Q4. What is top-k accuracy and why is it useful?** *(notebook quiz cell, original)*
→ Top-k accuracy = score 1 if the true label is in the top-k predictions. Useful when class boundaries are inherently ambiguous (husky vs malamute on ImageNet) — top-5 is the historical ImageNet metric for exactly this reason.

**Q5. Why does transfer learning outperform from-scratch on small datasets?** *(notebook quiz cell, original)*
→ ImageNet weights encode universal visual features learned from 1.28M images; you start with edges, textures, and object-part detectors already in place. Training from scratch on 737 images can't learn those features (not enough data) — so you spend all your capacity learning the wrong things.

**Q6. You have 500 labelled images for a 5-class classification problem. Walk through your full transfer-learning approach.** *(adapted from `chiphuyen/ml-interviews-book`, applied-ML chapter)*
→ (1) **Pick a pretrained backbone** — **ResNet-50 with ImageNet weights**, `include_top=False`. (2) **Freeze it** (`backbone.trainable = False`). (3) **Add a small head**: `GlobalAveragePooling2D → Dropout(0.5) → Dense(5, softmax)`. (4) **Compile** with `Adam(1e-3)` and train for 10–20 epochs *with augmentation* (flip, rotate, crop) — small data means augmentation matters more than the backbone choice. (5) **If val acc plateaus** and you want more, unfreeze the top 1–2 conv blocks and re-train with `Adam(1e-5)`. (6) **Report test accuracy once** at the end.

**Q7. What is catastrophic forgetting in transfer learning, and how do you prevent it?** *(adapted from `alexeygrigorev/data-science-interviews`)*
→ **Catastrophic forgetting** is when fine-tuning a pretrained model on a new task destroys the original learned knowledge — large gradient steps overwrite ImageNet features with task-specific noise, leaving you worse off than feature extraction alone. Prevention: (a) **always do feature extraction first** so the new head's gradients are small before they reach the backbone; (b) use a **10× smaller LR** (`1e-5`) during fine-tuning; (c) optionally unfreeze layers gradually rather than all at once.

**Q8. Your pretrained model has BatchNorm layers. Why does `backbone.trainable = False` *not* fully freeze it?** *(adapted from `andrewekhalel/MLQuestions`, BN section)*
→ `trainable = False` freezes the **learnable** weights (`γ`, `β`) but **does not** stop BN from updating its **running mean / variance** (those aren't "weights" — they're statistics). Set `training=False` when calling the backbone (`backbone(x, training=False)`) — or freeze before compile — so BN uses ImageNet's running stats instead of updating to the new domain's batch stats. Otherwise the BN layer slowly drifts away from ImageNet's distribution and the pretrained features stop matching.

**Q9. Why are 3×3 filters in VGG often preferable to 7×7?** *(common FAANG question)*
→ Two stacked 3×3 convs cover the same **receptive field** as one 5×5 (and three stacked 3×3 cover a 7×7) — but with fewer params *and* an extra ReLU non-linearity per stack. For two 3×3 convs: `2 × 9 × C² = 18C²` params; one 5×5 conv: `25C²`. Smaller filters + deeper stacks → richer features, fewer params.

**Q10. When does transfer learning *fail*?** *(adapted from `alexeygrigorev/data-science-interviews`)*
→ When the source and target domains differ drastically. ImageNet → natural photos transfers great (this notebook). ImageNet → medical X-rays transfers poorly because there are no X-rays in ImageNet — the texture and edge detectors are different. The deeper into the backbone you go, the more domain-specific the features become, so domain mismatch hurts more for fine-tuning than for feature extraction.

**Q11. What does `include_top=False` do, and why is it non-negotiable for this notebook?** *(notebook-specific, original)*
→ It strips VGG16's original 1,000-class FC head, leaving just the convolutional backbone. Non-negotiable because: (a) we have 10 classes, not 1,000; (b) the original head outputs ImageNet class probabilities, which are meaningless for landmark classification; (c) you can't keep that head and just "swap labels" — the 1,000 outputs are tied to specific ImageNet classes, not to your label space.

**Q12. Why does this notebook resize images to `(224, 224, 3)` specifically?** *(notebook-specific, original)*
→ VGG16's pretrained weights were trained on 224×224 ImageNet images, so the filter sizes and pooling strides assume that input scale. Feeding a different size still runs (because conv layers are size-agnostic), but the pooling structure produces a feature map of a different shape, breaking the assumption that the output is `(7, 7, 512)` — and the head you attach is sized for that exact output.

**Q13. Why is `Flatten()` (not `GlobalAveragePooling2D()`) used in this notebook's head, and what would change if you swapped it?** *(notebook-specific, original)*
→ `Flatten()` keeps all `7 × 7 × 512 = 25,088` features and lets `Dense(10)` learn an arbitrary linear combination — gives the head more capacity. Swapping to `GlobalAveragePooling2D()` would reduce the feature vector to `512` numbers (one mean per channel), cutting the head from 250k to ~5k params — likely *equal or better* accuracy at this dataset size (less head capacity = less overfit risk on 737 images), and much smaller checkpoint files.

**Q14. What's the difference between `weights='imagenet'` and `weights=None` in `tf.keras.applications.VGG16(...)`?** *(notebook-specific, original)*
→ `weights='imagenet'` triggers a download of the pretrained weights file (~58 MB) and loads it. `weights=None` initialises the same architecture with random weights — equivalent to "from scratch." The notebook's Approach A uses `None` (→ 11.6% test acc), Approach B uses `'imagenet'` (→ 79.1%). Same architecture, different starting weights, 6.8× accuracy difference.

[🔝 Back to top](#top)

## 🪞 Extra ladder — feature extraction → fine-tuning

**Basic** — feature extraction: freeze everything, train only the head.
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

**Advanced** — **differential learning rates** per depth:
```python
# PyTorch idiom: parameter groups with different LRs
optim = torch.optim.AdamW([
    {'params': backbone.head.parameters(),     'lr': 1e-3},   # new layers, high LR
    {'params': backbone.layer4.parameters(),   'lr': 1e-4},   # last block, medium
    {'params': backbone.layer3.parameters(),   'lr': 1e-5},   # earlier block, low
])
```

[🔝 Back to top](#top)

## What comes next

This notebook used VGG16 *for classification*. [Notebook 4 →](../4.Image%20Similarity%20%3A%20Understanding%20Embeddings/) uses **ResNet-50** the same way — but instead of classifying, extracts the penultimate-layer **embedding** and uses it for reverse image search.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
