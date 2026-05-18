<a id="top"></a>
# CV Notebook 2 — Tackling Overfitting in CNNs (Deep Dive)

> Per-notebook companion to the master guide. For the cross-notebook synthesis, cheat sheet, glossary and drill, see [`../CV_Revision_Guide.md` §2](../CV_Revision_Guide.md#2-module2). **This deep-dive is standalone** — every concept the notebook touches has its own Concept Definition Template entry below; you should never have to leave this file to understand a term.

## What this notebook actually demonstrates

The baseline CNN from Notebook 1 hit **51.6% test acc with a 40-point train/val gap** — textbook overfitting. This notebook iteratively applies the regularisation toolkit and tracks accuracy + gap at each step:

| Iteration | Technique Added | Train | Val | Test | Gap (Train − Val) |
|---|---|---|---|---|---|
| 0. Baseline (single Conv) | — | 99.9% | 59.8% | 51.6% | **40.1%** |
| 1. Deeper conv blocks + GAP | Architecture | 66.9% | 59.2% | 68.6% | 7.6% |
| 2. + Dropout + BatchNorm | Regularisation | ~88% | 67.7% | 75.3% | 20.4% |
| 3. + L2 weight decay + LR scheduler + EarlyStopping | Weight decay | ~85% | ~78% | 77.4% | ~7% |
| 4. + Data augmentation | Augmentation | ~87% | 80%+ | **78.0%** | ~7% |

**Test accuracy: 51.6% → 78.0%** with the gap closing from 40% to ~7%. Same data, same dataset — the difference is regularisation discipline.

## 🪜 Mental anchors for this notebook

- **Overfitting = memorising the textbook, failing the exam.** Train ↑ while val plateaus or drops → the model is learning training-set noise, not the underlying pattern. *Why it matters here:* the baseline has 99.9% train / 59.8% val — textbook memorisation.
- **Dropout = implicit ensemble** of sub-networks. *Why it matters here:* iteration 2's `Dropout(0.5)` after the dense head is the cheapest closer of the train/val gap.
- **BatchNorm = re-centre the assembly line.** Per-batch normalisation with learnable shift/scale. *Why it matters here:* iteration 2 also adds BN to every conv block, which is what lets us crank the learning rate up.
- **Augmentation = virtual data multiplication.** Same content, varied appearance — flip, rotate, crop, jitter the same image so the network sees it as different. *Why it matters here:* iteration 4 closes the final gap by augmenting on the fly during training.
- **Penalty for being too big** — L2 weight decay adds a "weights-are-large" tax to the loss so the optimiser keeps weights small. *Why it matters here:* iteration 3's `l2(1e-3)` produces smooth shrinkage, which is mostly what drives the third accuracy bump.
- **Quit while you're ahead** — EarlyStopping watches val_loss and halts training when it stops improving, restoring the best weights. *Why it matters here:* the notebook trains for up to 100 epochs but the callback often halts at 30–40.

[🔝 Back to top](#top)

<a id="walkthrough"></a>
## 📖 Concept walkthroughs

> Every concept this notebook touches gets a full Concept Definition Template entry below. No link-only blocks. A beginner reading top-to-bottom should walk away with the *what / why / how / where / related* for every term, with the notebook's specific hyperparameters and code substituted in.

### Overfitting

> **🪜 Mental model:** *Memorising the textbook, failing the real exam.* The model fits training images so specifically that it can't recognise anything slightly different.

**What it is.** **Overfitting** happens when a model fits the **training** data too well — including its noise, quirks, and accidents — and consequently does *worse* on **unseen** data. The fingerprint symptom: **training accuracy keeps climbing while validation accuracy stalls or drops**, producing a widening "train/val gap". In this notebook, the baseline single-Conv CNN from Module 1 ends training at **99.9% train accuracy** with **59.8% validation accuracy** — a 40-point gap that screams overfitting.

**Why it matters.** Overfitting is the #1 reason a model "works in development but fails in production." If you don't detect and counteract it, the test-set accuracy you report won't match what users see. Every CV interview probes "how do you know your model is overfitting and what do you do?" — get this wrong and you'll fail the round. The remedy ladder this whole notebook walks (deeper architecture → Dropout → BN → L2 → augmentation) exists *because* overfitting is so common, and each rung is one tool in your toolkit.

**How it works.** Training drives the loss on the training set down via gradient descent. With enough capacity and not enough regularisation/data, the model finds weights that *interpolate* the training data exactly — including pixel-level noise that doesn't generalise. The validation loss (computed on held-out data) starts climbing while training loss keeps falling. The gap is the diagnostic.

**Where it's used.**
- Diagnosing every CNN you train — plot `train_loss` vs `val_loss` per epoch.
- **Iteration 0** of this notebook is the "before" picture; iterations 1–4 are the "after" — each one shrinks the gap.
- FAANG interviews: "your model has 99% train acc and 60% val acc — what do you try?" is the canonical opener for this topic.

**Related terms.**
- **Underfitting** — the opposite (both train and val are bad).
- **Generalisation** — the goal: low loss on data the model has never seen.
- **Regularisation** — any technique that limits the model's capacity to memorise (Dropout, BN, L1/L2, augmentation, early stopping).
- **Train/val gap** — the diagnostic signal — the difference at any given epoch.
- **Bias-variance tradeoff** — the formal statistical framework; overfitting = high variance.

**Gotcha.** A *small* train/val gap doesn't always mean "all good" — both could be plateauing at a bad level (underfit). Always check absolute accuracy too. The combo "small gap + low absolute accuracy" means you need a **bigger** model, not more regularisation.

### The diagnostic plot — train loss ↓ while val loss ↑

> **🪜 Mental model:** *Two curves diverging on a chart.* The training loss keeps falling because the model is still memorising; the validation loss starts climbing because each memorised quirk hurts on unseen data. The exact crossing point is when overfitting kicks in.

**What it is.** During training, you plot two curves per epoch: **training loss** (computed on the data the model is fitting) and **validation loss** (computed on a held-out set the model never trains on). In a healthy run both curves fall together and plateau. In an overfit run, training loss keeps dropping while validation loss bottoms out and then rises again. The "diagnostic plot" is just this side-by-side curve view — produced from the `history` object that `model.fit` returns.

**Why it matters.** Numbers in a console can hide everything; a plot makes overfitting obvious in 2 seconds. Every CV interview question about "how do you know your model is overfitting" expects this answer — point at the diverging curves and name the moment. This is also how you tune `EarlyStopping`'s `patience` argument and pick the right epoch budget. In this notebook, the gap visibly shrinks from huge (iteration 0) to small (iteration 4) — the diagnostic plot is the headline evidence.

**How it works.**
1. Call `history = model.fit(train_ds, validation_data=val_ds, epochs=N)`.
2. `history.history` is a dict with keys `loss`, `accuracy`, `val_loss`, `val_accuracy` — one value per epoch.
3. Plot `loss` and `val_loss` on the same axes, x-axis = epoch.
4. **Healthy pattern:** both curves drop, both plateau, gap stays small (≤ ~5 points).
5. **Overfitting pattern:** `loss` keeps dropping, `val_loss` reaches a minimum at some epoch *k* and then rises. The model's "best" weights are from epoch *k*.
6. **Underfitting pattern:** both curves are flat and high, neither dropping further.

**Where it's used.**
- Every Keras / PyTorch training loop.
- This notebook plots the curves after each of the 5 iterations, and the gap visibly shrinks.
- In FAANG ML rounds, "draw the loss curves for an overfit vs underfit run" is a whiteboard staple.

**Related terms.**
- **`history.history`** — the per-epoch metrics dictionary returned by `fit`.
- **EarlyStopping** — automates "stop at epoch *k* where val_loss bottomed out".
- **Train/val gap** — the difference at any given epoch; the *signal* the plot reveals.
- **Learning curve** — alternative diagnostic where you plot performance vs *training-set size* (not epoch).

```python
import matplotlib.pyplot as plt
h = history.history
plt.plot(h['loss'],     label='train')
plt.plot(h['val_loss'], label='val')
plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss')
```

**Gotcha.** A small train/val gap is *not* automatically good — both could be plateauing at a bad level (underfit). Always check absolute accuracy too. The combination "small gap + low absolute accuracy" means you need a **bigger** model, not more regularisation.

### Train / validation / test split

> **🪜 Mental model:** *Three drawers.* Train = what the model fits on; Validation = what you tune hyperparameters on; Test = sealed until the very end, opened *once*.

**What it is.** Splitting your labelled data into three disjoint subsets:
- **Train** (~70%) — gradient descent fits the weights on this.
- **Validation** (~15%) — used during training to monitor for overfitting and pick hyperparameters (LR, model size, augmentation strength).
- **Test** (~15%) — kept sealed. Used **once** at the end to report an unbiased estimate of generalisation.

In this notebook, the data already comes in three pre-split folders: `path/train/`, `path/val/`, `path/test/`. Each is loaded with a separate `image_dataset_from_directory` call.

**Why it matters.** If you tune hyperparameters on the test set, you've contaminated it — your final number is optimistic. The validation set absorbs the "tuning noise" so the test set stays clean. Skipping this discipline is the most common cause of "great test numbers, embarrassing production performance." Because this notebook's splits are folder-based, test images are physically isolated — accidental leakage is impossible.

**How it works.** Use `train_test_split` once to peel off the test set; split the rest into train + val. For small datasets, k-fold cross-validation replaces the val split. For time-series, splits must respect time order (no leakage from future to past). For pre-split datasets like this notebook's, just load each folder separately.

**Where it's used.**
- **Cell 0** of this notebook: three separate `image_dataset_from_directory` calls.
- Every supervised CV pipeline. `keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset='training')` is the auto-split form for unsplit data.

**Related terms.**
- **Holdout** — the simple train/val/test approach (what this notebook uses).
- **Cross-validation** — k-fold rotation; used when data is scarce.
- **Data leakage** — when test info sneaks into training (e.g., normalising the test set's stats into the train preprocessor).
- **Stratified split** — preserves class balance across splits.

**Gotcha.** Random splits leak when the same identity appears in train *and* val (e.g., the same person in face data). Split by **identity**, not by image. Here, because the splits are folder-based, class balance per split is fixed; if it's uneven, you live with it.

### Dropout

> **🪜 Mental model:** *Random sparring partner.* Each training step you knock out a random subset of neurons; the network can't rely on any single one.

**What it is.** **Dropout** is a regularisation layer that, **during training only**, randomly sets a fraction `p` of the previous layer's activations to zero. At inference the dropout layer is a no-op (and the framework rescales activations by `1/(1−p)` internally during training so the expected magnitude matches at inference). Typical rates: `0.5` after Dense layers (what this notebook uses), `0.1–0.25` inside conv blocks. The notebook places Dropout *only* in the dense head: `Dense(256) → BN → Dropout(0.5) → Dense(10, softmax)`.

**Why it matters.** Dropout is the *single most effective* regulariser for Dense layers. It forces the network to spread information across redundant neurons (no single neuron is reliable). At inference the model behaves like an **ensemble** of many random sub-networks — a free averaging effect that famously cut test error on AlexNet. In this notebook, Dropout is half of the iteration-2 boost (BatchNorm is the other half) that takes test acc from 68.6% → 75.3%.

**How it works.**
1. At each training forward pass, sample a Bernoulli `(1 − p)` mask the same shape as the activation.
2. Multiply activations by the mask (some go to zero).
3. Rescale the surviving activations by `1/(1 − p)` so their expected magnitude matches inference. (Keras does this for you — this is "inverted dropout".)
4. Backprop only through the surviving neurons.
5. At inference, no mask, no rescaling.

**Where it's used.**
- **Cell 2** of this notebook: `layers.Dropout(0.5)` after `Dense(256)` in the classifier head.
- `layers.Dropout(0.5)` after `Dense` layers in any classifier head.
- Less commonly inside conv blocks (typically `0.1–0.25` — see the gotcha below).

**Related terms.**
- **Bernoulli mask** — the random 0/1 array used to zero out neurons.
- **DropConnect** — drops *weights* instead of activations.
- **BatchNorm** — orthogonal regulariser; often used together with Dropout.
- **Inverted dropout** — the default scheme; rescales during *training* (not at inference).
- **Implicit ensemble** — what dropout does at inference; each forward pass is a different sub-network during training.

```python
# Notebook usage — Dropout in the dense head only
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)                      # ← 50% drop rate
out = layers.Dense(10, activation='softmax')(x)
```

**Gotcha.** Don't apply `Dropout(0.5)` inside conv blocks — that's too aggressive for spatial features (you'd kill half the feature map's pixels every step). Use `0.1–0.25` there, and reserve `0.5` for fully-connected heads. Also: don't call `model(x, training=True)` accidentally at evaluation time — that re-enables Dropout and your eval metrics become noise.

### BatchNormalization (BN)

> **🪜 Mental model:** *Re-centre the assembly line.* For each batch, shift and scale each channel so values look "standard normal" before passing to the next layer.

**What it is.** **Batch Normalisation** is a layer that, for each channel, normalises activations across the batch (and spatial dims for conv layers) to have **zero mean and unit variance**, then applies a learnable affine transform `γ · x + β`. The `γ` and `β` are trained like any other weights; the mean and variance are computed *from the current batch* at training time and from a **running exponential moving average** at inference time. This notebook places BN after every Conv (in `block(x, filters)`) and after the hidden Dense.

**Why it matters.** BN dramatically stabilises training: it allows higher learning rates, reduces sensitivity to weight initialisation, and provides a mild regularising effect (the batch-level noise in its statistics acts like a tiny dropout). Modern CNNs train roughly 2–10× faster with BN than without. In this notebook, BN is what lets the model in iteration 2 train stably at the standard Adam LR — without BN, you'd need to baby-sit the LR much more carefully.

**How it works.**
1. Compute batch mean `μ` and variance `σ²` over the batch (per channel for conv).
2. Normalise: `x̂ = (x − μ) / √(σ² + ε)` (where `ε` is a tiny constant for numerical stability).
3. Affine: `y = γ · x̂ + β`. The `γ, β` are learnable parameters.
4. Update running mean / variance with an exponential moving average (`momentum=0.99` by default).
5. **At inference**, use the running stats (not the batch's), so behaviour is deterministic.

**Where it's used.**
- **Cell 2** of this notebook: `block(x, filters)` = `Conv → ReLU → BN → MaxPool`. The notebook commits to "ReLU then BN" consistently (the alternate "BN then ReLU" is also valid — pick one).
- Every modern Conv2D block typically has a BN.
- ResNet, EfficientNet, DenseNet all rely on BN.
- The exception: very small batches (size 1–2), where BN's statistics are too noisy → use **GroupNorm** or **LayerNorm** instead.

**Related terms.**
- **LayerNorm** — normalises across features, not the batch; used in transformers.
- **GroupNorm** — for small batches.
- **Running mean / variance** — the EMA used at inference.
- **Internal covariate shift** — the original motivation in the BN paper (now somewhat disputed).
- **`training=True/False`** — controls whether BN uses batch stats or running stats; auto-set by Keras during `fit` vs `predict`.

```python
def block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)          # ← after activation in this notebook
    x = layers.MaxPooling2D()(x)
    return x
```

**Gotcha.** BN at inference uses **running** stats, not batch stats. If you forget `training=False` (TF) or `.eval()` (PyTorch), your val/test metrics look wildly different from training because BN is computing fresh statistics from the eval batch. Also: `backbone.trainable = False` freezes BN's learnable γ, β but **does not** stop the running stats from updating — bites in transfer learning (Module 3).

### Data augmentation

> **🪜 Mental model:** *Free extra training data.* Show the same image multiple times with random flips/rotations/colour shifts — the network sees it as different.

**What it is.** **Data augmentation** applies random, label-preserving transformations to training images on the fly: horizontal flip, rotation, crop, brightness, contrast, colour jitter, translation, MixUp, CutMix, RandAugment, etc. Each epoch the model sees slightly different versions of each image, learning *invariance* to those transformations. This notebook chains six built-in Keras augmentation layers: `Resizing(156, 156) → RandomCrop(128, 128) → RandomFlip('horizontal') → RandomRotation(0.1) → RandomBrightness(0.2) → RandomContrast(0.2)`. The "over-resize then random-crop" pattern is the classic recipe for cheap spatial diversity — you resize to slightly larger than the target and then crop a random window each batch.

**Why it matters.** When you have ~3,000 images (this notebook's training set), augmentation is the biggest single accuracy boost you can get for almost no cost. Every random crop is a "new" datapoint. In this notebook, augmentation is the *single biggest accuracy bump in iteration 4*, taking test acc from ~77% to ~78% **after** every other regulariser is already in place — meaning it's adding value those couldn't.

**How it works.** Apply random transforms inside the data pipeline so each batch sees different versions of the images. The transformations must **preserve the label** — flipping a digit "6" makes it look like a "9", so flip-augmentation is a bad idea for MNIST but great for clothing (a shirt is still a shirt when flipped horizontally). The augmentation pipeline is wired in via `train_ds.map(lambda x, y: (augment(x), y))`.

**Where it's used.**
- **Cell 5** of this notebook: the `augment` `Sequential` model applied to `train_ds`.
- Every CNN training pipeline that uses augmentation — which is to say, every modern one.
- PyTorch uses `torchvision.transforms` or the more powerful **albumentations** library.

**Related terms.**
- **Test-time augmentation (TTA)** — augment at *inference* and average predictions; gives a small accuracy boost. Separate from regular augmentation.
- **MixUp / CutMix** — modern augmentations that blend two images.
- **Albumentations** — the leading augmentation library for PyTorch.
- **RandAugment / AutoAugment** — search-based recipes that auto-pick the right transforms.

```python
augment = keras.Sequential([
    layers.Resizing(156, 156),         # over-resize first
    layers.RandomCrop(128, 128),       # then random crop
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.Rescaling(1.0 / 255),
])
train_ds = train_data.map(lambda x, y: (augment(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
```

**Gotcha.** Not all transforms preserve labels. Don't flip digits (a 6 flipped is a 9), don't rotate signs (a stop sign rotated 90° isn't a stop sign), don't colour-jitter when colour is the label (a red car vs a blue car). Pick augmentations that match your task's true invariances.

### Augmentation must be train-only (the trap that bites every beginner)

> **🪜 Mental model:** *The train set is a gym; the val/test set is the actual race.* You don't run the race with weights strapped to your ankles — augmentation belongs on the training side only.

**What it is.** **Augmentation is applied only to the training pipeline.** Validation and test pipelines see **raw, deterministic** images (rescaled to `[0, 1]` only). Concretely in this notebook:
```python
train_ds = train_data.map(lambda x, y: (augment(x), y), ...)  # random augmentation
val_ds   = val_data  .map(lambda x, y: (rescale(x), y))        # rescale only, no randomness
test_ds  = test_data .map(lambda x, y: (rescale(x), y))        # rescale only, no randomness
```

**Why it matters.** Two failures, one cause:
1. **Augment val/test → noise in your metrics.** A rotated/cropped val image is a *different* sample than the original; if you re-run evaluation you get a different number each time, because the augmentation is random. Your reported accuracy stops being repeatable.
2. **Augment val/test → optimistic numbers.** If your augmentation makes images "easier" on average (e.g., random crops tend to centre the object), test accuracy is inflated relative to deployment.

The correct mental rule: **augmentation distorts the gym, never the race**. The val/test pipeline must reflect deployment conditions exactly.

**How it works.**
1. Build a `Sequential` augmentation pipeline (`Resizing → RandomCrop → RandomFlip → ...`).
2. Apply it via `train_ds.map(lambda x, y: (augment(x), y))`.
3. Build a *separate* preprocessing pipeline for val/test that contains **only the deterministic steps** (resize and rescale). No random layers.
4. Verify by iterating one batch and comparing two passes — augmented batches differ between passes; raw val batches are identical across passes.

**Where it's used.** Every CNN training pipeline that uses augmentation. The split applies to the augmentation pipeline only; the model itself sees both pipelines normally. Test-time augmentation (TTA) is a *separate* technique where you augment at inference and *average* predictions over the augmented versions — but that's an explicit ensembling trick, not regular val/test evaluation.

**Related terms.**
- **Test-time augmentation (TTA)** — a separate, explicit ensembling technique done at inference.
- **Deterministic preprocessing** — what val/test gets: resize + rescale only, no random layers.
- **Data leakage** — what you get when augmentation accidentally "leaks" into val/test.
- **`training=True` / `training=False`** — most Keras augmentation layers are no-ops when `training=False`; they only randomise during fit.

**Gotcha.** Some Keras layers (notably `Resizing` and `Rescaling`) are deterministic and *do* apply at inference even with `training=False` — which is desirable (they're preprocessing, not augmentation). But `RandomFlip`, `RandomRotation`, `RandomCrop`, `RandomBrightness`, `RandomContrast` should **never** appear in the val/test pipeline. If you accidentally include them, your eval metrics become non-deterministic.

### L1 / L2 regularization (weight decay)

> **🪜 Mental model:** *Penalty for being too big.* Add a "weights-are-large" tax to the loss; the optimiser keeps weights small to dodge the tax.

**What it is.** **Weight decay** adds a penalty term to the training loss proportional to the size of the model's weights:
- **L2 (Ridge):** `λ · Σ w²` — penalises *squared* weights; shrinks every weight smoothly toward 0 (but never exactly to 0).
- **L1 (Lasso):** `λ · Σ |w|` — penalises *absolute* weights; drives many weights *exactly* to 0 (produces sparse models / does feature selection).

`λ` (the regularisation strength) controls the tradeoff between fitting and shrinkage. This notebook applies `regularizers.l2(1e-3)` as `kernel_regularizer=` on every Conv and Dense layer — so `λ = 1e-3` everywhere.

**Why it matters.** Smaller weights → smoother decision boundaries → better generalisation. L2 is the default regulariser in deep learning; L1 is rare for CNNs (used more in linear models for feature selection). Most modern optimisers (AdamW) bake in L2 weight decay directly. In this notebook, L2 is mostly responsible for iteration 3's accuracy bump from ~75% to ~77% — gentle, smooth shrinkage across every weight.

**How it works.** During gradient descent, the gradient of the penalty (`2λw` for L2, `λ · sign(w)` for L1) is added to the gradient of the data-loss. The result: every step, weights are nudged toward zero, in addition to whatever the data-fitting gradient says.

**Where it's used.**
- **Cell 3** of this notebook: `kernel_regularizer=regularizers.l2(1e-3)` on every Conv and Dense.
- `kernel_regularizer=regularizers.l2(1e-3)` argument to any `Conv2D` or `Dense` layer.
- In PyTorch: `optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)`.

**Related terms.**
- **AdamW** — variant of Adam that decouples weight decay correctly; preferred for most modern training.
- **Sparsity** — what L1 produces; many weights exactly zero.
- **Ridge / Lasso** — the linear-regression names for L2 / L1.
- **`λ` (lambda)** — the regularisation strength; typical values `1e-4` to `1e-2`.

```python
# Notebook usage — applied to every Conv and Dense
layers.Conv2D(32, 3, padding='same',
              kernel_regularizer=regularizers.l2(1e-3))
```

**Gotcha.** L1 + Adam interacts badly — Adam's adaptive learning rate distorts the L1 penalty. Use AdamW (or SGD + weight decay) instead. Also: the notebook chose **L2, not L1**, because L2 produces smooth shrinkage (every weight gets a little smaller) — appropriate for a deep CNN where you want all filters to remain active. L1 would zero out many weights entirely, which is great for feature selection in linear models but harmful for a CNN that needs all its filters.

### Early stopping

> **🪜 Mental model:** *Quit while you're ahead.* Watch the val loss; if it stops improving for a while, stop training and restore the best weights.

**What it is.** **Early stopping** is a Keras callback that monitors a chosen metric (almost always `val_loss`) during training. If the metric hasn't improved for `patience` consecutive epochs by at least `min_delta`, training halts. The best-weights-so-far are typically restored via `restore_best_weights=True`. It's the cheapest, simplest regulariser — it just stops training when overfitting begins. This notebook uses `EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)`.

**Why it matters.** Without it, you have to manually guess how many epochs to train; the network often starts overfitting after some sweet spot and your final weights are worse than weights you had three epochs ago. With early stopping you set a generous epoch budget (this notebook uses 100) and the callback picks the right time to stop. Almost free, and prevents the worst case.

**How it works.** After each epoch, the callback compares the new validation metric to the best so far. If better (by at least `min_delta=0.001`), update the best and reset the patience counter. If worse, increment the patience counter; when it reaches `patience=10`, halt and restore the saved-best weights. `min_delta=0.001` means "an improvement of less than 0.001 in val_loss doesn't count" — useful to ignore tiny stochastic wiggles.

**Where it's used.**
- **Cell 4** of this notebook, alongside `ReduceLROnPlateau`.
- Every Keras training loop. Coupled with `ReduceLROnPlateau` and `ModelCheckpoint` it's the standard callback trio.

**Related terms.**
- **ModelCheckpoint** — saves model weights when val improves; partial substitute for `restore_best_weights`.
- **ReduceLROnPlateau** — reduces LR instead of halting; the sibling callback (next entry).
- **`patience`** — how many bad epochs to tolerate before stopping.
- **`min_delta`** — the minimum improvement that counts as "better".
- **`restore_best_weights=True`** — restores the weights from the epoch with the best val metric (without this, you keep the *last* weights, which are typically worse).

```python
keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10,
    min_delta=0.001, restore_best_weights=True,
)
```

**Gotcha.** Without `restore_best_weights=True`, you keep the *final* (worse) weights instead of the best ones — defeats the purpose. **Always set this to `True`.** Also: the patience hierarchy matters — set `EarlyStopping(patience=10)` *larger* than `ReduceLROnPlateau(patience=5)` so the LR drop gets a chance to rescue training before early stop pulls the plug.

### Learning-rate schedules (ReduceLROnPlateau)

> **🪜 Mental model:** *Big steps early, baby steps later.* Start the optimiser with a large learning rate to explore, then shrink it for fine-grained convergence.

**What it is.** A **learning-rate (LR) schedule** changes the optimiser's learning rate over the course of training instead of keeping it fixed. Common Keras schedules:
- **`ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5)`** — multiply LR by `factor=0.3` (i.e., shrink to 30%) when val_loss stalls for `patience=5` epochs. This is the one this notebook uses.
- **Step decay** — LR ×= 0.1 every 30 epochs.
- **Cosine annealing** — smooth cosine-shaped decrease over the whole training run.
- **Warmup** — start with a small LR and ramp up linearly for the first few hundred steps; common for large batch sizes and transformers.

**Why it matters.** A fixed LR is a compromise — too small ⇒ slow training, too large ⇒ unstable. Schedules give you both: aggressive early progress and precise late convergence. In this notebook, the LR drop kicks in when val_loss plateaus, often rescuing training and squeezing out another epoch or two of improvement before EarlyStopping eventually fires.

**How it works.** A callback queries the current epoch's val metric. If it hasn't improved for `patience` epochs, multiply the current LR by `factor` and continue. The optimiser uses the new LR for its next update step. The LR change is logged so you can see it on the diagnostic plot.

**Where it's used.**
- **Cell 4** of this notebook, in the callback list.
- Standard 3-callback combo: `EarlyStopping + ReduceLROnPlateau + ModelCheckpoint`.

**Related terms.**
- **Warmup** — gradually increase LR at the start to stabilise.
- **OneCycleLR** — cyclic schedule popular with super-convergence.
- **Optimizer** — the algorithm (SGD, Adam) that uses the LR.
- **`factor`** — how much to multiply the LR by (e.g., 0.3 = shrink to 30%).
- **`patience`** — how many stagnant epochs to wait before dropping.

```python
# Notebook usage — coupled with EarlyStopping
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                  min_delta=0.001, restore_best_weights=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

**Gotcha.** A too-aggressive schedule (factor 0.1) early in training can stall the model in a bad local minimum. Start with `factor=0.3, patience=5` and adjust. Also: **the patience hierarchy is deliberate** — LR-reduce fires at patience 5, early-stop fires at patience 10. This gives the LR-drop a chance to rescue training before early-stop halts.

### GlobalAveragePooling2D (architecture trick from iteration 1)

> **🪜 Mental model:** *One number per filter, no Flatten explosion.* Collapse each `(H, W)` feature map to a single average value — the classifier sees `Cout` numbers, not `H · W · Cout`.

**What it is.** **GlobalAveragePooling2D** (GAP) is a Keras layer that averages each feature map down to a single scalar. Input shape `(B, H, W, Cout) → (B, Cout)`. It has zero parameters. In this notebook, GAP replaces the `Flatten + Dense(huge)` pattern used in Module 1 — the iteration-1 model ends with `GlobalAveragePooling2D → Dense(256) → Dense(10)` instead of `Flatten → Dense(256, 17M params) → Dense(10)`.

**Why it matters.** Replacing `Flatten + Dense(256)` after a `(16, 16, 256)` feature map with `GAP + Dense(256)` cuts the head's parameter count from `16 · 16 · 256 · 256 ≈ 17M` to `256 · 256 ≈ 65k` — a 250× reduction. The model has way less capacity to memorise the training set, which is itself a regulariser. GAP is one of two reasons iteration 1 closes the gap so dramatically (the other being the deeper conv stack).

**How it works.** For each of the `Cout` feature maps, compute the mean of all `H · W` values. The output is a length-`Cout` vector per image. This forces each feature map to be interpretable as "how much of feature *k* is present in this image, on average" — a useful inductive bias for classification.

**Where it's used.**
- **Cell 1** of this notebook: `layers.GlobalAveragePooling2D()` at the end of the conv stack.
- Modern CNNs (ResNet, EfficientNet, MobileNet) all use GAP at the end instead of Flatten.
- Class Activation Mapping (CAM) techniques rely on GAP for interpretability.

**Related terms.**
- **Flatten** — the old approach that produces a huge vector.
- **Global Max Pooling** — sibling; takes the max instead of the mean.
- **Class Activation Mapping (CAM)** — visualisation technique enabled by GAP.

```python
# Notebook usage — replaces Flatten + Dense(huge)
x = layers.GlobalAveragePooling2D()(x)     # (B, H, W, 256) → (B, 256)
x = layers.Dense(256, activation='relu')(x)
out = layers.Dense(10, activation='softmax')(x)
```

**Gotcha.** GAP is a much harder bottleneck than Flatten — your earlier conv layers must produce informative feature maps because the classifier only sees a single mean per channel. If accuracy drops after switching from Flatten to GAP, you probably need more filters in the last conv block.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Deeper conv stack with GAP (Iteration 1)
```python
model = keras.Sequential([
    layers.Conv2D(16,  3, padding='same', activation='relu', input_shape=(128, 128, 3)),
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
Five Conv blocks with increasing filters (16 → 256). `GlobalAveragePooling2D()` replaces `Flatten`, slashing the final-dense params.

### 2. Add BatchNorm + Dropout (Iteration 2)
```python
def block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)         # after activation
    x = layers.MaxPooling2D()(x)
    return x

# Dense head — Dropout placed here, not in conv blocks
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
Apply `l2(1e-3)` to *every* Conv and Dense `kernel_regularizer`. Adds `λ · Σ w²` to the loss — gentle, smooth shrinkage.

### 4. Callbacks: EarlyStopping + ReduceLROnPlateau
```python
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=5,
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
    layers.Rescaling(1.0 / 255),
])

# Train-only augmentation
train_ds = train_data.map(
    lambda x, y: (augment(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Val/test get only the rescaling — no random transforms!
val_ds = val_data.map(lambda x, y: (layers.Rescaling(1./255)(x), y))
```

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `layers.BatchNormalization()` | Per-batch normalisation with learnable γ, β |
| `layers.Dropout(rate)` | Stochastic activation zeroing, train-only |
| `regularizers.l1(λ)` / `l2(λ)` | Weight-decay penalties as `kernel_regularizer=` |
| `layers.RandomFlip / Rotation / Crop / Brightness / Contrast / Translation` | Built-in augmentation layers |
| `layers.GlobalAveragePooling2D()` | Replaces `Flatten + huge Dense` |
| `keras.callbacks.EarlyStopping(monitor, patience, min_delta, restore_best_weights)` | Auto-halt + restore best |
| `keras.callbacks.ReduceLROnPlateau(monitor, factor, patience)` | LR decay on plateau |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **Augmentation must be train-only.** Apply to val/test and your metrics become noise (see the dedicated walkthrough entry above).
2. **`BatchNormalization` at inference uses running stats** — call `model(x, training=False)` explicitly if needed. Forgetting this in transfer learning (Module 3) is a major source of accuracy gaps.
3. **`Dropout` is disabled at inference** automatically. Don't manually scale activations.
4. **`EarlyStopping(restore_best_weights=True)` is the safe default** — without it, you get the last-epoch weights (often worse than the peak).
5. **`ReduceLROnPlateau` interacts with `EarlyStopping`** — give LR-reduce a smaller patience (5) than early-stop patience (10), so LR gets to act before training stops.
6. **L2 = `λ · Σ w²`, NOT just `Σ w²`.** `kernel_regularizer=l2(1e-3)` — the `λ` is `1e-3`.
7. **Don't `Dropout` inside Conv blocks at 0.5** — that's too aggressive for spatial features. Use `0.1–0.25` there; reserve `0.5` for dense heads.
8. **Pick L2 over L1 for CNNs.** L1 zeros out many weights entirely — fine for feature selection in linear models, harmful for a CNN that needs all its filters active.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

> ≥ 10 questions, ≥ 5 sourced and cited from canonical interview banks. Every concept in the 📖 walkthrough section above has at least one Q&A item below.

**Q1. Why does Batch Normalization help?** *(notebook quiz cell, original)*
→ Normalises layer inputs per minibatch (zero mean, unit variance per channel) → stabilises training → allows larger LRs without divergence. Also provides a mild regularising effect via batch-noise.

**Q2. What is dropout, and why is it required?** *(notebook quiz cell, original)*
→ Stochastically zero a fraction `p` of activations during training → reduces co-adaptation between neurons → behaves like an implicit ensemble of sub-networks at inference. Cheapest closer of a train/val gap in fully-connected heads.

**Q3. How does BatchNorm behave at inference?** *(notebook quiz cell, original)*
→ Uses **running mean / variance** (EMA computed during training), not the current batch. This makes inference deterministic.

**Q4. Where do you place Dropout vs BatchNorm?** *(notebook quiz cell, original)*
→ **Dropout:** between Dense layers (0.5), or post-MaxPool (0.1–0.25) in conv blocks. **BatchNorm:** after Conv (some teams put it before the activation, some after — pick one and be consistent).

**Q5. L1 vs L2 — practical difference?** *(notebook quiz cell, original)*
→ **L1** produces *sparse* weights (drives many to exactly 0 → feature selection). **L2** produces smooth shrinkage (every weight gets a little smaller). Default in CNNs is L2.

**Q6. Your model has 99% train accuracy and 60% val accuracy. What three things do you try, in order?** *(adapted from `chiphuyen/ml-interviews-book`, ML systems chapter)*
→ (1) **More data / augmentation** — cheapest, most effective; flip, rotate, crop your existing training images. (2) **More regularisation** — `Dropout(0.5)` to the dense head, BN after each conv, `kernel_regularizer=l2(1e-3)` on the convs. (3) **Smaller model or `EarlyStopping`** — reduce capacity or stop at the epoch where val_loss bottomed. Try them in that order because augmentation has the best ROI.

**Q7. Why is BatchNorm sometimes called a regulariser?** *(adapted from `andrewekhalel/MLQuestions`)*
→ Because BN computes its mean/variance **per minibatch** during training, the statistics are slightly noisy (different from batch to batch). This noise acts like a mild data augmentation injected into the activations themselves — the model can't rely on exact activation values, which discourages overfitting. It's not as strong as Dropout, but it's "for free" once you've added BN for its stabilisation effect.

**Q8. If you must pick exactly one regulariser for a CNN trained on 5,000 images, which one and why?** *(common FAANG question)*
→ **Data augmentation.** It's the only technique that effectively *creates new data*; Dropout/BN/L2 only reshape how existing data is used. With ~5,000 images the bottleneck is data, not model capacity — augmentation hits the bottleneck directly. The next best single technique is `EarlyStopping(restore_best_weights=True)` because it's free and prevents the worst case.

**Q9. Why does Dropout *hurt* a model that's already underfitting?** *(adapted from `alexeygrigorev/data-science-interviews`)*
→ Dropout works by reducing effective capacity — randomly zeroing out neurons. An underfit model is already capacity-starved; reducing capacity further makes it worse. The diagnostic: if **training accuracy** is low, don't add regularisation — make the model bigger or train longer instead.

**Q10. Why should augmentation be applied only to the training set?** *(adapted from `andrewekhalel/MLQuestions`)*
→ Two reasons: (a) **non-determinism** — random augmentation makes your val/test accuracy different on every run, so reported numbers stop being repeatable; (b) **inflated metrics** — if augmentation accidentally makes images "easier" (e.g., random crops centre objects), test accuracy is optimistic vs deployment. Val/test must reflect deployment conditions exactly.

**Q11. Why does `restore_best_weights=True` matter for `EarlyStopping`?** *(adapted from `Sroy20/machine-learning-interview-questions`)*
→ Without it, training halts at epoch *k+patience*, leaving you with the *current* (worse) weights — even though the *best* weights were back at epoch *k*. `restore_best_weights=True` rewinds to that best epoch. It's the whole point of EarlyStopping; without it, you might as well not use the callback.

**Q12. In this notebook, why does iteration 1's deeper architecture *also* close the gap (not just iteration 2's regularisers)?** *(notebook-specific, original)*
→ Two things happened in iteration 1: (a) the conv stack got deeper (5 blocks, 16 → 256 filters), giving better hierarchical features; (b) `GlobalAveragePooling2D` replaced `Flatten + Dense(huge)`, cutting parameter count by 250×. Less capacity to memorise + better features = smaller gap, even before any explicit regulariser is added. Architecture *is* regularisation.

**Q13. What's the patience hierarchy this notebook uses for the two callbacks, and why is it ordered that way?** *(notebook-specific, original)*
→ `ReduceLROnPlateau(patience=5)` and `EarlyStopping(patience=10)`. LR-reduce fires *first* (at 5 stagnant epochs) → giving the smaller LR a chance to rescue training. If after another 5 stagnant epochs val_loss still hasn't budged, EarlyStopping pulls the plug. Reversing the order would cause early-stop to fire before the LR drop gets a chance.

[🔝 Back to top](#top)

## 🪞 Extra ladder — diagnosing train vs val gap

**Basic** — if train acc ≫ val acc → **overfitting** → add regularisation (Dropout, BN, augmentation, L2, EarlyStopping).

**Intermediate** — if both train and val are low → **underfitting** → bigger model or longer training. *Do not* add regularisation; you'll make it worse.

**Advanced** — flat train AND flat val AND non-zero gap → **data leak, label noise, or a frozen layer you forgot to unfreeze**. Look at sample predictions (not just numbers) — if confidently wrong on obvious examples, it's a data issue, not a capacity issue.

[🔝 Back to top](#top)

## What comes next

This notebook gets you from 51% → 78% test acc *training from scratch* on Clothing-Small. [Notebook 3 →](../3.Transfer%20learning%201/) hits 79% on a 10-class landmarks dataset with **only 737 training images** by using a pretrained backbone (transfer learning).

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
