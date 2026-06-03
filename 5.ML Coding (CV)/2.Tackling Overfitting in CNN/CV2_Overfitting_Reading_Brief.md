<a id="top"></a>
# CV2 — Tackling Overfitting in CNNs — Reading Brief

> **Read this ONCE, end to end, before opening the notebook or the deep-dive.** Target time: ~22 minutes. By the time you reach the longer material, every word will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`CV2_Overfitting_Jargon_Card.md`](./CV2_Overfitting_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebook:** `Tackling_Overfitting_in_CNN.ipynb` in this folder.
> **The deep dive (slow read, ~40 min):** [`CV2_Tackling_Overfitting_Interview_Prep_Guide.md`](./CV2_Tackling_Overfitting_Interview_Prep_Guide.md).

---

## 🎯 30-second TL;DR

**The baseline CNN from Notebook 1 had a 40-point train/val gap — textbook overfitting (99.9% train, 59.8% val, 51.6% test).** This notebook applies the regularisation toolkit **one tool at a time**, tracks accuracy + gap at each step, and ends at:

> **78.0% test accuracy, gap closed to ~7%.** Same data, same dataset. The difference is **regularisation discipline.**

The five iterations (this is the punchline you should remember):

| Iteration | Added | Train | Val | Test | Gap |
|---|---|---|---|---|---|
| 0. Baseline (single Conv) | — | 99.9% | 59.8% | **51.6%** | **40.1%** |
| 1. + Deeper conv stack + GAP | Architecture | 66.9% | 59.2% | 68.6% | 7.6% |
| 2. + Dropout + BatchNorm | Regularisation | ~88% | 67.7% | 75.3% | 20.4% |
| 3. + L2 weight decay + LR schedule + EarlyStopping | Weight decay | ~85% | ~78% | 77.4% | ~7% |
| 4. + Data augmentation | Augmentation | ~87% | 80%+ | **78.0%** | ~7% |

**That table is the whole notebook.** Everything else is *why* each row works and *how* to wire it up in Keras.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **The diagnostic** — plotting `loss` and `val_loss` per epoch; the train/val gap is the signal.
2. **Three drawers** — train / validation / test split, and why test is sealed until the very end.
3. **Iteration 1: architectural fix** — deeper conv blocks (3 conv stacks instead of 1) + `GlobalAveragePooling2D` replacing `Flatten + huge Dense`. This alone drops 90% of the parameters and shrinks the gap.
4. **Iteration 2: dropout + batch normalisation** — `Dropout(0.5)` in the dense head; `BatchNormalization` after every Conv. The two most effective per-layer regularisers, applied together.
5. **Iteration 3: L2 weight decay + LR schedule + EarlyStopping** — `kernel_regularizer=l2(1e-3)` on every Conv and Dense; `ReduceLROnPlateau(factor=0.5, patience=3)`; `EarlyStopping(patience=5, restore_best_weights=True)`.
6. **Iteration 4: data augmentation** — `RandomFlip`, `RandomRotation`, `RandomZoom` *as part of the training pipeline only* (the train-only trap). Final ~3-point bump.
7. **The "augmentation must be train-only" trap** — what goes wrong if you augment val/test.
8. **Putting it together** — final architecture, final hyperparameters, final 78.0% test acc.

---

## 🧠 The big idea

> **Overfitting = memorising the textbook, failing the exam.**

A model with enough capacity will eventually memorise every quirk of the training set — including the noise. Memorisation looks great on training loss but kills generalisation. The diagnostic is simple: **plot train_loss and val_loss together; if they diverge, you're overfitting.**

The fix is **regularisation** — any technique that *limits the model's ability to memorise*. The notebook walks the canonical ladder:

1. **Architectural regularisation** — smaller / smarter architecture (deeper conv + GAP instead of huge Flatten). Reduces param count → less to memorise with.
2. **Per-layer regularisation** — Dropout (random "knock out" neurons), BatchNorm (re-centre activations + mild noise).
3. **Loss regularisation** — add a "weights-are-large" tax (L1/L2 / weight decay).
4. **Training-loop regularisation** — quit while you're ahead (EarlyStopping); slow down as you approach the optimum (ReduceLROnPlateau).
5. **Data regularisation** — invent new training examples (augmentation).

You almost never need *all five* — but knowing which one to reach for is the FAANG-interview question for this topic.

---

## 📖 Core concept primers

### 1. Overfitting & the train/val gap

> **🪜 Mental model:** *two curves diverging.* Train loss keeps falling; val loss bottoms out and rises. The exact crossing point is when overfitting kicks in.

**What it is.** A model overfits when training accuracy keeps climbing while validation accuracy stalls or drops — the model is fitting noise instead of pattern. The numeric signal is the **train/val gap** at the end of training: 1–5 points is healthy; 20+ is bad.

**How to diagnose.** Capture the dict returned by `fit()`:

```python
history = model.fit(train_ds, validation_data=val_ds, epochs=N)
plt.plot(history.history['loss'],     label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss')
```

**Three patterns to recognise on this plot:**
- **Healthy:** both curves fall together, plateau, gap stays small.
- **Overfitting:** train keeps dropping, val plateaus then *rises*.
- **Underfitting:** both flat and high. Doesn't matter how much you regularise — you need a **bigger** model, not more regularisation.

**Why it matters.** This is the canonical FAANG opener: *"Your model has 99% train and 60% val accuracy. What do you do?"* The answer is the rest of this brief.

### 2. Dropout

> **🪜 Mental model:** *random sparring partner.* Each training step, knock out a random subset of neurons. The network can't rely on any single one.

**What it is.** A layer that, **only during training**, randomly sets a fraction `p` of the previous layer's outputs to zero. At inference it's a no-op. Keras handles the rescaling for you (inverted dropout — surviving neurons are multiplied by `1/(1-p)` during training so the expected magnitude matches at eval time).

**How to use in this notebook.**

```python
# Placed only in the dense head, NOT inside conv blocks
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)        # ← 50% drop rate
out = layers.Dense(10, activation='softmax')(x)
```

**Standard rates:**
- After **Dense** layers → `Dropout(0.5)` (what the notebook uses).
- After **Conv** layers → `Dropout(0.1–0.25)`. **Never 0.5 on conv** — that nukes too much of the feature map at once.

**Why it works.** Each forward pass uses a different random sub-network. At inference all neurons fire, so the prediction is effectively averaged over the many sub-networks seen during training — a free **implicit ensemble**.

### 3. BatchNormalization

> **🪜 Mental model:** *re-centre the assembly line.* For each batch, shift and scale each channel to look "standard normal" before passing to the next layer.

**What it is.** A layer that **normalises activations per channel** within each training batch (mean 0, variance 1), then applies a learnable affine `γ·x + β`. The `γ` and `β` are trained like any other weights. At inference, BN uses a *running average* of the mean/variance accumulated during training — so a single test image gives a stable output.

**How it's wired in the notebook.**

```python
def block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)             # ← after every Conv
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)             # ← and again
    x = layers.Activation('relu')(x)
    return layers.MaxPooling2D()(x)
```

**Why it works.**
- Stabilises training → you can use a higher learning rate.
- The batch-level noise in its statistics is a mild regulariser (like a tiny dropout).
- Modern CNNs train **2–10× faster** with BN than without.

**Gotcha.** BN behaves differently at train and inference — at inference it uses the running stats, not the current batch's stats. If you accidentally evaluate in training mode, your eval is noisy.

### 4. L2 weight decay (regularisation)

> **🪜 Mental model:** *a tax on being big.* The loss now penalises large weights, so the optimiser keeps them small.

**What it is.** Add a term to the loss: `total_loss = data_loss + λ · Σ w²`. Big weights cost more → the optimiser shrinks them. `λ` is the regularisation strength; this notebook uses `1e-3`.

**How to use in Keras.**

```python
from tensorflow.keras.regularizers import l2

layers.Conv2D(64, 3, kernel_regularizer=l2(1e-3))    # every Conv
layers.Dense(256, kernel_regularizer=l2(1e-3))       # every Dense
```

**The two flavours:**
- **L1** (`λ · Σ |w|`) — encourages **exactly zero** weights (sparsity). Used for feature selection.
- **L2** (`λ · Σ w²`) — encourages **small but non-zero** weights (smooth shrinkage). The standard regulariser for deep learning. Synonym: **weight decay**.

**Why it matters in this notebook.** Iteration 3 adds `l2(1e-3)` to every layer and the gap drops from 20% to ~7%. Most of the third accuracy bump comes from this.

### 5. Data augmentation (train-only)

> **🪜 Mental model:** *virtual data multiplication.* Same content, varied appearance. Flip, rotate, crop, jitter the same image so the network sees it as many different images.

**What it is.** A preprocessing pipeline that applies **random transforms** to each training image *every epoch*. Same image, different augmentations each time — effectively infinite training data.

**The notebook's augmentation block:**

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),       # ± ~10°
    layers.RandomZoom(0.1),           # ± 10%
])

# Apply ONLY to the training pipeline
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
# val_ds and test_ds get NO augmentation — only the rescaling
```

**The trap (the most-asked CV interview question).** Augmentation **must be train-only**. If you augment val/test:
- Val_loss becomes noisy — you can't tell whether the model actually improved.
- Test accuracy is no longer comparable to anyone else's number.
- `EarlyStopping` makes wrong decisions because its signal is noisy.

**Why it works.** Augmentation regularises by making it impossible for the model to memorise pixel-level details. The shape "cat" must hold up under flip + 10° rotation, so the model learns the *shape* not the *pixels*.

### 6. EarlyStopping + ReduceLROnPlateau (training callbacks)

> **🪜 Mental model:** **quit while you're ahead**, and **slow down as you approach the optimum**.

**EarlyStopping.** Monitors a metric (almost always `val_loss`). If it doesn't improve for `patience` epochs, halt training. With `restore_best_weights=True`, roll back to the best epoch.

```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

**ReduceLROnPlateau.** When val_loss plateaus, **cut the learning rate** so the optimiser can make finer adjustments.

```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
# When val_loss plateaus for 3 epochs → halve the LR.
```

**How they pair.** ReduceLROnPlateau fires first (cut LR), giving the model a few more epochs to squeeze accuracy. If even at the lower LR val_loss still doesn't improve, EarlyStopping eventually halts. In the notebook, max epochs is 100, but training typically stops around epoch 30–40.

**Always set `restore_best_weights=True`.** Without it, training halts at the worst recent epoch (not the best). This is the #1 forgotten-flag bug in Keras training scripts.

### 7. GlobalAveragePooling2D — the architectural regulariser

> **🪜 Mental model:** *one number per channel.* Throw away `H × W`; keep one average value per feature map.

**What it is.** Replaces `Flatten + Dense(huge)` at the end of a CNN. A `(H, W, C)` feature map collapses to a length-`C` vector by averaging over the spatial dims.

```python
# OLD (Notebook 1): Flatten produces H·W·C → Dense weights = (H·W·C) × hidden
x = layers.Flatten()(x)                          # length 16,384 for (32, 32, 16)
x = layers.Dense(256)(x)                         # 4.2M params just here!

# NEW (this notebook): GlobalAveragePooling2D produces just C
x = layers.GlobalAveragePooling2D()(x)           # length 128 for (8, 8, 128)
x = layers.Dense(256)(x)                         # 33K params
```

**Why it matters.** Two things at once:
- **Param reduction.** ~90% fewer parameters in the classifier head → less to memorise with → less overfitting.
- **Translation invariance.** Doesn't matter *where* in the image the feature appeared — just *whether* it appeared. Useful for classification (less so for localisation).

**This single change drives Iteration 1's huge improvement** (51.6% → 68.6% test acc, gap from 40 to 7.6).

---

## 🔥 The five-iteration ladder — at a glance

| If you have... | Reach for... | Because... |
|---|---|---|
| **Big train/val gap, huge Flatten layer** | Architectural fix: deeper conv + GAP | Fewer params = less to memorise |
| **Big gap, Dense head is the suspect** | `Dropout(0.5)` after Dense | Implicit ensemble; the standard fix |
| **Training unstable, can't crank LR** | `BatchNormalization` after every Conv | Stabilises activations |
| **Mild but persistent overfitting** | `l2(1e-3)` on every Conv and Dense | Smooth weight shrinkage |
| **Training too long / wastes compute** | `EarlyStopping(patience=5, restore_best_weights=True)` | Quit while ahead |
| **Loss plateau before convergence** | `ReduceLROnPlateau(factor=0.5, patience=3)` | Fine-tune phase |
| **Out of training data, model still overfits** | `RandomFlip + RandomRotation + RandomZoom` (train-only!) | Virtual data; same content, varied appearance |

**Use in combination — the notebook stacks all of these** and reaches 78.0% test acc.

---

## 🧮 Key numbers & defaults

- **Dropout rate** — `0.5` after Dense; `0.1–0.25` after Conv. Never `0.5` after Conv.
- **L2 strength** — `1e-3` (this notebook). Range: `1e-5` (gentle) to `1e-2` (aggressive).
- **EarlyStopping patience** — `5` epochs (typical). Set `restore_best_weights=True`.
- **ReduceLROnPlateau** — `factor=0.5, patience=3, min_lr=1e-6`.
- **Augmentation strength** — `RandomRotation(0.1)` = ±10°. Don't go wild: rotating a digit by 90° makes "6" into "9".
- **Baseline metrics to remember** — start: 51.6% test / 40-point gap. End: 78.0% test / ~7-point gap.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–10** | Loading the 3 splits with `image_dataset_from_directory`; rescaling; visualising the baseline overfit | **Read normally** — ~5 min. Note the 40-point gap. |
| **11–25** | Iteration 1: deeper conv stack + GAP; param count comparison vs Notebook 1 | **FOCUS** — ~8 min. The architectural win. |
| **26–40** | Iteration 2: Dropout + BatchNorm; their placement (Dropout in dense head only; BN after every Conv) | **FOCUS** — ~8 min. The standard per-layer regulariser pair. |
| **41–55** | Iteration 3: `kernel_regularizer=l2(1e-3)` on every layer; EarlyStopping + ReduceLROnPlateau callbacks | **Read carefully** — ~6 min. Loss regularisation + training-loop callbacks. |
| **56–75** | Iteration 4: data augmentation Sequential, train-only pipeline wiring, the final 78% result | **FOCUS** — ~8 min. The trap section is critical. |
| **76–end** | Final model summary, side-by-side accuracy/loss curves across all 5 iterations | **Read** — ~3 min. The visual payoff. |

**Total target read time for the notebook itself:** ~38 min. Add this brief's ~22 min and you're at **~60 min** — vs. a cold read with jargon Googling closer to 2 hours.

---

## ✅ Walk-away checklist

After reading the notebook (or the deep-dive), you should be able to say in your own words:

- [ ] **The diagnostic for overfitting** — plot train_loss and val_loss together; diverging = overfitting; both flat + high = underfitting.
- [ ] **Why a small gap isn't always good** — could be underfitting; check absolute accuracy too.
- [ ] **Dropout's mental model and its 0.5-vs-0.1 rule** — 0.5 after Dense; 0.1–0.25 after Conv.
- [ ] **What BatchNorm does and where to put it** — normalises per-channel; placed after every Conv (and after hidden Dense).
- [ ] **L2 weight decay's role** — adds `λ·Σw²` to the loss; encourages small weights.
- [ ] **Why augmentation must be train-only** — augmenting val/test makes val_loss noisy and breaks EarlyStopping.
- [ ] **EarlyStopping + ReduceLROnPlateau** — quit while ahead + slow down at plateau. Always `restore_best_weights=True`.
- [ ] **Why GAP beats Flatten** — ~90% fewer params + translation invariance.
- [ ] **The 51.6% → 78.0% headline** — and which iteration delivered which chunk of that.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Your model has **99% train accuracy and 60% val accuracy**. Walk through the regularisation ladder you'd apply, in order, and name the *first* tool you'd reach for.
2. Your colleague applied `RandomFlip` and `RandomRotation` to *all three* splits (train, val, test). Their `val_loss` curve is wildly noisy. Why, and what's the single-line fix?
3. You set `EarlyStopping(monitor='val_loss', patience=5)`. Training stops at epoch 18. The model returned by `fit()` has test acc 70%, but you remember val_loss was minimum at epoch 13 with test acc 74%. What flag did you forget?
4. Why is `Dropout(0.5)` a bad idea **between conv layers**, while it's the *default* in the **dense head**?
5. The baseline model had test acc **51.6%** with a 40-point train/val gap. After replacing `Flatten + Dense` with `GlobalAveragePooling2D + Dense`, test acc jumps to **68.6%** with a 7.6-point gap. The model trained for **fewer** epochs. Why does *less capacity* lead to *higher* test accuracy?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **The ladder:** (1) Architectural regularisation — deeper conv + GAP; (2) Dropout + BatchNorm; (3) L2 weight decay; (4) EarlyStopping + ReduceLROnPlateau; (5) Data augmentation. **First tool: Dropout(0.5) in the dense head** — it's the cheapest, single-line fix that closes the largest fraction of the gap for the smallest amount of code change. (If you have a giant Flatten, fix that first via GAP.)

2. **They augmented the validation pipeline.** Each epoch the val_loss is computed on *differently-augmented* val images — so the val_loss isn't measuring "did the model improve," it's measuring "how lucky were today's augmentations." This also breaks `EarlyStopping`, which uses val_loss as its signal. **The fix:** apply `data_augmentation` only inside the train pipeline's `.map(...)` step; val and test get only the rescaling. One line.

3. **You forgot `restore_best_weights=True`.** Without it, EarlyStopping halts training but leaves the model at the *last* epoch's weights (epoch 18, the worst recent state). With it, the model rolls back to the epoch with the lowest val_loss (epoch 13, your 74% test acc). **Always set this flag.**

4. A `0.5` dropout on a conv feature map would zero out **half the spatial activations**, killing the spatial pattern the next conv layer needs to detect. Conv features are spatially correlated, so dropping half destroys structure. The Dense head has unstructured, abstract features — dropping half is fine and produces the implicit-ensemble benefit. **Rule: 0.1–0.25 after Conv; 0.5 after Dense.**

5. **Because the baseline's `Flatten + Dense(big)` head had millions of parameters with nothing to do but memorise.** GAP slashes ~90% of those parameters; with so much less capacity, the model *can't* memorise as well, so it has to find genuinely-generalising features instead. This is **architectural regularisation** — reducing capacity is itself a regulariser. Surprising but consistent finding: smaller-but-smarter architecture beats bigger-but-overfit architecture, on this dataset.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./CV2_Overfitting_Jargon_Card.md) · [→ Deep Dive](./CV2_Tackling_Overfitting_Interview_Prep_Guide.md)
