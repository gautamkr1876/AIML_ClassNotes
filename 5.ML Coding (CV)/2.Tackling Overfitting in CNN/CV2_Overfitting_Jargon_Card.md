<a id="top"></a>
# CV2 — Tackling Overfitting Jargon Card

> **Use this file like a dictionary.** Skim it once (~7 min) before opening the notebook or the deep-dive. Then keep it open in a side tab — when you hit an unknown word, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`CV2_Overfitting_Reading_Brief.md`](./CV2_Overfitting_Reading_Brief.md) FIRST. This card is just the dictionary.
> **Source notebook:** `Tackling_Overfitting_in_CNN.ipynb` in this folder.
> **Deep dive (slow read, 40 min):** [`CV2_Tackling_Overfitting_Interview_Prep_Guide.md`](./CV2_Tackling_Overfitting_Interview_Prep_Guide.md).

---

## A

**Adam** — The default optimiser in modern deep learning. Adapts the learning rate per parameter. The notebook uses Adam with `lr=1e-3` by default, then lets `ReduceLROnPlateau` cut it down when val_loss plateaus.

**Augmentation (data augmentation)** — A regulariser that **invents new training images on the fly** by randomly flipping, rotating, cropping, or jittering the colour of each batch. Same content, varied appearance — the network sees the cat from many angles instead of one. **Train-only**; never apply to val/test.

## B

**Batch normalization (BN)** — A layer that **standardises activations per channel** within each training batch (mean 0, variance 1) and then applies a learnable affine `γ·x + β`. Stabilises training, allows higher learning rates, has a mild regularising effect. Placed after every Conv (and after the hidden Dense) in this notebook.

**Bernoulli mask** — A random 0/1 array drawn from a Bernoulli distribution. Dropout uses one to decide which activations to zero out at each step.

**Bias-variance tradeoff** — The formal framing for the underfit/overfit balance. High bias = model too simple (underfit). High variance = model memorises noise (overfit). Regularisation tries to reduce variance without inflating bias.

## C

**Callback** — A hook that runs at specific points during training (epoch end, batch end, etc.). Examples used here: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`. Passed as a list to `model.fit(callbacks=[...])`.

**Cross-validation (k-fold)** — Splitting data into `k` folds and rotating which one is the validation set. Used when data is too scarce for a stable single train/val/test split. Not used in this notebook (the data is folder-pre-split).

## D

**Data leakage** — When information from the val/test set sneaks into training. Common culprits: computing normalisation stats over all data instead of just train; using the same person/identity in both train and val. Folder-based splits in this notebook prevent the easy mistakes.

**Diagnostic plot** — Plotting `loss` and `val_loss` on the same axes vs epoch. The visual telltale of overfitting: train keeps falling while val flattens or rises. The single most useful diagnostic in deep learning.

**Dropout** — A regularisation layer that, **only during training**, randomly zeros out a fraction `p` of the previous layer's activations. At inference it's a no-op. Standard rates: `0.5` after Dense layers (what this notebook uses); `0.1–0.25` after conv layers. Acts like an implicit ensemble of sub-networks.

## E

**EarlyStopping** — A callback that monitors a metric (usually `val_loss`) and **halts training when it stops improving for `patience` epochs**. With `restore_best_weights=True`, it rolls back to the epoch where val_loss was lowest. Cheap insurance against wasted compute.

**Ensemble (implicit ensemble)** — What dropout produces at inference. Each training forward pass uses a different random subset of neurons; at inference all neurons fire, and the result behaves like an average over the many sub-networks seen during training.

**Epoch** — One full pass through the training set. The notebook trains for up to 100 epochs but early-stops around 30–40 once val_loss plateaus.

## F

**Flatten** — Reshapes a `(H, W, C)` feature map into a 1D vector of length `H·W·C` before the Dense head. Produces a huge number of parameters. **Replaced by GlobalAveragePooling2D in this notebook** to slash params.

## G

**Generalisation** — The goal: low loss on data the model has **never seen**. Overfitting is the failure mode where training-set loss is great but generalisation is bad.

**GlobalAveragePooling2D (GAP)** — Replaces `Flatten + huge Dense` with a per-channel average over the spatial dimensions. A `(H, W, 128)` feature map becomes a length-128 vector. Massively cuts parameter count, acts as a structural regulariser, and is invariant to small spatial shifts.

**`γ`, `β` (BatchNorm parameters)** — Learnable per-channel scale (`γ`) and shift (`β`) that BN applies *after* normalising. They let the network decide it doesn't want pure zero-mean / unit-variance — it can undo the normalisation if the optimiser finds it useful.

## H

**`history.history`** — The dict returned by `model.fit()` with per-epoch metrics: `loss`, `accuracy`, `val_loss`, `val_accuracy`. Plotting these is how you produce the diagnostic plot.

**Hyperparameter** — A configuration setting not learned by gradient descent: learning rate, dropout rate, batch size, L2 strength, augmentation strength. Tuned on the **validation** set; never on the test set.

## I

**`image_dataset_from_directory`** — The Keras helper that loads images from a folder structure (`path/class_a/img.jpg`, `path/class_b/img.jpg`) into a `tf.data.Dataset`. Used three times in the notebook — one per train/val/test split.

**Inverted dropout** — The default dropout scheme used by Keras: during training, surviving neurons are multiplied by `1/(1-p)` so the expected magnitude matches inference (no rescaling needed at eval).

## K

**Kernel regularizer** — The `kernel_regularizer=l2(1e-3)` argument on Conv2D/Dense layers. Tells the layer to add `λ · Σ w²` to the loss, so the optimiser keeps weights small. Different from `bias_regularizer` (which usually you leave alone).

## L

**L1 regularization** — Adds `λ · Σ |w|` to the loss. Encourages **sparse** weights (many exact zeros). Used for feature selection.

**L2 regularization (a.k.a. weight decay)** — Adds `λ · Σ w²` to the loss. Encourages **small but non-zero** weights — a "tax on being big." The notebook uses `l2(1e-3)` on every Conv and Dense (iteration 3) and gets ~2 points of accuracy from it alone.

**Learning rate (LR)** — How big a step gradient descent takes per update. Too high → diverges; too low → trains forever. Adam's default is `1e-3`.

**Learning rate schedule** — A rule that **changes the LR during training**. Common pattern: cosine decay, step decay, or — used here — `ReduceLROnPlateau`.

## M

**Monitor (in a callback)** — The metric the callback watches. `EarlyStopping(monitor='val_loss')` halts based on val_loss. Always use a *validation* metric — never train metrics, which would defeat the purpose.

## O

**Overfitting** — The model fits training-data noise and quirks rather than the underlying pattern. Symptom: training accuracy keeps climbing while validation accuracy plateaus or drops. The 40-point train/val gap in this notebook's baseline is textbook overfitting.

## P

**Patience (in callbacks)** — How many epochs without improvement before the callback fires. `EarlyStopping(patience=5)` waits 5 stagnant epochs before stopping. `ReduceLROnPlateau(patience=3)` waits 3 before cutting the LR.

## R

**ReduceLROnPlateau** — A callback that **cuts the learning rate** (by `factor`, e.g., `0.5`) when val_loss stops improving for `patience` epochs. Lets training continue in fine-tuning mode after the initial big-step phase plateaus.

**Regularisation** — The umbrella for any technique that limits a model's capacity to memorise: Dropout, BN, L1/L2, augmentation, early stopping, smaller architectures. Everything in this notebook is a flavour of regularisation.

**Rescaling** — Normalising pixel values from `0–255` to `0–1` (divide by 255). Done in preprocessing so the network trains stably. Apply the same rescaling to val/test (no other augmentation).

**`restore_best_weights=True`** — An `EarlyStopping` flag. Without it, training stops at the worst recent epoch. With it, the model rolls back to whichever epoch had the best `val_loss`. **Always set this to True.**

**Running mean / running variance (BN at inference)** — Exponential moving averages of the per-batch statistics, accumulated during training. At inference time, BN uses these fixed running stats instead of the current batch's stats — so a single test image produces the same output every time.

## S

**Skip connection** — Not used in this notebook (Notebook 1 had basic CNNs; ResNet-style skips come in CV3 Transfer Learning).

**`SparseCategoricalCrossentropy`** — The loss function for integer class labels (e.g., `[0, 3, 2, ...]` for 10 classes). Use this when labels are integers; use `CategoricalCrossentropy` for one-hot vectors. The notebook uses sparse.

## T

**Test set** — The sealed final set, opened **once** at the very end to report unbiased generalisation. If you tune anything on the test set, the number is contaminated. The notebook's test acc of **78.0%** in iteration 4 is the headline.

**Train/val gap** — The difference between training and validation accuracy (or loss) at a given epoch. The diagnostic signal: a big gap means overfitting; a small gap means either healthy training or underfitting (check absolute numbers too).

**Train-only augmentation (the trap)** — Augmentation layers (`RandomFlip`, `RandomRotation`, etc.) **must be applied only to the training pipeline**. If you augment val/test, the val_loss becomes noisy and you can't tell whether the model improved or just got a lucky augmentation. The notebook makes a separate `data_augmentation` Sequential that's only attached to the train pipeline.

## U

**Underfitting** — The opposite of overfitting: both train and val losses are bad. Means the model is too small or hasn't trained enough. The fix is **more** capacity (deeper net, more epochs) — not more regularisation.

## V

**Validation set** — The held-out set used **during training** to monitor for overfitting and pick hyperparameters. The training loop never updates weights on this — it just evaluates each epoch to produce `val_loss` and `val_accuracy`. Used by `EarlyStopping` and `ReduceLROnPlateau`.

## W

**Weight decay** — Synonym for L2 regularisation in the Keras API. Adds `λ · Σ w²` to the loss; pushes weights toward zero.

---

[🔝 Back to top](#top)
