<a id="top"></a>
# CV Notebook 8 — Siamese Networks (Deep Dive)

> Per-notebook companion to the master guide. For the cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §8](../CV_Revision_Guide.md#8-module8). This deep dive is **standalone** — every concept below carries its own full Concept Definition Template entry (mental model + what + why + how + where + related + code + gotcha), substituted with the notebook's actual shapes, code, and dataset (BHSig260 signatures, ResNet-50 encoder, 128-D embeddings). You should never need to click through to the master to understand a term.

## What this notebook actually demonstrates

- **Task: signature verification** — given two signatures, decide whether they're from the same person.
- **Why not a classifier?** A classifier would need a fixed list of signers; adding a new person would force retraining with `K+1` classes. Siamese networks instead learn a **similarity function** — adding a new signer is just storing one new embedding.
- **Dataset:** **BHSig260** — 260 signers (160 Hindi + 100 Bengali) × ~54 signatures each (24 genuine + 30 forged). Total ~14,040 images, resized to `180 × 180 × 3`.
- **Backbone:** ResNet-50 (ImageNet-frozen) → `GlobalAveragePooling2D` → `Dense(128)` → 128-D embedding. Called twice in the same graph so the **same weights** apply to both inputs.
- **Two loss formulations implemented (and compared):**
  1. **Contrastive loss** — pair-based: `y·d² + (1-y)·max(0, margin − d)²`
  2. **Triplet loss** — triplet-based: `max(0, d(a,p) - d(a,n) + α)`
- **Inference:** sweep the threshold `t` on validation, then classify pairs as same when distance ≤ `t`.

## 🪜 Mental anchors for this notebook

- **Twin networks with one set of brains.** Two branches process two inputs through *the same* encoder → embeddings can be compared meaningfully. Reuse the Model object; don't build two copies.
- **Contrastive (pairs):** *"Pull same closer, push different apart up to the margin."*
- **Triplet (anchor / positive / negative):** *"Anchor's positive must be closer than its negative by at least α."*
- **Hard negatives drive learning; easy ones are gradient-zero waste.** Random triplet sampling stalls training within an epoch.
- **Distance is a learnable construct.** A trained Siamese's L2 distance is meaningful *only inside* one model — never compare distances across runs.

[🔝 Back to top](#top)

## 📖 Concept walkthroughs

### Siamese architecture — what "shared weights" actually means in code (full template)

> **🪜 Mental model:** *Twin networks with one set of brains.* Build the encoder **once**; call it **twice** on different inputs. The two calls share every weight — TensorFlow's autograd glues their gradients together automatically.

**What it is.** A **Siamese network** is an architecture where two (or more) inputs are passed through the **same** sub-network (the encoder, or "tower") to produce comparable **embeddings**, which are then combined via a distance metric (L2 or cosine) to yield a similarity score. The word "shared weights" simply means: there is *one* set of parameters, used on both inputs. The beginner confusion is whether to build "two towers that look the same" or "one tower called twice."

**Why it matters.** Without shared weights, the network learns two different mappings `f_A: X → Z_A` and `f_B: X → Z_B`. The embeddings `f_A(A)` and `f_B(B)` would live in *different latent spaces*, and the distance between them would be meaningless. Shared weights guarantee `f(A)` and `f(B)` are points in the *same* space, so `d(f(A), f(B))` is symmetric and well-defined. **The "shared weights" constraint is what makes the Siamese setup work at all** — break it and you've built something incoherent.

**How it works in code (the only correct way).** Build one Keras `Model` for the encoder; call it twice in the surrounding pair model:

```python
# 1. Build the encoder ONCE — single set of weights live in this Model.
embedding = build_encoder(input_shape=(180, 180, 3))

# 2. Define two input tensors, one per pair-side.
img_a = Input((180, 180, 3))
img_b = Input((180, 180, 3))

# 3. Call the SAME embedding model on both — TF sees this as one weight tensor.
feat_a = embedding(img_a)      # ← same weights
feat_b = embedding(img_b)      # ← same weights

# 4. Distance between the two outputs.
dist = layers.Lambda(euclidean_distance)([feat_a, feat_b])

pair_model = Model([img_a, img_b], dist)
pair_model.summary()           # ← params count is the encoder's, not 2×
```

The *wrong* way (subtle bug): calling `build_encoder()` twice — that creates two separate Models with two separate weight sets. The pair model would then have ~2× the parameter count, and the embeddings would never be in the same space. **A quick sanity check:** print `pair_model.summary()` — the trainable parameter count should equal the encoder's, not twice it.

**Where it's used.** Face verification (FaceNet, ArcFace), signature verification (this notebook), fingerprint matching, near-duplicate image retrieval, sentence-similarity embeddings (Sentence-BERT in NLP — same trick).

**Related terms.**
- **Embedding** — the output of the shared encoder; the vector that gets compared.
- **Symmetric distance** — `d(f(A), f(B)) = d(f(B), f(A))` follows automatically from sharing weights.
- **Twin / triplet / quadruplet** networks — same idea with two / three / four shared towers.
- **Pseudo-Siamese** — rare variant where the two towers are *similar* but not weight-shared (e.g., text↔image dual encoders); used only when the two inputs are from different modalities.

```python
# Forward pass — embedding called twice, weights shared.
embedding = build_encoder()
e_A = embedding(img_a)
e_B = embedding(img_b)
distance = K.sqrt(K.sum(K.square(e_A - e_B), axis=1))
```

**Gotcha.** Calling `build_encoder()` twice (instead of building once and calling twice) silently doubles the parameter count and breaks the symmetry. Always check `model.summary()` before training.


### One-shot / few-shot learning

> **🪜 Mental model:** *"Here's one signature — find more like it."* Recognise a new class (signer, face, identity) from a single (or a handful of) labelled examples by comparing embeddings, not by training a fresh classifier.

**What it is.** **One-shot learning** is the task of learning to recognise a new class from a *single* labelled example, and **few-shot learning** generalises that to a handful (typically 1–5). A classical CNN classifier with `K` output classes cannot do this — you'd need many examples per class and retraining whenever a class is added. Instead, one-shot models learn a generic **similarity function** `s(x, y)` during training (using many classes), and at deployment time recognise a new class by storing one reference embedding per class and comparing new inputs to it. The model itself never has to be retrained when you enrol a new identity.

**Why it matters.** Real-world identity tasks (face unlock, signature verification, fingerprint matching) face a fundamental constraint: **the set of identities changes constantly and is much larger than any training set could cover**. A bank with 10M customers can't retrain a 10M-class classifier whenever someone opens an account. One-shot via Siamese embeddings sidesteps this: train once on a representative set of identities to learn what "same" vs "different" looks like, then at deployment store one embedding per enrolled user. Adding user #10,000,001 is a database insert, not a training run.

**How it works (in this notebook).** The BHSig260 dataset is turned into a one-shot-style training problem in two ways:
1. **Pair sampling (for contrastive loss):** for every signer, build `(img_a, img_b, label)` tuples — `label = 1` if both signatures are from the same signer, `label = 0` otherwise. The model learns "are these two from the same identity?"
2. **Triplet sampling (for triplet loss):** for every signer, iterate over pairs of genuine signatures — `(anchor, positive)` are two genuines from the same signer; `negative` is a random forged signature. The model learns "the anchor is closer to its positive than to any negative."
3. **At deploy / enrolment:** store one reference embedding per new signer (`enrolled[signer_id] = embedding(reference_sig)`).
4. **At verification:** compute the embedding of a query signature, measure its distance to the stored reference for the claimed signer, compare against a threshold.

**Where it's used.**
- **Face Unlock (Face ID, Windows Hello)** — enrol with 1–5 photos, recognise from then on.
- **Bank signature verification** — this notebook's task.
- **Fingerprint / iris / voice biometrics.**
- **Wildlife re-identification** (individual whales / tigers from photographs).
- **Near-duplicate image retrieval** at scale.

**Related terms.**
- **Zero-shot learning** — recognise a new class from *no* labelled examples, using auxiliary information (text descriptions, attributes); CLIP is the canonical modern example.
- **Metric learning** — the umbrella ML sub-field that trains a similarity function; one-shot is a downstream application.
- **N-way K-shot evaluation** — standard one-shot benchmark format: at test time, pick one of `N` classes given `K` examples each.
- **Enrolment** — the deployment step where reference embeddings for new identities are stored.
- **Verification vs identification** — verification = "is this person X?" (1:1); identification = "who is this?" (1:N).

```python
# This notebook's enrolment + verification pseudocode
ref_embedding = embedding(reference_signature)        # store at enrolment
query_emb     = embedding(query_signature)
dist          = np.linalg.norm(ref_embedding - query_emb)
is_same       = dist <= threshold                     # threshold tuned on validation
```

**Gotcha.** "One-shot" refers to *the deployment regime, not the training regime*. The training set still contains many examples per training-class — what's "one-shot" is the *number of references at inference for a previously-unseen identity*. Beginners often think one-shot means "trained on one example total."

### Contrastive loss

> **🪜 Mental model:** *"Pull same closer, push different apart up to the margin."* Operates on a pair of images plus a binary label; squared distance for positives, squared margin-violation for negatives.

**What it is.** **Contrastive loss** (Hadsell et al., 2006) is a pair-based metric-learning loss. Given a pair `(x_A, x_B)` with binary label `y` (`1` if same identity, `0` if different) and the Euclidean distance `d = ‖f(x_A) − f(x_B)‖₂` between their embeddings, the loss is:

`L(y, d) = y · d²  +  (1 − y) · max(0, m − d)²`

Translating each term word by word:
- `y · d²` — *the "pull" term*. Active only on positive pairs (`y = 1`). Squared distance is minimised by pulling the two embeddings together (toward `d = 0`).
- `(1 − y) · max(0, m − d)²` — *the "push" term*. Active only on negative pairs (`y = 0`). Penalises negatives whose distance is *less than the margin* `m`, with no penalty once `d ≥ m`. So negatives are pushed apart only until they reach the margin gap; past that, the gradient is zero.
- `m` — the **margin** (hyperparameter; this notebook uses `m = 1`). Required gap between negative-pair distances and zero; without it, the network can trivially satisfy the loss by collapsing all embeddings to one point.

**Why it matters.** A classifier can't recognise unseen identities (Module 8's whole motivation). Contrastive loss instead trains the embedding to have the **geometric property** that same-identity points cluster and different-identity points stay apart — independent of how many identities there are. It's the simplest metric-learning loss; you should know its formula cold for any face/signature/fingerprint verification interview.

**How it works (in this notebook).**
1. Sample a balanced batch of pairs `(img_a, img_b, label)` — roughly equal positives and negatives.
2. Forward both images through the *shared* encoder to get `(feat_a, feat_b)`.
3. Compute `d = sqrt(sum((feat_a − feat_b)²))` per pair.
4. Compute the loss above; average over the batch.
5. Backprop — the encoder learns to compress same-identity pairs and separate different ones.

**Where it's used.**
- **This notebook's pair model** — `pair_model.compile(loss=contrastive_loss, ...)`.
- **Early face-verification systems** (pre-FaceNet).
- **Self-supervised representation learning** (SimCLR uses a normalised contrastive-style loss).
- **Near-duplicate detection, sentence-similarity embeddings.**

**Related terms.**
- **Triplet loss** — sibling loss using triples instead of pairs; enforces *relative* ordering rather than absolute distance.
- **Margin** — the required gap `m` between negative pairs and zero.
- **Embedding collapse** — the failure mode when margin is too small (or absent).
- **N-pair / InfoNCE / NT-Xent** — modern multi-negative variants that contrastive loss generalises to.

```python
def contrastive_loss(y, d, margin=1):
    y = tf.cast(y, d.dtype)
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))
```

**Gotcha.** Without a margin (or with one near zero), the model can drive all embeddings to a single point — both `y·d²` and `(1−y)·max(0, 0)² = 0` are satisfied. Always set `m > 0` and watch the mean embedding norm to detect collapse early.

### Triplet loss

> **🪜 Mental model:** *"Anchor's positive must be closer than its negative by at least α."* Three images per training example; the loss only fires if the relative ordering is wrong (or too close to wrong).

**What it is.** **Triplet loss** (Schroff et al., FaceNet 2015) is a triplet-based metric-learning loss. Each training example is three images: an **anchor** `a`, a **positive** `p` (same identity as the anchor), and a **negative** `n` (different identity). Let `d(a,p) = ‖f(a) − f(p)‖²` (squared L2) and similarly for `d(a,n)`. The loss is:

`L(a, p, n) = max(0,  d(a, p)  −  d(a, n)  +  α)`

Translating word by word:
- `d(a, p)` — anchor-to-positive distance. We want this small.
- `d(a, n)` — anchor-to-negative distance. We want this large.
- `d(a, p) − d(a, n)` — negative when the ordering is correct (positive closer); positive when it's wrong.
- `+ α` — the **margin** (this notebook uses `α = 0.5`). Even when the ordering is correct, the gap must exceed `α` for the loss to reach zero.
- `max(0, …)` — the "hinge" — once the constraint is fully satisfied (positive at least `α` closer than negative), the loss is exactly zero and contributes no gradient.

**Why it matters.** Triplet loss enforces a *relative* ordering rather than absolute distances, which often produces sharper embedding clusters than contrastive loss. The famous FaceNet paper showed this approach can achieve human-level face-verification accuracy with the right negative-mining strategy. It's the central loss in modern face-recognition systems and a near-mandatory interview topic — knowing the formula and the failure modes (collapse, easy negatives) is table stakes.

**How it works (in this notebook).**
1. **Generate triplets** with `get_triplets()`: for each signer, every pair `(j, k)` of their genuine signatures becomes `(anchor=j, positive=k)`; the negative is a random *forged* signature.
2. Three inputs go through the *same* shared encoder → three embeddings `(ea, ep, en)`.
3. Compute `d_ap = sum((ea − ep)²)` and `d_an = sum((ea − en)²)`.
4. Stack into a `(batch, 2)` tensor; pass to `triplet_loss` which computes `max(0, d_ap − d_an + 0.5)`.
5. Backprop only when the loss is non-zero (the easy triplets contribute zero gradient — which is why hard-negative mining matters).

**Where it's used.**
- **FaceNet** (the canonical face-recognition system, 2015–present).
- **This notebook's triplet model.**
- **Signature, fingerprint, iris verification.**
- **Sentence-BERT** in NLP (same trick on text).
- **Image retrieval at scale** (Pinterest, Google Image search).

**Related terms.**
- **Anchor / positive / negative** — the three roles. Anchor and positive share identity; anchor and negative do not.
- **Hard / semi-hard / easy negative** — three regimes defined by where `d(a, n)` falls relative to `d(a, p)` and the margin.
- **Hard-negative mining** — the strategy of intentionally selecting informative triplets (otherwise random ones are mostly easy → zero gradient → stalled training).
- **Contrastive loss** — sibling; pair-based, enforces absolute distances.
- **N-pair / InfoNCE** — modern multi-negative generalisations.

```python
def triplet_loss(_, dists, alpha=0.5):
    ap, an = dists[:, 0], dists[:, 1]
    return K.mean(K.maximum(0.0, ap - an + alpha))

# Use inside a model that produces both distances:
a, p, n = Input(shape), Input(shape), Input(shape)
ea, ep, en = embedding(a), embedding(p), embedding(n)
d_ap = K.sum(K.square(ea - ep), axis=-1, keepdims=True)
d_an = K.sum(K.square(ea - en), axis=-1, keepdims=True)
out  = layers.Concatenate(axis=1)([d_ap, d_an])
```

**Gotcha.** Random triplet sampling stalls training within an epoch because most random triples have the negative already far away (`d_an >> d_ap + α`) → `max(0, …) = 0` → zero gradient. **You must either pre-construct hard triplets (this notebook, by using forgeries as negatives) or use online semi-hard mining inside each batch.**

### Margin — why you need it (full template — the beginner-killer)

> **🪜 Mental model:** *Required safety gap.* Even after the model is "right," push positives and negatives apart by at least this much. Without it, the loss has a trivial optimum: collapse all embeddings to one point.

**What it is.** The **margin** (denoted `m` in contrastive loss, `α` in triplet loss) is a constant hyperparameter that defines the minimum required gap between same-identity and different-identity pairs in embedding space. In contrastive loss `y·d² + (1-y)·max(0, m − d)²`, the margin `m` is the distance beyond which different-identity pairs stop being penalised. In triplet loss `max(0, d(a,p) − d(a,n) + α)`, `α` is how much closer the positive must be than the negative for the loss to be zero.

**Why it matters.** Without a margin, the loss landscape has a degenerate minimum. Consider triplet loss without `α`: the constraint becomes `d(a,p) < d(a,n)`. The trivial solution is `f(x) = 0` for all inputs — every embedding collapses to the origin, all distances become zero, the constraint is "satisfied," and the network learns nothing. The margin breaks this degeneracy by requiring an *actual gap* between positive and negative distances; collapsing to a point now gives distance zero on both sides and the constraint `0 < 0 + α` is still violated, so the gradient stays alive. **Margin is the single most impactful hyperparameter in metric learning.** Too small → collapse; too large → optimisation stalls because the model can't satisfy the constraint even in principle.

**How it works.** It's literally just a constant inside the loss formula. The trick is its effect on the gradient: it shifts where the loss function starts contributing non-zero gradient. With `α = 0.5` in this notebook's triplet loss:
- If `d(a,p) = 0.3` and `d(a,n) = 1.0` → loss = `max(0, 0.3 − 1.0 + 0.5) = 0`. Already satisfied; no gradient (easy negative).
- If `d(a,p) = 0.3` and `d(a,n) = 0.6` → loss = `max(0, 0.3 − 0.6 + 0.5) = 0.2`. Within the margin; gradient flows (semi-hard).
- If `d(a,p) = 0.6` and `d(a,n) = 0.3` → loss = `max(0, 0.6 − 0.3 + 0.5) = 0.8`. Constraint violated; large gradient (hard).

**Where it's used.** Contrastive loss (this notebook), triplet loss (this notebook), ArcFace (additive angular margin softmax), CosFace (additive cosine margin), N-pair loss. Anywhere metric learning happens.

**Related terms.**
- **Embedding collapse** — the failure mode when margin is too small (all embeddings → one point).
- **Easy / semi-hard / hard negative** — the three regimes defined relative to the margin.
- **L2 normalisation** — common pre-processing before applying the margin; bounds distance scale to `[0, 2]` so margin tuning becomes reproducible.
- **ArcFace's angular margin** — modern generalisation that applies the gap in *angular* space.

```python
# Contrastive loss with margin
def contrastive_loss(y, d, margin=1):
    y = tf.cast(y, d.dtype)
    return K.mean(y * K.square(d) + (1-y) * K.square(K.maximum(margin - d, 0)))

# Triplet loss with margin α
def triplet_loss(_, dists, alpha=0.5):
    ap, an = dists[:, 0], dists[:, 1]
    return K.mean(K.maximum(0.0, ap - an + alpha))
```

**Gotcha.** Margin must be tuned to the *scale* of your embedding distances. After adding L2-normalisation (embeddings on the unit sphere), distances are bounded in `[0, 2]` — margin = 0.2 is reasonable. Without L2-norm on a 128-D Dense output, distances can be 10+ — margin = 1.0 may be way too small.


### Hard negative mining

> **🪜 Mental model:** *Train on the borderline cases.* Most random triplets / pairs contribute zero gradient; pick the informative ones (the ones where the model is currently wrong or nearly wrong) so every training step actually teaches something.

**What it is.** **Hard negative mining** is the strategy of intentionally selecting *informative* negative examples during metric-learning training, rather than picking negatives at random. A **hard negative** for a given anchor is a different-identity example that the current model places *closer* to the anchor than the anchor's own positive — i.e., `d(a, n) < d(a, p)`. A **semi-hard negative** is one that's farther than the positive but still within the margin: `d(a, p) < d(a, n) < d(a, p) + α`. **Easy negatives** are already outside the margin (`d(a, n) ≥ d(a, p) + α`) and contribute zero gradient.

**Why it matters.** With random sampling, the vast majority of negatives end up "easy" — the network already separates them correctly — and `max(0, d(a,p) − d(a,n) + α) = 0`. Zero loss means zero gradient means zero learning. Training stalls within an epoch on any non-trivial dataset (face, person re-id, etc.). Hard-negative mining keeps every batch educational by ensuring most triplets actually exert pressure on the embedding geometry. **This is the single most impactful trick in metric learning** — FaceNet's accuracy gains over earlier Siamese systems came largely from semi-hard mining.

**How it works.**
- **Offline hard mining**: every K epochs, run inference on the full training set, find the hardest negatives for each anchor (those with smallest `d(a, n)`), and use them in the next K epochs of training.
- **Online (batch) hard mining**: within each mini-batch of anchors-and-positives, search the batch itself for the hardest available negative for each anchor.
- **Online semi-hard mining (FaceNet)**: select, per anchor, the negative `n` satisfying `d(a,p) < d(a,n) < d(a,p) + α` — these violate the margin but are *not* already misclassified. Empirically more stable than hardest-of-hard mining (which can hurt training by chasing outliers / mislabelled examples).

**Where it's used in this notebook.** The notebook does **not** implement online semi-hard mining. Its `get_triplets()` function picks a *random forged signature* as the negative. This works only because BHSig260's forgeries are *intentionally designed to look like genuines* — they're hard by construction. So the dataset itself acts as a one-shot hard-negative miner. On a generic face dataset (mostly easy random negatives), the same code would stall within an epoch.

**Where it's used (broader).**
- **FaceNet** — defining example of online semi-hard mining at scale.
- **Person re-identification** at industrial scale.
- **OpenAI CLIP** and other contrastive vision-language models — use *all* in-batch negatives, an extreme form of "always-on" hard mining.
- **Triplet-loss-based recommender systems** (item retrieval).

**Related terms.**
- **Easy / semi-hard / hard negative** — the three regimes defined relative to the margin and the positive distance.
- **Curriculum** — broader idea: increase difficulty over training. Hard mining is curriculum's metric-learning special case.
- **Margin** — defines the boundary between semi-hard and easy.
- **Mining ratio** — fraction of hard:semi-hard:random in each batch; tunable.

```python
# Sketch — online semi-hard mining inside a batch
def semi_hard_mine(anchors, positives, all_embeddings, alpha=0.5):
    d_ap = pairwise_d(anchors, positives)                    # per anchor
    d_an = pairwise_d(anchors, all_embeddings)               # anchor vs every other
    mask_semi_hard = (d_an > d_ap) & (d_an < d_ap + alpha)   # FaceNet recipe
    # Pick one negative per anchor from the masked set
    return select_one(mask_semi_hard)
```

**Gotcha.** *Hardest*-negative mining (always pick the absolute closest negative) often *hurts* training — those are frequently mislabelled examples or outliers, and the gradient drags the embedding into instability. The semi-hard window (`d_ap < d_an < d_ap + α`) is what FaceNet showed empirically works best.

### Face verification

> **🪜 Mental model:** *"Two photos in, same person? out."* The canonical Siamese application; signature verification (this notebook) is structurally identical.

**What it is.** **Face verification** is the 1:1 task of deciding whether two face images depict the same person. It is distinguished from **face identification** (1:N — "which of the enrolled identities is this?") and **face recognition** (the umbrella term covering both). A modern verification pipeline is: detect the face → align it to a canonical pose → embed it with a trained encoder (FaceNet, ArcFace, etc.) → measure embedding distance → threshold to a yes/no decision.

**Why it matters.** Face verification powers every Face Unlock system (Face ID, Windows Hello, Pixel Face Unlock) and most modern biometric authentication. The 1:1 framing is also what makes signature verification (this notebook), fingerprint verification, and speaker verification *structurally identical* — the same Siamese-encoder + margin-loss + threshold pipeline works for all of them. **If you can explain face verification end-to-end, you can explain every biometric verification system.**

**How it works (the canonical FaceNet pipeline).**
1. **Detection** — find face bounding boxes (MTCNN, RetinaFace, BlazeFace).
2. **Alignment** — rotate/crop so eyes are at fixed positions; reduces intra-class variation.
3. **Embedding** — pass the aligned face through a trained Siamese-style network (FaceNet, ArcFace) → 128-D (or 512-D) embedding.
4. **L2-normalise** the embedding (project onto the unit sphere) so distances are bounded in `[0, 2]`.
5. **Distance** — compute `‖f(A) − f(B)‖₂` between the two faces' embeddings.
6. **Threshold** — declare "same person" iff `d ≤ t`, where `t` was tuned on a held-out validation set (e.g., to achieve a target FAR / FRR).

**How this notebook matches the pipeline.** No detection / alignment (signatures already cropped); ResNet-50 encoder + `Dense(128)` plays the role of FaceNet; no L2-normalisation (a deviation from production best practice); contrastive *or* triplet loss; `best_threshold` sweep on validation to pick `t`.

**Where it's used.**
- **Face Unlock** on every modern smartphone.
- **Bank account / payment authentication** (selfie verification).
- **Border control** at airports.
- **Photo-app face grouping** ("Photos of Alice").
- **Signature verification (this notebook), fingerprint, speaker, iris verification** — same pipeline structure.

**Related terms.**
- **Face identification** — sibling 1:N task: "which of the enrolled identities is this?" Solved by nearest-neighbour search over enrolled embeddings.
- **Verification vs identification** — 1:1 vs 1:N; the operating curve and threshold differ.
- **FAR / FRR (False Accept Rate / False Reject Rate)** — the two error rates traded off via the threshold.
- **EER (Equal Error Rate)** — the threshold where FAR = FRR; common single-number metric.
- **FaceNet / ArcFace / CosFace** — landmark embedding architectures using triplet / angular-margin softmax losses.

```python
# Inference pipeline (this notebook's signature variant; identical shape for faces)
ref_emb   = embedding(reference_image)          # one-time enrolment
query_emb = embedding(query_image)
dist      = np.linalg.norm(ref_emb - query_emb)
verified  = dist <= threshold                    # threshold from validation sweep
```

**Gotcha.** Always set the verification threshold *per-deployment* based on the operating point you need (e.g., bank: very low FAR; consumer unlock: low FRR for UX). A global "0.7" threshold is meaningless across different models, datasets, or device cameras.

### Embedding distance interpretation — what "distance < margin" means at inference (full template)

> **🪜 Mental model:** *Distance is a similarity score in disguise.* Small distance = same identity; large distance = different. The threshold you pick at inference is independent of the margin you trained with.

**What it is.** After a Siamese network is trained, the **embedding distance** `d = ‖f(A) − f(B)‖₂` is a real-valued similarity score: small distance = the model thinks `A` and `B` are the same identity, large distance = different. At inference time, you turn this score into a binary decision by comparing against a **threshold** `t`: if `d ≤ t`, declare "same"; else, "different." The threshold is **not** the training margin — it's a separately tuned hyperparameter that depends on the operating point you want (precision vs recall tradeoff).

**Why it matters.** Beginners often confuse the training margin with the inference threshold. They're related but not the same:
- The **margin** is a constant inside the loss formula. It tells the optimiser "make positive pairs at least this much closer than negative pairs." It's a *property of the loss*.
- The **threshold** is a decision boundary applied at inference time. It tells the classifier "anything closer than this counts as same." It's a *property of the deployed system*.

You can train with margin = 1.0 and still set the inference threshold to 0.7 (because at convergence most same-identity distances ended up around 0.4 and most different-identity ones around 1.2 — the natural separation is what matters, not the margin literal value).

**How it works (this notebook's `best_threshold` sweep).**
1. Compute distances on a held-out **validation** set: `dists = [d(A_i, B_i) for each pair]`. You also know each pair's true label.
2. Sweep candidate thresholds `t ∈ [dists.min(), dists.max()]` in small steps (e.g., 0.01).
3. For each `t`, classify every pair as same when `d ≤ t`. Compute accuracy.
4. Pick the `t*` that maximises accuracy (or another metric like F1 / EER depending on the use case).
5. **Freeze `t*`** and report performance on the test set — never re-sweep on test (that's leakage).

```python
def best_threshold(distances, labels):
    best_acc, best_t = 0, 0
    for t in np.arange(distances.min(), distances.max(), 0.01):
        pred = (distances <= t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc: best_acc, best_t = acc, t
    return best_t, best_acc
```

**Where it's used.** Every Siamese deployment. Face Unlock (per-device-calibrated threshold on enrolment images). Bank signature verification (per-customer or global threshold tuned on validation). Speaker verification.

**Related terms.**
- **Margin** — training-time constant; distinct from inference threshold.
- **EER (Equal Error Rate)** — the threshold where false-accept rate equals false-reject rate; common verification metric.
- **ROC curve** — sweep all thresholds, plot TPR vs FPR; the threshold is just one operating point on this curve.
- **L2 normalisation** — applying it puts all embeddings on the unit sphere, bounding distances to `[0, 2]` and making thresholds more interpretable.

**Gotcha.** **Never tune the threshold on the test set.** Sweep on validation, freeze, then evaluate on test. Also: embedding distances are *not* comparable across different trained models — a threshold of 0.7 for model A means nothing for model B. Always re-tune `t*` per model.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset
- **BHSig260** signatures dataset.
- 260 signers × ~54 sigs = 14,040 images.
- Per signer: 24 genuine + 30 forged.
- Format: `.tif`, resized to 180 × 180 × 3.
- For pairs: `(input_a, input_b, label)` where `label = 1` if same person, `0` otherwise.
- For triplets: `(anchor, positive, negative)` — anchor & positive from same signer, negative is a forged signature.

### 2. Build the shared embedding network
```python
def build_encoder(input_shape=(180, 180, 3)):
    base = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=input_shape,
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    out = layers.Dense(128)(x)
    return Model(base.input, out, name='embedding')

embedding = build_encoder()
```
ResNet-50 is frozen (transfer learning); only the two small Dense heads are trained. This Model is built **once** and reused.

### 3. Contrastive loss — pair training
```python
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

img_a = Input((180, 180, 3))
img_b = Input((180, 180, 3))
feat_a = embedding(img_a)
feat_b = embedding(img_b)
dist   = layers.Lambda(euclidean_distance)([feat_a, feat_b])

pair_model = Model([img_a, img_b], dist)

def contrastive_loss(y, d, margin=1):
    y = tf.cast(y, d.dtype)
    return K.mean(y * K.square(d) + (1-y) * K.square(K.maximum(margin - d, 0)))

pair_model.compile(loss=contrastive_loss, optimizer='adam')
```
The same `embedding` model is called twice — that's the "shared weights" trick. The `Lambda` wraps the Euclidean-distance Python function into a Keras layer so it shows up in the computational graph.

### 4. Triplet loss — triplet training
```python
def triplet_loss(_, dists, alpha=0.5):
    ap, an = dists[:, 0], dists[:, 1]
    return K.mean(K.maximum(0.0, ap - an + alpha))

a, p, n = Input(shape), Input(shape), Input(shape)
ea, ep, en = embedding(a), embedding(p), embedding(n)

d_ap = K.sum(K.square(ea - ep), axis=-1, keepdims=True)
d_an = K.sum(K.square(ea - en), axis=-1, keepdims=True)
out  = layers.Concatenate(axis=1)([d_ap, d_an])

triplet_model = Model([a, p, n], out)
triplet_model.compile(loss=triplet_loss, optimizer='adam')
```
The same `embedding` is now called *three* times. Outputs `d_ap` and `d_an` are stacked into a `(batch, 2)` tensor that the loss splits apart internally. Keras requires a `y_true` arg even though we don't use it (hence `_`).

### 5. Triplet generation (offline)
```python
def get_triplets(orig_groups, forg_groups_all):
    anchors, positives, negatives = [], [], []
    for orig_gp in orig_groups:            # per signer
        for j in range(len(orig_gp)):
            for k in range(j+1, len(orig_gp)):
                anchors.append(orig_gp[j])     # genuine signature
                positives.append(orig_gp[k])   # ANOTHER genuine from same signer
                negatives.append(             # random forged signature
                    forg_groups_all[np.random.randint(len(forg_groups_all))]
                )
    return anchors, positives, negatives
```
Iterating over `(j, k)` pairs gives `C(n, 2)` triplets per signer. Negatives are random forged sigs — this is *offline* sampling; production would use online semi-hard mining.

### 6. Custom training step (TF subclass model)
```python
class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name='loss')
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            anchor, positive, negative = self.siamese_network(data)
            pos = tf.reduce_sum(tf.square(anchor - positive), -1)
            neg = tf.reduce_sum(tf.square(anchor - negative), -1)
            loss = tf.maximum(pos - neg + self.margin, 0.0)
        grads = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}
```
This is a more idiomatic TF2 way to train — override `train_step` so you can keep using `.fit()` instead of writing a raw training loop. The `GradientTape` records ops for automatic differentiation.

### 7. Inference + threshold sweep
```python
def best_threshold(distances, labels):
    best_acc, best_t = 0, 0
    for t in np.arange(distances.min(), distances.max(), 0.01):
        pred = (distances <= t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc: best_acc, best_t = acc, t
    return best_t, best_acc
```
The threshold is swept on validation; freeze it before evaluating on test. See the "Embedding distance interpretation" template above for why this is *not* the same as the training margin.

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Concept | Implementation |
|---|---|
| Shared encoder | One `Model`, called twice in the same graph → weights shared |
| `layers.Lambda(fn)` | Wrap a Python function as a Keras layer (e.g., `euclidean_distance`) |
| `K.sqrt`, `K.sum`, `K.square`, `K.maximum`, `K.mean` | Backend ops for custom losses |
| `tf.GradientTape()` | Manual gradient computation for custom training loops |
| `model.train_step(data)` | Override for custom losses without `.fit()`-incompatible code |
| Contrastive loss | `y·d² + (1-y)·max(margin-d, 0)²` |
| Triplet loss | `max(0, d(a,p) - d(a,n) + α)` |
| Triplet sampling | Anchor + positive from same identity + negative from different |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **Shared weights are easy to break** — if you build two separate encoder copies, the network learns *two different* functions and similarity becomes meaningless. Always reuse the **same** Model object on both inputs. Sanity check: `pair_model.summary()` parameter count should equal `embedding.summary()`'s.
2. **Margin too small → embedding collapse.** All embeddings shrink toward zero (any distance satisfies the constraint). Diagnose by checking `np.linalg.norm(embeddings).mean()` — should stabilize, not go to 0.
3. **Margin too large → can't satisfy.** Training loss plateaus high. Reduce margin or check that your hard negatives aren't truly indistinguishable.
4. **Random triplet sampling stalls training.** Most random triplets are *easy* (negatives already far) → gradient = 0. Use **online semi-hard mining** for production.
5. **Don't compare embedding distances across training runs.** The scale depends on initialization and regularization. Always set a threshold per model.
6. **Threshold-on-test-set is leakage** — sweep the threshold on a *validation* set, then report accuracy on test with that fixed threshold.
7. **Forged signatures aren't true negatives in BHSig260.** They were intentionally made to look similar to genuines → they're hard negatives by construction. This is actually ideal for training on this dataset, but the same sampling code would stall on a "mostly easy negatives" dataset like raw face pairs.
8. **`Dense(128) → Dense(128)` back-to-back is redundant** unless an activation sits between them. The notebook's architecture has them linear; production would put a ReLU or L2-normalisation between, or skip the second Dense.

[🔝 Back to top](#top)

## 🎯 Notebook quiz cells

**Q1.** Why can't traditional CNNs handle signature verification well? *(adapted from `chiphuyen/ml-interviews-book`, face-recognition chapter)*
→ Classifier needs fixed classes. Adding a new signer would require retraining with `K+1` classes and re-deploying. Siamese learns a similarity metric — new identities just need a stored reference embedding; no retraining.

**Q2.** Contrastive vs triplet loss — difference?
→ Contrastive on pairs (binary same/different label). Triplet on triples (relative ordering: anchor closer to positive than negative by margin). Triplet often gives sharper embeddings because it enforces *relative* distances rather than absolute.

**Q3.** What is hard negative mining? *(adapted from `alexeygrigorev/data-science-interviews`, metric learning)*
→ Sample negatives that are close to the anchor (`d(a,n) < d(a,p)`). They violate the margin maximally → produce high loss → drive learning. Random negatives are mostly easy and contribute zero gradient. FaceNet uses *semi-hard* negatives — `d(a,p) < d(a,n) < d(a,p) + α` — for stability.

**Q4.** Why does margin matter in triplet loss? *(common FAANG metric-learning question)*
→ Forces a minimum gap between positive and negative distances. Too small → embeddings collapse to a point (trivial optimum). Too large → optimisation stalls (constraint can never be satisfied).

**Q5.** How do shared weights enforce symmetric similarity?
→ Both branches compute `f(·)` with the same parameters. Therefore `f(A) = f(B)` when `A = B`, and `d(f(A), f(B)) = d(f(B), f(A))`. Without sharing, you'd learn two different functions and comparison would be meaningless.

**Q6.** What's the difference between the training margin and the inference threshold?
→ The margin (`m` or `α`) is a constant inside the loss formula — it tells the optimiser how far apart to push positives and negatives. The threshold is a decision boundary applied at inference — sweep on validation, pick the value that maximises accuracy (or EER / F1), freeze for deployment. Same conceptual axis, different roles.

**Q7.** Why use a Lambda layer for the distance computation? *(adapted from `andrewekhalel/MLQuestions`)*
→ The Keras functional API needs every step to be a Layer with input/output tensors. `Lambda` wraps an arbitrary Python function so it integrates into the graph and TF can backprop through it. Modern alternative: write a `tf.keras.layers.Layer` subclass for more complex ops.

**Q8.** Why are BHSig260 forgeries actually good training negatives?
→ The forgeries were *intentionally* drawn by humans to look like genuines — they're hard-by-construction. Random "different signer" negatives would be easy (totally different handwriting style) and contribute zero gradient. Forgeries put the network into a useful learning regime by default — no online mining needed for this specific dataset.

**Q9.** What's the difference between face verification and face identification, and how does it change the deployment pipeline? *(adapted from `chiphuyen/ml-interviews-book`, face-recognition chapter)*
→ Verification is 1:1 ("is this person X?") — compare query embedding to one stored reference; threshold the distance. Identification is 1:N ("who is this?") — compare query against all enrolled embeddings; return nearest neighbour (often with an "unknown" fallback if even the nearest is too far). Identification needs a fast nearest-neighbour index (FAISS at scale); verification just needs one distance computation.

**Q10.** Explain one-shot learning with a Siamese network as if to a junior engineer. *(common FAANG metric-learning question)*
→ A classifier learns to map inputs to a fixed set of class labels; adding a new class means retraining. A Siamese network instead learns a *similarity function* — given two inputs, output "how alike are they." Once trained, enrolling a new identity is just storing one reference embedding; verifying a query is one distance computation against that reference. No retraining, no fixed class set, scales to millions of identities. The training data needs many identities (to learn what "similar" looks like), but each *new* identity at inference is recognised from a single example.

**Q11.** If you train a Siamese with margin = 0.2 and then deploy with threshold = 0.7, is that a bug? *(adapted from `andrewekhalel/MLQuestions`)*
→ No. The training margin tells the optimiser how far apart to push positives and negatives during training; the inference threshold is a *separate* decision boundary tuned on validation. After training, you sweep candidate thresholds on a held-out validation set and pick the one that maximises whatever metric matters (accuracy, F1, EER). The two numbers can coincide but usually don't — they answer different questions.

[🔝 Back to top](#top)

## 🪞 Extra ladder — triplet sampling strategies

**Basic** — random triplets. Each batch: anchor + same-identity positive + random-different negative. ~70% of triplets are easy → mostly wasted compute.

**Intermediate** — **offline hard-negative mining**: precompute embeddings every K epochs, find the hardest negatives, use them in the next K epochs. Effective but adds overhead.

**Advanced** — **online semi-hard mining** (FaceNet, Schroff 2015): within each batch, for each (anchor, positive), search the batch for negatives `n` such that `d(a,p) < d(a,n) < d(a,p) + α`. These contribute meaningful gradient without being so hard they destabilize training. Standard production technique.

[🔝 Back to top](#top)

## What comes next

Siamese networks discriminate. [Notebook 9 →](../9.GANs%20for%20Image%20Generation/) takes the opposite tack — *generate* new images from random noise via adversarial training.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
