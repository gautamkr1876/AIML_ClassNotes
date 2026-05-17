<a id="top"></a>
# CV Notebook 8 — Siamese Networks (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §8](../CV_Revision_Guide.md#8-module8).

## What this notebook actually demonstrates

**Signature verification** — given two signatures, decide whether they're from the same person. Classical classifier can't do this: every new signer would need retraining. **Siamese networks** solve it via **metric learning** — learn an embedding where same-identity signatures land near each other.

Dataset: **BHSig260** — 260 signers (160 Hindi + 100 Bengali) × ~54 signatures each (24 genuine + 30 forged). Total ~14,040 images.

Two loss formulations implemented:
1. **Contrastive loss** — pair-based: `y · d² + (1-y) · max(margin - d, 0)²`
2. **Triplet loss** — triplet-based: `max(0, d(a,p) - d(a,n) + α)`

Backbone: ResNet-50 (frozen) → Global Average Pool → Dense(128) → 128-D embedding.

## 🪜 Mental anchors for this notebook

- **Twin networks with shared weights.** Two branches process the two inputs, both produce embeddings using the SAME weights → distance is symmetric.
- **Contrastive (pairs):** "Push same closer, push different apart up to margin."
- **Triplet (anchor, positive, negative):** "Anchor's positive must be closer than its negative by at least α."
- **Hard negatives drive learning;** easy ones are gradient-zero waste.

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

## ⚠️ Notebook-specific gotchas

1. **Shared weights are easy to break** — if you build two separate encoder copies, the network learns *two different* functions and similarity becomes meaningless. Always reuse the **same** Model object on both inputs.
2. **Margin too small → embedding collapse.** All embeddings shrink toward zero (any distance satisfies the constraint). Diagnose by checking `np.linalg.norm(embeddings).mean()` — should stabilize, not go to 0.
3. **Margin too large → can't satisfy.** Training loss plateaus high. Reduce margin or check that your hard negatives aren't truly indistinguishable.
4. **Random triplet sampling stalls training.** Most random triplets are *easy* (negatives already far) → gradient = 0. Use **online semi-hard mining** for production.
5. **Don't compare embedding distances across training runs.** The scale depends on initialization and regularization. Always set a threshold per model.
6. **Threshold-on-test-set is leakage** — sweep the threshold on a *validation* set, then report accuracy on test with that fixed threshold.
7. **Forged signatures aren't true negatives in BHSig260.** They were intentionally made to look similar to genuines → they're hard negatives by construction. This is actually ideal for training.

## 🎯 Notebook quiz cells

**Q1.** Why can't traditional CNNs handle signature verification well?
→ Classifier needs fixed classes. Adding a new signer would require retraining with K+1 classes. Siamese learns a similarity metric — new identities just need a stored reference embedding.

**Q2.** Contrastive vs triplet loss — difference?
→ Contrastive on pairs (binary same/different label). Triplet on triples (relative ordering: anchor closer to positive than negative by margin). Triplet often gives sharper embeddings.

**Q3.** What is hard negative mining?
→ Sample negatives that are close to the anchor (`d(a,n) < d(a,p)`). They violate the margin maximally → produce high loss → drive learning. Random negatives are mostly easy and contribute zero gradient.

**Q4.** Why does margin matter in triplet loss?
→ Forces a minimum gap between positive and negative distances. Too small → embeddings collapse. Too large → optimization stalls.

**Q5.** How do shared weights enforce symmetric similarity?
→ Both branches compute `f(·)` with the same parameters. Therefore `f(A) = f(B)` when `A = B`, and `d(f(A), f(B)) = d(f(B), f(A))`. Without sharing, you'd learn two different functions and comparison would be meaningless.

## 🪞 Extra ladder — triplet sampling strategies

**Basic** — random triplets. Each batch: anchor + same-identity positive + random-different negative. ~70% of triplets are easy → mostly wasted compute.

**Intermediate** — **offline hard-negative mining**: precompute embeddings every K epochs, find the hardest negatives, use them in the next K epochs. Effective but adds overhead.

**Advanced** — **online semi-hard mining** (FaceNet, Schroff 2015): within each batch, for each (anchor, positive), search the batch for negatives `n` such that `d(a,p) < d(a,n) < d(a,p) + α`. These contribute meaningful gradient without being so hard they destabilize training. Standard production technique.

## What comes next

Siamese networks discriminate. [Notebook 9 →](../9.GANs%20for%20Image%20Generation/) takes the opposite tack — *generate* new images from random noise via adversarial training.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
