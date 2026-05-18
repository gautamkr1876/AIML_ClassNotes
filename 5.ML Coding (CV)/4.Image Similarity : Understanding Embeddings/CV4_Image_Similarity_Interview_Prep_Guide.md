<a id="top"></a>
# CV Notebook 4 — Image Similarity & Embeddings (Deep Dive)

> Per-notebook companion to the master guide. For the full cross-notebook cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §4](../CV_Revision_Guide.md#4-module4). **This deep-dive is standalone** — every concept it introduces is explained end-to-end here. You should never have to leave this file to understand a term.

## What this notebook actually demonstrates

A working **reverse image search** on Caltech-101 (8,677 images, 101 categories). Extract a **2048-dim embedding** from ResNet-50's penultimate layer, L2-normalize, then do nearest-neighbour search:

| Approach | Top-5 Accuracy | Query Time |
|---|---|---|
| Brute-force, 2048 dims | 83.98% | ~59.8 ms |
| Brute-force, PCA → 150 dims | 84.37% | ~5.3 ms (**~10× faster**) |
| **Annoy approximate NN, 150 dims** | ~84% | **~42.5 µs** (~**1400× faster** than brute force) |

The killer insight: **PCA from 2048 → 150 dims actually *improved* accuracy by 0.4%** while making everything 10–1000× faster. Compression often denoises.

## 🪜 Mental anchors for this notebook

- **Embeddings are coordinates in feature space** — similar images sit near each other; dissimilar ones live far apart. The pretrained CNN gave you the coordinate system for free. Once you have coordinates, similarity is geometry → nearest neighbours.
- **Embed once, query forever** — the gallery is embedded offline (slow, one-time); each user query is just one embedding + one index lookup (sub-millisecond). This split is *the* economic trick that makes image search affordable at web scale.
- **Compression denoises** — dropping 2 048 dims to 150 via PCA *improved* accuracy by 0.4%. The discarded dimensions were mostly noise; principal components are the signal.
- **Angle vs straight line** — cosine asks "do these point the same way?"; Euclidean asks "are these points close?" After L2-normalisation, the two answers rank neighbours identically.

## 📖 Concept walkthroughs

> Beginner-first introduction of every concept this notebook touches. Each entry is a full Concept Definition Template — mental model → what / why / how / where / related → code → gotcha — substituted with this notebook's actual shapes, numbers, and code. After this section the cell-by-cell walkthrough makes sense without re-explaining concepts.

### Embedding

> **🪜 Mental model:** *Image as a coordinate.* Each image becomes a fixed-length vector (its "address") in a high-dimensional space; similar images land near each other.

**What it is.** An **embedding** is a fixed-length dense vector — here **2 048 numbers** — that summarises an image in a way that makes "semantic similarity = vector closeness." Two images of dogs land near each other in this space; a dog and a car land far apart. In this notebook the embedding is pulled from ResNet-50's **penultimate layer** (the layer right before the 1 000-class classifier), accessed via `ResNet50(include_top=False, pooling='avg')`. For VGG-16 the embedding dim is 4 096; for MobileNet-V2 it's 1 280; for ResNet-50 it's 2 048.

**Why it matters.** Embeddings unlock everything a classifier can't do — similarity search, clustering, near-duplicate detection, face verification, "find more like this" recommendations. They let you ask *"which image is most like this one?"* instead of *"which of these 1 000 classes is this?"* Almost every modern visual-retrieval system (Google Lens, Pinterest, Tineye) starts here. In interviews, "build a reverse image search" is the single most common CV system-design ask, and the first sentence of the answer is always "extract embeddings from a pretrained backbone."

**How it works.**
1. Load a pretrained CNN with `include_top=False, pooling='avg'`. `include_top=False` drops the final softmax; `pooling='avg'` adds global average pooling on the last conv map so the output is a flat vector, not a 7×7×2048 tensor.
2. Preprocess one image the way the backbone expects: `preprocess_input` does ImageNet-specific channel-mean subtraction (it is *not* `/255`).
3. `model.predict(img[None])` produces a `(1, 2048)` array — that's the embedding.
4. L2-normalise (`e = e / ‖e‖₂`) so all embeddings sit on the unit sphere.
5. Store one embedding per gallery image. At query time, repeat steps 1–4 on the query and find nearest neighbours.

For this notebook, step 3 over all 8 677 Caltech-101 images produces a `(8 677, 2 048)` array — ~71 MB in `float32`. Small enough to keep in RAM and search by brute force; large enough that compressing with PCA pays off.

**Where it's used.**
- Reverse image search (Google Lens, Pinterest visual search, TinEye).
- Face verification (FaceNet stores one embedding per identity).
- Product / fashion recommendation by visual similarity.
- Near-duplicate detection in social-media moderation and dataset deduplication.
- The notebook uses it in **cell 3** (single-image embedding) and **cell 4** (batched extraction over the whole gallery).

**Related terms.**
- **Penultimate layer** — the layer the embedding is read off of (next entry).
- **Embedding space / latent space** — the high-dim vector space all embeddings live in.
- **L2 normalisation** — the unit-length step that makes cosine ≡ Euclidean.
- **Logits / softmax** — the layer *after* the penultimate; classification-only, lossy for similarity.
- **Word embedding** — the same idea in NLP (word2vec, BERT [CLS] vector).

```python
backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')
emb = backbone.predict(preprocess_input(img[None]))   # shape (1, 2048)
emb = emb / np.linalg.norm(emb)                       # L2-normalise → unit vector
```

**Gotcha.** Embeddings from a pretrained classifier are *good but not optimal* — the network was trained to classify, not to cluster. Two visually different cats can have closer embeddings than a cat and a cat-shaped chair because the classifier had to separate "cat" from "chair." For best results, train a Siamese / triplet network (Module 8).

### Penultimate-layer features (why pretrained CNNs are great embedders)

> **🪜 Mental model:** *The classifier's last "draft" before it commits to a label.* Final logits collapse everything into class scores; the layer just before them still has the rich, multi-purpose description.

**What it is.** A pretrained classifier CNN ends with: backbone → global pooling → fully-connected layer → softmax (1 000 class probabilities). The **penultimate layer** is the global-pooling output (the vector right before the final classifier). It's a fixed-length summary of the image *before* it gets squashed into class probabilities. In this notebook we expose it by loading `ResNet50(include_top=False, pooling='avg')` — that combination removes the final 1 000-way softmax and surfaces the **2 048-D pooled feature map** as the model's output.

**Why it matters.** This is the single most important practical trick in image retrieval, and the notebook's core lesson. Final logits are *too compressed* — 1 000 numbers can't preserve texture, shape, parts, or layout. The penultimate layer still carries that nuance but already has *high-level* semantics (a dog and a wolf are closer than a dog and a car) because it sits right under the classifier. So even though ResNet-50 was trained for classification, you get a **free semantic similarity space** as a side-effect.

**How it works.**
1. Train a CNN end-to-end on a big classification dataset (ImageNet, 1 000 classes, 1.2 M images).
2. Cross-entropy gradient descent optimises the *whole* network. The only way the final softmax can separate classes is if the penultimate layer pushes same-class images together and different-class images apart.
3. Discard the final softmax at inference time (`include_top=False`).
4. Replace the rest with global average pooling (`pooling='avg'`) — collapses the last conv feature map of shape `(7, 7, 2048)` into a flat `(2048,)` vector.
5. Feed any new image through and read off this 2 048-D vector. Distances/angles in that space approximate semantic similarity.

**Where it's used.**
- This notebook in **cell 2** (`ResNet50(include_top=False, pooling='avg')`).
- Every off-the-shelf reverse image search starts here.
- Transfer learning in general — freeze the backbone, replace the head.
- In PyTorch: `nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])`.

**Related terms.**
- **Embedding** — the vector you read off the penultimate layer.
- **Logits / softmax** — the layer *after* the penultimate; classification-only, lossy for similarity.
- **Global average pooling** — what converts the last conv map to a flat vector (`pooling='avg'`).
- **Transfer learning** — broader pattern: reuse a pretrained network's middle layers; embeddings are a special case.
- **Metric learning** — better-but-harder alternative: train a network *specifically* to produce a similarity-friendly space (Module 8).

```python
backbone = ResNet50(weights='imagenet',
                    include_top=False,    # drop the 1000-class softmax
                    pooling='avg')         # global average pool the last conv map
emb = backbone.predict(preprocess_input(img[None]))   # shape (1, 2048)
```

**Gotcha.** Don't use the *logits* layer for embeddings — it has only 1 000 numbers and they're all "probability mass for one of 1 000 ImageNet classes." Almost all visual nuance is gone by that point.

### Cosine similarity

> **🪜 Mental model:** *Same direction?* — measure the angle between two vectors, ignore their lengths.

**What it is.** **Cosine similarity** measures how aligned two vectors are by computing the cosine of the angle between them. The formula is `cos(θ) = (x · y) / (‖x‖ · ‖y‖)` — *take the dot product `x · y`, then divide by the product of the two vectors' lengths*. Range `[-1, 1]`: 1 = identical direction, 0 = perpendicular, −1 = opposite direction. Most embedding values are non-negative so the practical range is `[0, 1]`. **Cosine distance** = `1 − cos(θ)` (lower = more similar).

**Why it matters.** Pretrained-CNN embeddings have different magnitudes depending on image brightness/contrast — a bright image's embedding is naturally larger. If you use raw Euclidean distance, you'd partly be measuring brightness, not content. Cosine ignores magnitude and measures only *direction* (which is what encodes the semantic content). It is the default similarity metric for embeddings (text and image alike), and the answer interviewers expect when they ask *"what metric do you use for embeddings?"*

**How it works.**
1. Compute the **dot product** `x · y = Σᵢ xᵢ · yᵢ` — adds up element-wise products.
2. Compute each vector's **L2 norm** `‖x‖ = √(Σᵢ xᵢ²)` — its straight-line length from the origin.
3. Divide the dot product by the product of the two norms.
4. The result is the cosine of the angle θ between the two vectors — `1` if they point the same way, regardless of how long they are.

**Where it's used.**
- `sklearn.NearestNeighbors(metric='cosine')` for any embedding-based retrieval.
- Annoy's `metric='angular'` — confusingly named, but mathematically cosine-equivalent on the inside.
- FAISS's `IndexFlatIP` (inner product) — equals cosine on L2-normalised vectors.
- In this notebook: the **Annoy index** (cell 8) is built with `'angular'`, i.e., cosine on the L2-normalised PCA-reduced embeddings.

**Related terms.**
- **Euclidean distance** — sibling metric; measures straight-line distance, sensitive to magnitude (next entry).
- **Dot product** — the numerator of cosine.
- **L2 norm** — the denominator; the vector's length.
- **L2 normalisation** — divide by L2 norm so all vectors have length 1; makes cosine ≡ Euclidean rank-equivalent.
- **Angular distance** — Annoy's term; effectively a monotonic transform of cosine, not the geometric angle in radians.

```python
def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
# or, for many query pairs at once:
sklearn.metrics.pairwise.cosine_similarity(X, Y)
```

**Gotcha.** **Cosine similarity** ranges `[-1, 1]` (higher = more similar); **cosine distance** is `1 − cos(θ)` and ranges `[0, 2]` (lower = more similar). Mixing them up flips the ranking. Also: Annoy's `'angular'` metric is cosine-based but uses a transformed scale — read the docs before interpreting raw distances.

### Euclidean distance

> **🪜 Mental model:** *Straight line.* Lay a ruler between the two points in the high-dim space.

**What it is.** **Euclidean distance** (also called **L2 distance**) is the straight-line distance between two vectors in vector space. The formula is `‖x − y‖₂ = √(Σᵢ (xᵢ − yᵢ)²)` — *element-wise subtract, square each difference, sum them, take the square root*. Range `[0, ∞)`: 0 = identical vectors, larger = further apart. It's the same Pythagorean distance you used in school, generalised to 2 048 dimensions.

**Why it matters.** Euclidean is the default geometric distance in any vector space, and the most efficient distance to compute on most hardware. For embeddings, it's a tossup with cosine — the right answer depends on whether you've normalised. After L2-normalisation **cosine and Euclidean rank neighbours identically** (mathematically, `‖x − y‖² = 2 − 2 cos(θ)` on the unit sphere). Without normalisation, Euclidean mixes "different content" with "different brightness/contrast" because it includes magnitude. Many libraries (sklearn brute-force NN, FAISS `IndexFlatL2`) default to Euclidean, so it's the metric you'll see most in code even when the semantic intent is cosine.

**How it works.**
1. Subtract: `d = x − y` (element-wise, gives a 2 048-D difference vector).
2. Square each component: `d²ᵢ` — punishes any single-dimension mismatch quadratically.
3. Sum: `Σ d²ᵢ` — this is the **squared Euclidean distance**; many libraries return this to skip the sqrt.
4. Take the square root: `√(Σ d²ᵢ)` — back to a real distance.

For sorting nearest neighbours, the `sqrt` is unnecessary — squared Euclidean preserves ranking, so production code often skips it for speed.

**Where it's used.**
- `sklearn.NearestNeighbors(metric='euclidean')` — used in **cells 5 and 6** of this notebook for brute-force baseline retrieval.
- FAISS `IndexFlatL2` — the default brute-force index.
- Annoy's `metric='euclidean'`.
- k-means clustering, k-NN classifiers, almost every classical ML algorithm assumes Euclidean distance unless you tell it otherwise.

**Related terms.**
- **Cosine similarity** — sibling metric; measures angle, ignores magnitude (previous entry).
- **L2 norm** — the same square-root-of-sum-of-squares formula applied to one vector (its length from the origin).
- **Squared Euclidean** — drop the `sqrt`; same ranking, faster compute.
- **Manhattan / L1 distance** — sibling metric `Σᵢ |xᵢ − yᵢ|`; rarely used for embeddings.
- **L2 normalisation** — after this step, Euclidean and cosine give identical rankings.

```python
def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
# Vectorised pairwise:
from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(X, Y)            # shape (n_X, n_Y)
```

**Gotcha.** **Without L2-normalisation, Euclidean and cosine give *different* neighbours** — Euclidean is sensitive to magnitude, cosine isn't. The notebook uses Euclidean (sklearn) and cosine-equivalent (Annoy) interchangeably *because* every vector has been L2-normalised first — drop the normalisation and the two metrics diverge.

### L2 normalisation

> **🪜 Mental model:** *Shrink every arrow to length 1.* All vectors land on the surface of a unit sphere; comparisons then care only about direction.

**What it is.** **L2 normalisation** is the operation `x̂ = x / ‖x‖₂` — divide every component of a vector by the vector's total length. The result `x̂` has unit length (`‖x̂‖ = 1`) and points in the same direction as the original. Geometrically: every point gets projected onto the surface of a unit sphere. Algebraically: every embedding becomes a direction, not a (direction + magnitude) pair.

**Why it matters.** Pretrained-CNN embeddings have varying magnitudes (a bright, high-contrast image's embedding is larger; a faint image's is smaller). Raw Euclidean distance on these embeddings mixes content with luminance — that's a leakage you don't want. L2-normalising every embedding before storage and query forces all comparisons to be **direction-only**, which is what cosine measures. **Bonus property:** after L2-normalisation, cosine and Euclidean rank neighbours identically — so you can use whichever metric your NN library prefers without changing your results. This is *the* default preprocessing step in every embedding-based retrieval pipeline.

**How it works.**
1. Compute the vector's L2 norm: `‖x‖₂ = √(Σᵢ xᵢ²)`.
2. Divide every component by that norm: `xᵢ_new = xᵢ / ‖x‖₂`.
3. The resulting vector has length exactly 1.0.
4. For a batch of vectors of shape `(N, D)`, compute `norms = np.linalg.norm(X, axis=1, keepdims=True)` — the `keepdims=True` returns shape `(N, 1)` so broadcasting divides each row by its own scalar norm.

In code, this is the difference between four-character-typo silent failures and correct retrieval — see the gotcha below.

**Where it's used.**
- This notebook in **cell 3** (`features / norm(features)` for a single image) and **cell 4** (`feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)` for batch).
- Every embedding-based retrieval pipeline: do it once on the gallery, again on every query.
- Before building the PCA basis and the ANN index — both expect normalised input.
- Inside neural networks (`F.normalize(x, p=2, dim=1)`) when the loss function (e.g. ArcFace, contrastive) requires unit-norm features.

**Related terms.**
- **L2 norm** — the length being divided out (`‖x‖₂ = √Σxᵢ²`).
- **Unit sphere** — the surface every normalised vector ends up on.
- **Cosine similarity** — what L2-normalisation makes Euclidean-equivalent to.
- **Standardisation / z-score** — easy-to-confuse cousin; centres + scales each feature column. L2-norm is row-wise; standardisation is column-wise. Very different operations.

```python
# Single vector
v_hat = v / np.linalg.norm(v)

# Batch — note keepdims=True
X_hat = X / np.linalg.norm(X, axis=1, keepdims=True)   # X shape (N, D) → X_hat shape (N, D)
```

**Gotcha.** **`keepdims=True` is mandatory** in the batch form. Without it, `np.linalg.norm(X, axis=1)` returns shape `(N,)` not `(N, 1)`, and the broadcast divides each *column* by a row of norms — silently producing wrong embeddings. Verify with `np.linalg.norm(X_hat, axis=1)` — every entry should be 1.0.

### Nearest-neighbour search

> **🪜 Mental model:** *Closest-pin-in-the-atlas.* Given a query embedding, find the `k` stored embeddings that are closest.

**What it is.** **Nearest neighbour (NN) search** is the operation "given a query point, return the `k` stored points closest to it by some distance metric." For embeddings, the metric is cosine or Euclidean. **Brute-force NN** compares the query against *every* stored point — `O(N · D)` per query for `N` items in `D` dimensions. For this notebook's gallery (`N = 8 677, D = 2 048`), that's ~18M floating-point ops per query — runs in ~60 ms on a CPU. For larger `N` (1 M+), you switch to an approximate method (next entry).

**Why it matters.** NN search is the algorithm under reverse-image search, face verification, deduplication, and content recommendation. It's also a frequent system-design question: *"how would you scale image search to 1 B images?"* The answer is **not** brute force — but you should know exactly *why* it doesn't scale (`O(N·D)` per query × `Q` queries per second × `60 s/min` …). Brute force is the baseline you compare ANN against.

**How it works.**
1. Embed every gallery image into a `(N, D)` matrix (the notebook does this once, offline).
2. (Optional but recommended) L2-normalise rows and PCA-reduce to a smaller `D'` (the notebook uses `D' = 150`).
3. At query time: embed and normalise the query → `(1, D')`.
4. Compute distance from the query to every gallery row (`(N, D')` − `(1, D')` → `(N, D')` → norm along axis 1 → `(N,)`).
5. `argpartition` the distance vector to grab the `k` smallest. Sort those `k` for the final ranking.

The first returned neighbour of a gallery image is always *itself* (distance 0); the notebook explicitly skips index 0 when computing accuracy.

**Where it's used.**
- `sklearn.neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute').fit(features)` — this notebook's brute-force baseline (cell 5).
- KD-tree / Ball-tree algorithms (`algorithm='kd_tree'`) — work for low-dim data only; degenerate to brute force above ~20 dims (the **curse of dimensionality**).
- For ≥ 100K items in high dim, switch to Annoy / FAISS (next entry).
- k-NN classifiers (`KNeighborsClassifier`) — same search, then vote on labels.

**Related terms.**
- **k-NN classifier** — uses the same search but votes among neighbours for a label.
- **Approximate NN (ANN)** — sub-linear methods that trade a tiny accuracy loss for huge speedup (next entry).
- **Curse of dimensionality** — why tree methods fail in high dim.
- **Top-k retrieval / Recall@k** — the evaluation metric: did the true match appear in your top-`k` results?

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
nn.fit(feature_list)                      # (8677, 2048) gallery
distances, indices = nn.kneighbors([feature_list[query_idx]])
# indices[0][0] = the query itself; indices[0][1:5] = 4 most similar
```

**Gotcha.** Don't forget to **L2-normalise** stored and query embeddings before NN search if you want cosine semantics — otherwise you'll get magnitude-biased neighbours. Also: the query's first returned neighbour is always itself (distance 0); always slice it off (`indices[1:]`).

### PCA on embeddings

> **🪜 Mental model:** *Find the few axes where the data really varies; throw the rest away.* The discarded axes were mostly noise.

**What it is.** **PCA** (Principal Component Analysis) is a linear dimensionality-reduction technique. It finds the orthogonal directions of greatest variance in your data, projects the data onto the top `k` of them, and discards the rest. Input: an `(N, D)` matrix (here `(8 677, 2 048)`); output: an `(N, k)` matrix (here `(8 677, 150)`) — same `N`, smaller `D`. The first principal component (PC1) is the direction along which the data spreads most; PC2 is the next-largest spread direction, perpendicular to PC1; and so on.

**Why it matters.** Embeddings from a 2 048-dim layer have many redundant or low-information dimensions. PCA-reducing to 150 dims here gives three wins at once: (1) **10× faster** brute-force search (less arithmetic per pair), (2) **smaller ANN index file**, (3) **slightly higher accuracy** because the discarded dims were mostly noise (84.37% vs 83.98%). This "free lunch" — smaller, faster, *and* more accurate — is why PCA is the default compression step in production retrieval pipelines.

**How it works.**
1. Centre the data: `X_c = X − X.mean(axis=0)` (per-column mean subtraction).
2. Compute the covariance matrix `Σ = X_c.T @ X_c / (N−1)` — a `(D, D)` matrix capturing how features co-vary.
3. Eigendecompose `Σ` to get eigenvectors (directions) and eigenvalues (variance along each direction).
4. Sort eigenvectors by eigenvalue (largest first); the top `k` are the **principal components**.
5. Project: `X_pca = X_c @ V[:, :k]` — multiply by the `(D, k)` matrix of top-k eigenvectors. Output shape `(N, k)`.
6. `sklearn.PCA` does all of this in one call; pick `k` by inspecting `cumsum(pca.explained_variance_ratio_)` and stopping where the curve plateaus (typically ~95%).

In this notebook, the cumulative explained variance hits ~95% at `k = 150` — the empirical choice that won the speed/accuracy tradeoff.

**Where it's used.**
- This notebook in **cell 7** (`PCA(n_components=150).fit(feature_list)`).
- Almost every embedding-search pipeline above ~1 000 dims.
- EDA — visualising high-dim data by reducing to 2 or 3 dims (though t-SNE/UMAP are better for that).
- Whitening / decorrelating features before downstream models.
- Compressing word/sentence embeddings (768 → 128).

**Related terms.**
- **Principal component (PC)** — one of the orthogonal axes PCA finds, sorted by variance explained.
- **Explained variance ratio** — `pca.explained_variance_ratio_`; what fraction of total variance each PC captures.
- **t-SNE / UMAP** — sibling reduction techniques; *non-linear*, designed for 2D visualisation, *not* for retrieval (next entry).
- **Truncated SVD** — PCA's cousin that works on sparse matrices and doesn't require centring.
- **Whitening** — PCA followed by scaling each component to unit variance.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=150).fit(feature_list)         # fit ONCE on the gallery
feature_list_pca = pca.transform(feature_list)         # (8677, 150)

# Pick n_components empirically:
np.cumsum(pca.explained_variance_ratio_)               # plot, find the knee
```

**Gotcha.** **Fit PCA on the gallery, then `transform` (never refit) on the query.** Refitting PCA at query time produces a different basis from the gallery's and returns garbage neighbours. Persist the fitted `pca` object (pickle, joblib) alongside the index.

### t-SNE (for visualisation, not retrieval)

> **🪜 Mental model:** *Squish high-dim neighbourhoods into a 2-D map.* Preserves who-sits-near-whom locally; mangles distances globally.

**What it is.** **t-SNE** (t-distributed Stochastic Neighbour Embedding, van der Maaten & Hinton 2008) is a *non-linear* dimensionality-reduction technique designed for **visualisation**. It takes a high-dim embedding (e.g., `(8 677, 2 048)` or `(8 677, 150)`) and finds a 2-D or 3-D layout that preserves which points are close to which. The output is purely for plotting: a `(N, 2)` array you scatter-plot and colour by class.

**Why it matters.** It's the standard way to *see* embedding-space structure. In this notebook the t-SNE plot is the visual proof that the ResNet-50 + PCA pipeline works: faces cluster on one side, flowers on another, cars in their own region. If those clusters were absent in the t-SNE plot, the embeddings would be junk and brute-force NN accuracy would be near random. **Critical limitation:** t-SNE preserves *local* neighbourhoods but **distorts global distances** — points that look close in the 2-D plot may be far in the original 150-D space, and vice versa.

**How it works.**
1. Compute pairwise similarities in the original high-dim space using a Gaussian kernel (controlled by `perplexity`, typically 5–50).
2. Initialise a random 2-D layout for the same `N` points.
3. Compute pairwise similarities in the 2-D layout using a heavy-tailed Student-t distribution.
4. Use gradient descent to make the 2-D similarities match the high-dim ones — minimise the **KL divergence** between the two distributions.
5. Iterate ~1 000 steps; output the optimised 2-D coordinates.

In this notebook: `TSNE(n_components=2, perplexity=30).fit_transform(feature_list_pca[:4000])` — run on the first 4 000 PCA-reduced embeddings (t-SNE is `O(N²)`, slow on the full 8 677).

**Where it's used.**
- This notebook in **cell 9** (visualising the embedding space).
- Embedding inspection during model debugging.
- Communicating model quality to non-technical stakeholders.
- **Not** for retrieval: never run NN search on t-SNE coordinates.

**Related terms.**
- **UMAP** — newer alternative; faster, preserves global structure better, broadly preferred today.
- **Perplexity** — t-SNE's main hyperparameter; loosely "how many neighbours each point is influenced by." 5–50 typical.
- **PCA** — linear alternative; good for retrieval but visually less informative.
- **KL divergence** — what t-SNE minimises during optimisation.

```python
from sklearn.manifold import TSNE
emb_2d = TSNE(n_components=2, perplexity=30).fit_transform(feature_list_pca[:4000])
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[:4000], cmap='tab20')
```

**Gotcha.** **Never run NN search on t-SNE coordinates.** The 2-D projection preserves local clusters but distorts inter-cluster distances catastrophically — what looks like "closest neighbour" in the plot may be on the opposite side of the high-dim space. Use the original high-dim embeddings (or PCA-reduced ones) for actual retrieval; t-SNE is *inspection only*.

### FAISS / Annoy / Approximate nearest neighbours (ANN)

> **🪜 Mental model:** *Phone directory by region.* Instead of scanning every name, jump straight to the right region, then scan within.

**What it is.** **Approximate nearest neighbour (ANN)** algorithms trade a small accuracy loss (typically 0.1–1% recall) for a huge speedup (100–10 000× over brute force). They build an **index** offline that partitions or graphs the embedding space; at query time, they probe only a small region instead of every point. Three families dominate:
- **Tree-based:** **Annoy** (Spotify), KD-trees. A forest of random hyperplane splits; queries traverse log-depth.
- **Hash-based:** **LSH** (Locality-Sensitive Hashing). Hashes embeddings into buckets so that close points collide.
- **Graph-based / quantization:** **FAISS** (Facebook) — industrial-strength library with IVF (inverted file), PQ (product quantization), and HNSW (Hierarchical Navigable Small World) graphs. CPU or GPU. Scales to billions of vectors.

This notebook uses **Annoy** for ~8 677 images; FAISS is the typical choice above ~1 M items.

**Why it matters.** Brute force on 100 M images at 2 048 dims is ~200 s per query — impossible for any user-facing service. ANN brings it to milliseconds. Every production image-search system uses ANN under the hood (Pinterest Lens, Google Lens, TikTok For You). System-design interviews ask *"scale image search to 1 B"* — the answer is **FAISS + sharding + caching**.

**How Annoy works (this notebook's choice).**
1. Pick a random pair of points in the gallery; draw a hyperplane equidistant from them. All points get split into two halves.
2. Recurse on each half — repeat the random-hyperplane split until each leaf has ≤ `K` points.
3. That's one tree. **Build `n_trees` of them** (this notebook uses 40), each with different random splits — more trees = more accurate, larger file, slower build.
4. At query time: traverse every tree to its leaf, union the candidate leaves, then exact-search within that small candidate set. `O(log N)` per tree instead of `O(N)`.
5. **`build()` is one-way** — once called, you can't add more items. Always `add_item` everything first, then `build`, then save.

**How FAISS scales further.** `IndexFlatL2` is brute force. `IndexIVFFlat` clusters embeddings into Voronoi cells; at query time it only searches the cells nearest the query (sub-linear). `IndexIVFPQ` adds **product quantization** — splits each embedding into chunks, replaces each chunk with the ID of its nearest cluster centre, compresses 2 048 floats into ~64 bytes. `IndexHNSW` builds a navigable small-world graph; very fast lookups, larger memory.

**Where it's used.**
- This notebook in **cell 8** (`AnnoyIndex(150, 'angular')` → ~43 µs per query).
- Pinterest, Google Lens, TikTok recommendations.
- Vector databases (Pinecone, Weaviate, Qdrant, Milvus) — all wrap FAISS or HNSW.
- LLM Retrieval-Augmented Generation (RAG) — same algorithms, applied to text embeddings.

**Related terms.**
- **Recall@k** — the right metric for ANN accuracy: what fraction of true top-`k` neighbours does the ANN return?
- **PQ (Product Quantization)** — embedding compression, e.g. 2048 floats → 64 bytes.
- **HNSW** — graph-based ANN, current SOTA on most benchmarks.
- **IVF (Inverted File)** — partition embeddings into cells; search only nearest cells.
- **Brute force** — the baseline ANN compares against.

```python
from annoy import AnnoyIndex
idx = AnnoyIndex(150, 'angular')                   # angular = cosine on normalised vectors
for i, v in enumerate(features_pca):
    idx.add_item(i, v)
idx.build(40)                                       # 40 trees — production sweet spot
idx.save('caltech101.ann')

# Query
neighbours, distances = idx.get_nns_by_vector(query_vec, n=5, include_distances=True)
```

**Gotcha.** Annoy's `'angular'` is *cosine-based*, not the geometric angle in radians — confusing name. Also: **`build()` is one-way**, adding items after `build` errors. Always add all items first, *then* call `build`, *then* save.

### Image-retrieval pipeline (end-to-end)

> **🪜 Mental model:** *Cook the gallery once, serve queries forever.* All the expensive work (embedding every image, fitting PCA, building the index) happens offline. Each user query is one embedding + one index lookup.

**What it is.** The standard end-to-end recipe for content-based image retrieval (CBIR), instantiated for this notebook's Caltech-101 experiment:
1. **Backbone:** ResNet-50, `include_top=False, pooling='avg'` → 2 048-D embedding per image.
2. **Gallery embedding (offline):** read all 8 677 images via `image_dataset_from_directory` in batches of 128 → `(8 677, 2 048)`.
3. **L2-normalise (offline):** divide each row by its L2 norm → unit-length vectors.
4. **PCA (offline):** fit `PCA(n_components=150)` on the normalised gallery → `(8 677, 150)`.
5. **ANN index build (offline):** `AnnoyIndex(150, 'angular')`, add each row, `build(40)`, save.
6. **Query (online):** embed → normalise → `pca.transform(query)` (do NOT refit) → `index.get_nns_by_vector(...)` → return top-k image IDs.

**Why it matters.** This six-step recipe is the production architecture behind Google Lens, Pinterest visual search, TinEye, fashion-similarity systems, and any "find more like this" feature. A beginner who can reproduce these six steps on any dataset has implemented production-grade image search. FAANG system-design rounds ask exactly this pipeline (often phrased as *"design Pinterest Lens at 1 B images"*) and expect you to walk through it cleanly.

**How it works (the empirical lessons this notebook surfaces).**
- Brute-force on raw 2 048-D embeddings: **83.98% top-5 accuracy, ~60 ms/query.** Baseline.
- Brute-force on PCA-reduced 150-D embeddings: **84.37% accuracy, ~5 ms/query** — both *faster* and *more accurate* (compression denoised the embedding).
- Annoy ANN on 150-D embeddings: **~84% accuracy, ~43 µs/query** — 1 400× faster than brute force, with sub-1% accuracy loss.

The offline phase took ~1 minute on this notebook's CPU. For 1 M images it'd be ~1 hour on a GPU; for 1 B you'd parallelise across machines.

**Where it transfers.** Swap the backbone (ResNet-50 → EfficientNet, CLIP, DINO), swap the gallery (Caltech-101 → product catalogue, faces, satellite tiles), and the same six steps build the retrieval system. For >1 M items, swap Annoy for FAISS; for billion-scale, add IVF + product quantization + sharding.

**Related terms.**
- **Gallery / corpus** — the stored set of images you search against.
- **Query** — the input image whose neighbours you want.
- **CBIR (Content-Based Image Retrieval)** — academic name for the task.
- **Recall@k** — primary evaluation metric.
- **Vector database** — productionised wrapper around an ANN index (Pinecone, Weaviate).

```python
# OFFLINE — run once
features = backbone.predict(gallery_ds)                 # (N, 2048)
features = features / np.linalg.norm(features, axis=1, keepdims=True)
pca      = PCA(n_components=150).fit(features)
features_small = pca.transform(features)
idx = AnnoyIndex(150, 'angular')
for i, v in enumerate(features_small): idx.add_item(i, v)
idx.build(40); idx.save('gallery.ann')

# ONLINE — per query (~ms)
q = backbone.predict(preprocess(load(query_path))[None])
q = q / np.linalg.norm(q)
q = pca.transform(q)
neighbours, distances = idx.get_nns_by_vector(q[0], 5, include_distances=True)
```

**Gotcha.** The query *must* use **byte-identical preprocessing** to the gallery — same backbone, same `preprocess_input` (ResNet's mean-subtract is *not* VGG's), same L2-normalise, same PCA basis (call `.transform`, never `.fit`). Drift between offline and online preprocessing is the #1 silent failure mode for retrieval systems.

[🔝 Back to top](#top)

## 🧠 Cell-by-cell walkthrough

### 1. Dataset: Caltech-101
- 8,677 images, 101 object categories + background clutter.
- 40–800 images per class.
- Diverse: animals, vehicles, household objects, faces, scenes.

### 2. Load ResNet-50 as a feature extractor

`include_top=False` removes the final 1 000-class classifier; `pooling='avg'` collapses the last conv map to a flat 2 048-D vector via global average pooling. Net result: the model now outputs an embedding, not class probabilities.

```python
from tf.keras.applications.resnet50 import ResNet50, preprocess_input
from tf.keras.preprocessing import image
from numpy.linalg import norm
import numpy as np

model = ResNet50(weights='imagenet',
                  include_top=False,
                  pooling='avg',          # GlobalAveragePooling → 2048-dim vector
                  input_shape=(224, 224, 3))
```

### 3. Single-image embedding (with L2 normalization)

`preprocess_input` does ImageNet-specific channel mean subtraction (it is *not* a `/255` rescale — using the wrong one silently breaks the embedding). The final `/ norm(features)` is the L2-normalise step that makes cosine ≡ Euclidean ranking-equivalent downstream.

```python
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)              # ImageNet-specific!
    features = model.predict(arr).flatten()
    return features / norm(features)         # L2-normalize → unit vector
```

### 4. Batch extraction (8,677 images at once)

GPU shines here — one `model.predict(generator)` call vectorises the whole gallery in batches of 128. `axis=1, keepdims=True` broadcasts each row's norm against itself.

```python
generator = tf.keras.utils.image_dataset_from_directory(
    root_dir, shuffle=False, batch_size=128, image_size=(224, 224),
)
feature_list = model.predict(generator)        # shape: (N, 2048)
feature_list = feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)
```

### 5. Brute-force top-K nearest neighbours

`algorithm='brute'` forces sklearn to compute distances against every gallery point — the baseline you compare ANN against. The first returned neighbour is always the query itself (distance 0); skip it.

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
nn.fit(feature_list)

distances, indices = nn.kneighbors([feature_list[query_idx]])
# indices[0][0] = query itself (distance 0)
# indices[0][1:5] = 4 most similar images
```

### 6. Top-K accuracy evaluation

The metric: for each gallery image, how many of its 4 closest neighbours (excluding itself) share its class label? Aggregated across the whole gallery, this is the dataset-wide top-K retrieval accuracy.

```python
def calculate_accuracy(feature_list, labels):
    nn = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(feature_list)
    distances, indices = nn.kneighbors(feature_list)
    # For each query, count how many of its 4 nearest neighbours share its label
    neighbor_labels = labels[indices[:, 1:5]]
    query_labels    = labels[indices[:, 0]][:, None]
    correct = (neighbor_labels == query_labels).sum()
    total   = neighbor_labels.size
    return 100.0 * correct / total
```

### 7. PCA dimensionality reduction

Empirically pick `n_components` by plotting cumulative explained variance and finding the knee. Here, 150 dims captures ~95% variance and *improves* accuracy by 0.4% (noise removal).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=150).fit(feature_list)
feature_list_pca = pca.transform(feature_list)    # (N, 150)

# Cumulative explained variance — pick n_components by where curve plateaus
np.cumsum(pca.explained_variance_ratio_)
```

### 8. Annoy approximate NN (production-grade)

`metric='angular'` is cosine on normalised vectors. Add items first, then `build(n_trees)` (the structure becomes read-only), then `save` to persist. Higher `n_trees` = more accurate but slower build and larger file; 40 is a typical production sweet spot.

```python
from annoy import AnnoyIndex

dims = feature_list_pca.shape[1]
t = AnnoyIndex(dims, metric='angular')   # angular = cosine on normalized vectors
for i, v in enumerate(feature_list_pca):
    t.add_item(i, v)
t.build(n_trees=40)                       # tradeoff: more trees → more accurate but slower
t.save('caltech101.ann')

# Query
indices, distances = t.get_nns_by_vector(
    feature_list_pca[query_idx], n=5, include_distances=True,
)
```

### 9. t-SNE visualization

For *inspection only* — you should see clear clusters (faces, flowers, cars, etc.). Never run NN on these 2-D coordinates: t-SNE preserves local structure but distorts global distances.

```python
from sklearn.manifold import TSNE

emb_2d = TSNE(n_components=2, perplexity=30).fit_transform(feature_list_pca[:4000])
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[:4000], cmap='tab20')
# You should see clear clusters: faces, flowers, cars, scenes
```

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `ResNet50(weights='imagenet', include_top=False, pooling='avg')` | `pooling='avg'` adds `GlobalAveragePooling2D` → 2048-dim vector |
| `tf.keras.applications.resnet50.preprocess_input` | ImageNet mean-subtract; **don't** also `/255` |
| `numpy.linalg.norm(vec)` / `np.linalg.norm(arr, axis=1, keepdims=True)` | L2 normalization |
| `sklearn.neighbors.NearestNeighbors(n_neighbors, algorithm, metric)` | Brute-force / KD-tree NN |
| `sklearn.decomposition.PCA(n_components)` | Linear dim reduction |
| `sklearn.manifold.TSNE(n_components=2, perplexity)` | Non-linear visualization |
| `annoy.AnnoyIndex(dim, metric='angular'/'euclidean')` | Approximate NN index (tree-based) |
| `index.add_item(i, vec)` / `index.build(n_trees)` / `index.save(path)` | Build & persist Annoy index |
| `index.get_nns_by_vector(vec, n, include_distances)` | Query top-N neighbours |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **L2-normalize before any cosine/L2 search.** Without it, `metric='cosine'` and `metric='euclidean'` give different (and worse) results.
2. **`metric='angular'` in Annoy ≡ cosine** on normalized vectors. Confusing name.
3. **PCA must be fit on the SAME dataset** the index is built from. Don't refit PCA at query time — `pca.transform(query)` only, never `PCA().fit(query)`.
4. **`preprocess_input` is model-specific.** Using `ResNet50` with `vgg16.preprocess_input` silently degrades quality.
5. **`t-SNE` is for visualization, NOT retrieval.** The 2D coordinates don't preserve high-dim neighborhoods well enough to search.
6. **`Annoy` is read-only after `build()`.** Add items first, then build, then query. Adding after build = error.
7. **More trees in Annoy = more accurate but more memory + slower build.** 40–100 trees is the production sweet spot.
8. **Gallery and query preprocessing must match byte-for-byte** — same backbone, same `preprocess_input`, same L2-norm, same PCA matrix. Drift between offline build and online query is the #1 silent failure.
9. **`keepdims=True` in batch L2-norm is mandatory.** Without it, broadcasting silently divides by the wrong axis.
10. **First NN of a gallery image is itself.** Always slice off index 0 when computing top-k accuracy.

[🔝 Back to top](#top)

## 🎯 Notebook-specific Q&A

**Q1.** Why use penultimate-layer activations instead of final logits?
→ Logits are only 1 000 numbers (one per ImageNet class) — almost all visual nuance (texture, parts, layout) has been thrown away by then. The penultimate layer is 2 048 dims and still carries that nuance while having high-level semantics (dog and wolf land near each other). For similarity, that richness is everything. *(adapted from `andrewekhalel/MLQuestions`, CBIR section)*

**Q2.** Why L2-normalize embeddings?
→ Pretrained-CNN embeddings have magnitudes that vary with image brightness/contrast — raw Euclidean would partly measure brightness rather than content. L2-norm projects every vector to the unit sphere, removing magnitude bias and making cosine ≡ Euclidean rank-equivalent. *(common FAANG retrieval question)*

**Q3.** Difference between cosine similarity and Euclidean distance — and when does the choice matter?
→ Cosine measures the angle between vectors (direction only); Euclidean measures straight-line distance (direction + magnitude). On L2-normalised vectors they rank neighbours identically; otherwise Euclidean is biased by magnitude. Pick cosine when magnitude is noise (almost always for embeddings); pick Euclidean when magnitude carries information (rare in deep features). *(common FAANG, adapted from `chiphuyen/ml-interviews-book`)*

**Q4.** How does Annoy scale to millions of images?
→ Builds a forest of random-hyperplane binary trees. Queries traverse `O(log N)` per tree and union candidates across trees, then exact-search within the small candidate set. Speedup is 100–10 000× over brute force; recall loss is sub-1% with 40+ trees. *(adapted from `chiphuyen/ml-interviews-book`, retrieval chapter)*

**Q5.** Why PCA to 150 dims (not 2048 or 50)?
→ Empirically swept in this notebook: 1d → 14% accuracy; 50d → 82%; **150d → 84.4%**; 200d → 84.5%. Beyond 150 the gains are marginal vs the speed/memory cost. The 0.4% gain over raw 2 048-D is noise removal — the discarded dimensions were mostly noise.

**Q6.** Why does a pretrained classifier give good embeddings even though it wasn't trained for similarity?
→ Cross-entropy gradients push images of the same class together and different classes apart in the penultimate layer — that's the only way the final softmax can separate them. The geometry encodes semantic similarity as a side-effect of the classification objective. *(adapted from `alexeygrigorev/data-science-interviews`)*

**Q7.** Can you run nearest-neighbour search on t-SNE coordinates?
→ No — t-SNE preserves local clusters but distorts global distances. Two points 1 cm apart in the t-SNE plot may live in different regions of the original 150-D space. t-SNE is *inspection-only*; always do retrieval in the original (or PCA-reduced) high-dim space. *(common FAANG follow-up)*

**Q8.** What's FAISS and when would you reach for it over Annoy?
→ FAISS is Facebook's industrial-strength ANN library — supports inverted indices (IVF), product quantization (PQ), HNSW graphs, and CPU/GPU execution. Use it above ~1 M items, when you need GPU acceleration, or when memory is tight (PQ compresses 2 048 floats → ~64 bytes). Annoy is simpler and great for moderate scale (~100K–1M). *(common FAANG system-design, adapted from `chiphuyen/ml-interviews-book`)*

**Q9.** Design reverse image search for 1 B images — what changes vs this notebook?
→ Swap Annoy for **FAISS** (`IndexIVFPQ` for memory + speed). Shard the index across ~100 machines (10 M each). Add a per-shard LRU cache for hot queries. Use GPU FAISS for sub-millisecond per-shard latency. Replace ResNet-50 with a stronger encoder (CLIP, DINOv2) pretrained on a larger / domain-matched dataset. Batch queries to amortise overhead. *(common FAANG system-design, original)*

**Q10.** If queries come from a different distribution from the gallery (user phone photos vs studio catalogue shots), what would you change?
→ Fine-tune the backbone with **metric learning** (Siamese / triplet — Module 8) on `(query, matching-catalog-item)` pairs to align the two distributions in embedding space. Pure pretrained-ImageNet embeddings often put user photos in a different region than catalogue stock images, so similarity search fails even though both contain the same product. *(original, senior design-judgment)*

**Q11.** Why is the first returned neighbour always the query itself?
→ Because the query *is* in the gallery (in this notebook's accuracy evaluation), distance 0 to itself is by definition the smallest. Always slice off `indices[0]` before computing top-k accuracy or showing results to a user. *(common implementation bug, original)*

**Q12.** What is `keepdims=True` doing in `np.linalg.norm(X, axis=1, keepdims=True)`?
→ It preserves the reduced axis as size 1, returning shape `(N, 1)` instead of `(N,)`. Without it, the broadcast `X / norm` divides each *column* by a row of norms — silently wrong. Always pair batch-norm with `keepdims=True`. *(adapted from `rougier/numpy-100`)*

[🔝 Back to top](#top)

## 🪞 Extra ladder — production NN search

**Basic** — brute force.
```python
nn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(features)
```
Fine up to ~100k items. Scales O(N·D) per query.

**Intermediate** — PCA + brute force.
```python
features_small = PCA(n_components=150).fit_transform(features)
nn = NearestNeighbors(n_neighbors=5).fit(features_small)
```
~10× faster, no accuracy loss.

**Advanced** — Annoy / FAISS index + PCA.
```python
# FAISS for 1M+ items, GPU support
import faiss
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(150), 150, 100)
index.train(features_small); index.add(features_small)
distances, indices = index.search(query[None], k=5)
```
~1000× faster than brute force, 0.1–1% recall loss. **Production standard** for similarity search at scale.

[🔝 Back to top](#top)

## What comes next

This notebook turned ResNet-50 into a feature extractor for similarity. [Notebook 5 →](../5.Object%20localization%20and%20detection%201/) moves from "what's in the image" to "where in the image" — bounding boxes, IoU, NMS, and the R-CNN family.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
