<a id="top"></a>
# CV Notebook 4 — Image Similarity & Embeddings (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §4](../CV_Revision_Guide.md#4-module4).

## What this notebook actually demonstrates

A working **reverse image search** on Caltech-101 (8,677 images, 101 categories). Extract a **2048-dim embedding** from ResNet-50's penultimate layer, L2-normalize, then do nearest-neighbour search:

| Approach | Top-5 Accuracy | Query Time |
|---|---|---|
| Brute-force, 2048 dims | 83.98% | ~59.8 ms |
| Brute-force, PCA → 150 dims | 84.37% | ~5.3 ms (**~10× faster**) |
| **Annoy approximate NN, 150 dims** | ~84% | **~42.5 µs** (~**1400× faster** than brute force) |

The killer insight: **PCA from 2048 → 150 dims actually *improved* accuracy by 0.4%** while making everything 10–1000× faster. Compression often denoises.

## 🪜 Mental anchor for this notebook

**Embeddings are coordinates in feature space.** Similar images sit near each other; dissimilar ones live far apart. The pretrained CNN gave you the coordinate system for free. Once you have coordinates, similarity is geometry → nearest neighbours.

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset: Caltech-101
- 8,677 images, 101 object categories + background clutter.
- 40–800 images per class.
- Diverse: animals, vehicles, household objects, faces, scenes.

### 2. Load ResNet-50 as a feature extractor
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
```python
generator = tf.keras.utils.image_dataset_from_directory(
    root_dir, shuffle=False, batch_size=128, image_size=(224, 224),
)
feature_list = model.predict(generator)        # shape: (N, 2048)
feature_list = feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)
```

### 5. Brute-force top-K nearest neighbours
```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
nn.fit(feature_list)

distances, indices = nn.kneighbors([feature_list[query_idx]])
# indices[0][0] = query itself (distance 0)
# indices[0][1:5] = 4 most similar images
```

### 6. Top-K accuracy evaluation
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
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=150).fit(feature_list)
feature_list_pca = pca.transform(feature_list)    # (N, 150)

# Cumulative explained variance — pick n_components by where curve plateaus
np.cumsum(pca.explained_variance_ratio_)
```

### 8. Annoy approximate NN (production-grade)
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
```python
from sklearn.manifold import TSNE

emb_2d = TSNE(n_components=2, perplexity=30).fit_transform(feature_list_pca[:4000])
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[:4000], cmap='tab20')
# You should see clear clusters: faces, flowers, cars, scenes
```

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

## ⚠️ Notebook-specific gotchas

1. **L2-normalize before any cosine/L2 search.** Without it, `metric='cosine'` and `metric='euclidean'` give different (and worse) results.
2. **`metric='angular'` in Annoy ≡ cosine** on normalized vectors. Confusing name.
3. **PCA must be fit on the SAME dataset** the index is built from. Don't refit PCA at query time.
4. **`preprocess_input` is model-specific.** Using `ResNet50` with `vgg16.preprocess_input` silently degrades quality.
5. **`t-SNE` is for visualization, NOT retrieval.** The 2D coordinates don't preserve high-dim neighborhoods well enough to search.
6. **`Annoy` is read-only after `build()`.** Add items first, then build, then query. Adding after build = error.
7. **More trees in Annoy = more accurate but more memory + slower build.** 40–100 trees is the production sweet spot.

## 🎯 Notebook quiz cells (verbatim)

**Q1.** Why use penultimate-layer activations instead of final logits?
→ Logits collapse rich features into class probabilities. The penultimate layer carries 2048-D semantics (texture, shape, parts) — much richer for similarity.

**Q2.** Why normalize embeddings with L2?
→ Normalizes magnitude → cosine and Euclidean become monotonic-equivalent → metric is scale-invariant.

**Q3.** How does Annoy scale to millions of images?
→ Builds a forest of binary trees that partition the feature space. Queries traverse O(log N) tree nodes vs O(N) for brute force.

**Q4.** Why PCA to 150 dims (and not 2048 or 50)?
→ Tested empirically: 1d → 14% accuracy; 50d → 82%; **150d → 84.4%**; 200d → 84.5%. Beyond 150 the improvements are marginal vs the speed/memory cost.

**Q5.** Difference between image classification and image similarity?
→ Classification assigns a single discrete label; similarity ranks images by relevance. Similarity uses richer intermediate features; classification discards them.

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

## What comes next

This notebook turned ResNet-50 into a feature extractor for similarity. [Notebook 5 →](../5.Object%20localization%20and%20detection%201/) moves from "what's in the image" to "where in the image" — bounding boxes, IoU, NMS, and the R-CNN family.

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
