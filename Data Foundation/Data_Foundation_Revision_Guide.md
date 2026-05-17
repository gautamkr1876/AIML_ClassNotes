<a id="top"></a>
# Data Foundation — NumPy Master Revision Guide

> **Consolidated quick-revision notes for the three NumPy notebooks** (Food Delivery EDA 1–3). Built for fast review and advanced interview prep. Every concept, every API, every gotcha — in scannable form, with deep Q&A and a revision drill at the end.

**Companion guides (Pandas pulled out for focused study):**
- 🐼 [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) — Series, DataFrame, `iloc`/`loc`, dtype cleanup, `.str` accessor, etc.
- 🛒 [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) — applied pandas (joins, groupby, apply, reshape, datetime), visualization, and probability for the Amazon + Sachin notebooks.

**How to use:**
- **Pre-interview:** read the "🚀 Topic finder" → skim a module's cheat sheet → drill the Q&A.
- **Just before a coding round:** run the [§13 Revision Drill](#13-drill).
- **For depth on NumPy basics:** the per-notebook deep-dive is [`Food Delivery .../NumPy_EDA_Interview_Prep_Guide.md`](./Food%20Delivery%20Data%20Exploration%20and%20analysis%201/NumPy_EDA_Interview_Prep_Guide.md).

**External practice (use after you've drilled this guide):**
- 🎯 [`rougier/numpy-100`](https://github.com/rougier/numpy-100) — 100 NumPy exercises across L1/L2/L3 difficulty.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — theory + coding (NumPy, stats, ML).
- 🎯 [`kojino/120-Data-Science-Interview-Questions`](https://github.com/kojino/120-Data-Science-Interview-Questions) — classic DS interview bank.
- 🎯 [`chiphuyen/ml-interviews-book`](https://huyenchip.com/ml-interviews-book/) — open ML-interviews book.

---

## 🚀 Topic finder

Jump straight to the topic you need.

| Need to revise… | Go to |
|---|---|
| ML, EDA, lists vs arrays, why NumPy fast | [Module 1](#1-module1) |
| Array creation, shape, dtype, indexing, slicing, astype | [Module 1](#1-module1) → [Cheat sheet](#1c-cheat) |
| Boolean masking, fancy indexing, reshape, axis, aggregation | [Module 2](#2-module2) |
| `np.where`, `np.any`/`all`, `np.sort`/`argsort`, matrix multiply | [Module 2](#2-module2) |
| Vectorization, broadcasting (4 rules), `np.tile` | [Module 3](#3-module3) |
| `split`/`hsplit`/`vsplit`, `vstack`/`hstack`/`concatenate` | [Module 3](#3-module3) |
| Pandas Series, DataFrame, indexing, cleanup | → [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) |
| Joins, groupby, apply, reshape, datetime, plotting, probability | → [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) |
| All NumPy terms at once | [§9 Master terms](#9-terms) |
| Every NumPy API at once | [§10 API cheat sheet](#10-apis) |
| Common NumPy gotchas | [§11 Gotchas](#11-gotchas) |
| Hard interview questions (advanced) | [§12 Advanced Q&A](#12-advanced) |
| 🌐 Sourced interview questions (drill bank) | [Sourced bank](#sourced-bank) |
| Speed-run revision before interview | [§13 Drill](#13-drill) |
| Best practices | [§14 Best practices](#14-bestpractices) |

---

## 📑 Table of contents

1. [Module 1 — NumPy Foundation](#1-module1)
2. [Module 2 — Filtering, Reshape, Aggregation, Matrix Multiply](#2-module2)
3. [Module 3 — Broadcasting, Vectorization, Stack & Split](#3-module3)
4. *Pandas → moved to [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md)*
5. [Cross-module concept map](#5-conceptmap)
6. [The 5 mental anchors (memorize these)](#6-anchors)
7. [Zomato dataset cheat sheet](#7-zomato)
8. [Common business questions → which API](#8-businessmap)
9. [📚 Master terms glossary](#9-terms)
10. [⚙️ API cheat sheet (every method, one table)](#10-apis)
11. [⚠️ Gotchas & traps (all in one place)](#11-gotchas)
12. [🎯 Advanced interview Q&A](#12-advanced)
12B. [🌐 Sourced interview questions](#sourced-bank)
13. [🔁 NumPy revision drill (70 questions)](#13-drill)
14. [✅ Best practices](#14-bestpractices)
15. [📦 What's in each notebook (mapping)](#15-mapping)

---

<a id="1-module1"></a>
## 1. Module 1 — NumPy Foundation

> Notebook 1 — ML motivation, EDA, lists vs arrays, why NumPy is fast, array creation, shape/dtype, type coercion, `astype`, indexing, slicing. Deep dive in [NumPy_EDA_Interview_Prep_Guide.md](./Food%20Delivery%20Data%20Exploration%20and%20analysis%201/NumPy_EDA_Interview_Prep_Guide.md).

### 🪜 Mental model

**Tea-room analogy.** A Python list scatters its ingredients across the room — each integer is a separate object on the heap, accessed by pointer-chase. NumPy lines them all up on one shelf (contiguous memory + homogeneous dtype). *That single layout choice explains every NumPy speedup and every NumPy gotcha you'll meet.* Hold this image in your head and most behaviors become predictable.

### 🪞 Basic → Intermediate → Advanced — dtype & type coercion

**Basic** — a NumPy array has *one* dtype shared by every element.
```python
np.array([1, 2, 3]).dtype           # int64
```

**Intermediate** — mixing types promotes the whole array (priority: **str > float > int > bool**).
```python
np.array([1, 2.5, 3]).dtype         # float64
np.array([True, 6]).dtype           # int64 (True → 1)
```

**Advanced** — one stray string silently coerces everything to fixed-width Unicode and breaks subsequent math. Defend with explicit `dtype=` so the failure is loud.
```python
np.array([1, 2, 'oops']).dtype      # <U21 — arithmetic is now broken
np.array([1, 2, 'oops'], dtype=float)   # ValueError — fail loud, fix at the source
```

### 🪞 Basic → Intermediate → Advanced — indexing & slicing

**Basic** — slicing uses `start:end:step`, **end exclusive**, and returns a view (shared memory).
```python
arr = np.arange(10)
arr[2:5]                             # array([2, 3, 4])
```

**Intermediate** — 2D arrays index by `[row, col]`. Comma-form (`arr[r, c]`) is one C call; chained-form (`arr[r][c]`) is two.
```python
arr2d[1, 2]                          # idiomatic and faster
arr2d[1:3, 0]                        # 2 rows × 1 col → 1D
```

**Advanced** — basic slices are **views**, fancy/boolean indexing returns **copies**. Mutating a view changes the source; mutating a copy doesn't. Verify with `b.base is a` or `np.shares_memory(a, b)`.
```python
sub = arr[1:4]; sub[0] = 99          # arr is mutated
sub = arr[[1, 2, 3]]; sub[0] = 99    # arr unchanged (fancy = copy)
```

<a id="1c-cheat"></a>
### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| ML | Learning patterns from history to predict the future |
| EDA | Understand → clean → explore data before modelling |
| List vs array | `[1,2]*2 → [1,2,1,2]`; `np.array([1,2])*2 → [2,4]` |
| Why fast | Contiguous memory + homogeneous dtype + vectorized C/SIMD |
| `ndarray` | NumPy's core data structure |
| Shape | Tuple of dims, e.g. `(5, 2)` |
| ndim | Number of dimensions |
| size | Total element count |
| dtype | Element type (`int64`, `float64`, `<U7`) |
| Type priority | **String > Float > Int > Bool** |
| `astype` | Returns new array; **truncates** floats, doesn't round |
| Indexing | Positive, negative, fancy. 2D = `[row, col]` |
| Slicing | `start:end:step`. **End is exclusive.** View, not copy |
| Reverse | `arr[::-1]` |

### ⚙️ Top APIs

```python
np.array, np.zeros, np.ones, np.full, np.arange, np.linspace, np.eye
np.random.rand, np.random.randn, np.random.randint, np.random.seed
arr.shape, arr.ndim, arr.size, arr.dtype, arr.itemsize, arr.nbytes
arr.astype(dtype), arr.copy()
arr[i], arr[-i], arr[[i, j, k]], arr[i, j], arr[i:j:k]
```

### 🎯 Advanced Q&A — Module 1

> Mix of original drills and questions adapted from `rougier/numpy-100`, `alexeygrigorev/data-science-interviews`, and `kojino/120-Data-Science-Interview-Questions`.

1. **Why is NumPy 10×–100× faster than a Python list for math?** *(common FAANG opener)*
   Three reasons: (1) contiguous memory block (cache-friendly, no pointer-chase), (2) homogeneous dtype (CPU streams through with no per-element type checks), (3) operations dispatch to **vectorized C with SIMD** instructions instead of Python interpreter loops.

2. **What does `dtype='<U7'` mean and when does NumPy choose it?** *(adapted from `rougier/numpy-100`)*
   Little-endian Unicode string, up to 7 characters. NumPy picks it when **any** element in the input list is a string — type priority promotes the whole array.

3. **Does `astype(int)` round or truncate?**
   **Truncate.** `1.9 → 1`. To round properly: `np.round(arr).astype(int)`.

4. **What's the difference between `(3,)`, `(3, 1)` and `(1, 3)`?** *(common broadcasting trap)*
   `(3,)` is 1D. `(3, 1)` is a 2D column vector. `(1, 3)` is a 2D row vector. They contain the same values but behave differently under broadcasting and matmul.

5. **Why does `np.array([True, 6])` give `[1, 6]`?**
   Type priority: bool is silently upgraded to int (True→1, False→0). The whole array becomes `int`.

6. **`arr[1, 2]` vs `arr[1][2]` — same result?** *(adapted from `alexeygrigorev/data-science-interviews`)*
   Same value, different mechanism. `arr[1, 2]` is one C call. `arr[1][2]` is two operations (row 1, then element 2). The single-bracket form is the idiom and is faster.

[🔝 Back to top](#top)

---

<a id="2-module2"></a>
## 2. Module 2 — Filtering, Reshape, Aggregation, Matrix Multiply

> Notebook 2 — `np.arange`, fancy indexing, boolean masking, 2D reshape, axis/aggregations, `np.where`/`any`/`all`, sorting, element-wise vs matrix multiplication.

### 🪜 Mental model

**Axis-that-disappears.** When you write `arr.sum(axis=k)`, axis `k` is the dimension that *collapses*. `axis=0` (rows axis) disappears → you're left with one number per column → "per-column" results. `axis=1` (cols axis) disappears → per-row results. Memorize one and the other is just the opposite. *The axis you pass is the axis you kill.*

### 🪞 Basic → Intermediate → Advanced — boolean masking

**Basic** — a boolean array selects elements where True.
```python
arr[arr > 500]                       # keep elements > 500
```

**Intermediate** — masks from one array can filter another, as long as lengths match.
```python
costs[votes >= 500]                  # cost of restaurants with ≥500 votes
```

**Advanced** — compound conditions need `&` / `|` / `~` (bitwise) and parentheses around each comparison. Using `and`/`or` raises `ValueError`.
```python
arr[(arr > 0) & (arr < 5)]           # right
arr[arr > 0 and arr < 5]             # ValueError: truth value of array is ambiguous
```

### 🪞 Basic → Intermediate → Advanced — `reshape` & `-1`

**Basic** — restructure same data into a new shape; element count must match.
```python
np.arange(12).reshape(3, 4)          # (12,) → (3, 4)
```

**Intermediate** — `-1` means "infer this dim from the size."
```python
np.arange(12).reshape(-1, 4)         # NumPy fills in 3
np.arange(12).reshape(3, -1)         # NumPy fills in 4
```

**Advanced** — `reshape` is usually a view, but **only when strides are compatible**. After a transpose (C → F order) it must copy. Only one `-1` per call.
```python
arr.T.reshape(...)                   # may silently copy — check arr.flags
np.arange(12).reshape(-1, -1)        # ValueError — multiple -1 not allowed
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| `np.arange` | Like Python `range`, **end is excluded**. Supports floats. |
| Boolean mask | `arr[arr > 500]` → keep elements where True |
| Fancy index | `arr[[2, 5, 7]]` → pick specific positions. Returns **copy** |
| Aligned filter | `costs[votes >= 500]` — arrays must be same length |
| `reshape(r, c)` | Restructure; element count must match. **`-1`** infers dim |
| Reshape vs transpose | Reshape changes structure; transpose swaps axes |
| Axis mantra (**"C, R"**) | `axis=0` collapses rows → **C**olumn-wise. `axis=1` → **R**ow-wise |
| `np.any` / `np.all` | Reduce to single bool (or per-axis if `axis=` given) |
| `np.where(cond, a, b)` | If-else for arrays. With only `cond`: returns **indices** |
| `np.sort` | Sorted copy. `arr.sort()` is in-place |
| `np.argsort` | **Indices** that would sort the array |
| Element-wise `*` | Same shape (or broadcastable) |
| `np.dot` | Matrix mult; supports scalar too |
| `np.matmul` / `@` | Strict matmul; no scalar |
| Matmul rule | (m, k) × (k, n) → (m, n) |

### ⚙️ Top APIs

```python
np.arange(start, stop, step)
np.column_stack((a, b))               # 1D cols → 2D matrix
arr.reshape(r, c) / arr.reshape(-1, c) / arr.reshape(r, -1)
arr.sum/mean/max/min/std(axis=0 or 1)
np.any(cond, axis=...), np.all(cond, axis=...)
np.where(cond, val_true, val_false)
np.where(cond)                        # → indices
np.sort(arr, axis=...), np.argsort(arr)
arr[::-1]                             # reverse (descending after sort)
np.dot(A, B), np.matmul(A, B), A @ B
```

### 🧩 Code patterns

```python
# 1. Filter + aligned filter
high_votes = votes[votes >= 500]
costs_of_high_votes = costs[votes >= 500]

# 2. Reshape 1D → 2D and aggregate
data = np.arange(10, 110, 10).reshape(5, -1)
data.sum(axis=0)                      # per-column sums
data.mean(axis=1, keepdims=True)      # per-row mean, shape preserved

# 3. Conditional labelling
labels = np.where(ratings >= 4.2, "Green Flag", "Red Flag")

# 4. Multi-condition mask
costs[(votes > 200) & (costs < 600)]

# 5. Sort + descending
np.sort(votes)[::-1]                  # descending
data[data[:, 0].argsort()]            # sort 2D rows by col 0

# 6. Matrix multiply for composite scores
score = data @ weights                # (n, 3) @ (3,) → (n,)
```

### 🎯 Advanced Q&A — Module 2

> Mix of original drills and questions adapted from `rougier/numpy-100`, `alexeygrigorev/data-science-interviews`, and `andrewekhalel/MLQuestions`.

1. **Why is the **end** value of `np.arange` excluded?** *(common Python-convention question)*
   Same convention as Python `range` and Python slicing — half-open intervals make length arithmetic clean (`stop - start = count` for step 1). For inclusive ranges, use `np.linspace(start, stop, n)`.

2. **What does `arr.reshape(-1, 2)` mean exactly?**
   "Give me 2 columns; infer the number of rows from the size." NumPy computes the `-1` dimension automatically. Only one `-1` allowed per call.

3. **Why does `axis=0` give per-column results when "axis 0 = rows"?**
   The axis you pass is the one that **disappears**. `axis=0` collapses the row axis, so what survives is one value per column. Memory aid: "axis 0 disappears → columns remain → per-column."

4. **For a 3D `(3, 4, 5)` array, what's the shape of `arr.sum(axis=(1, 2))`?**
   `(3,)`. Axes 1 and 2 both collapse, leaving axis 0.

5. **`np.where(cond)` vs `np.where(cond, a, b)` — different return types?**
   With three args: an array same shape, with `a` where True and `b` where False. With one arg: a **tuple of index arrays** showing positions where `cond` is True (equivalent to `np.nonzero(cond)`).

6. **`np.sort(arr)` vs `arr.sort()` — what's the difference?**
   `np.sort` returns a new sorted array. `arr.sort()` sorts **in place** and returns `None`. Don't write `x = arr.sort()` — it'll be `None`.

7. **How do you sort 2D rows by one column without losing row alignment?**
   ```python
   sorted_rows = data[data[:, 0].argsort()]
   ```
   `argsort` gives the permutation; you apply it to the whole row range.

8. **What's the difference between `np.dot`, `np.matmul`, and `@`?**
   `@` is syntactic sugar for `np.matmul`. `np.matmul` and `@` strictly do matrix multiplication. `np.dot` does the same for 2D inputs but also handles scalar/vector cases that `matmul` rejects. For new code, prefer `@`.

9. **`sorted(['10', '2', '1', '20'])` — what's the output and why?**
   `['1', '10', '2', '20']`. Lexicographic (string) sort, not numeric. Cast to int first: `sorted(lst, key=int)` or `np.sort(np.array(lst).astype(int))`.

10. **Will `np.any(arr > 0)` return an array or a scalar?**
    A **scalar** bool by default. Pass `axis=` to get a per-axis result.

[🔝 Back to top](#top)

---

<a id="3-module3"></a>
## 3. Module 3 — Broadcasting, Vectorization, Stack & Split

> Notebook 3 — element-wise vs matmul recap, **vectorization** (`np.vectorize`), **broadcasting** (4 rules), `np.tile`, **splitting** (`split`/`hsplit`/`vsplit`), **stacking** (`vstack`/`hstack`/`concatenate`).

### 🪜 Mental model

**Right-align shapes.** For broadcasting, line up the two shape tuples by their **rightmost** dimension and compare position-by-position. Each pair must either be **equal** or one must be **1**. Missing left-side dimensions are silently treated as 1. That's the whole rule — the rest is bookkeeping. Visually:

```
A: (5, 2)        A: (5, 2)         A: (5, 2)
B:    (2,)   ✓   B: (5, 1)    ✓    B: (5,)     ✗ (2 vs 5, neither is 1)
```

### 🪞 Basic → Intermediate → Advanced — broadcasting

**Basic** — match a smaller shape against a larger one with no copy.
```python
np.array([1, 2, 3]) + 10             # (3,) + scalar → (3,)
```

**Intermediate** — center a 2D matrix by subtracting the per-column mean.
```python
data - data.mean(axis=0)             # (n, k) - (k,) → (n, k)
```

**Advanced** — per-row centering needs `keepdims=True` (or `[:, None]`); otherwise the shape collapses and broadcasting fails.
```python
data - data.mean(axis=1, keepdims=True)   # (n, k) - (n, 1) → OK
data - data.mean(axis=1)                  # (n, k) - (n,)   → ValueError
data - data.mean(axis=1)[:, None]         # also works
```

### 🪞 Basic → Intermediate → Advanced — stack vs concatenate

**Basic** — `concatenate` joins along an **existing** axis.
```python
np.concatenate([a, b], axis=0)       # add rows; cols must match
```

**Intermediate** — `vstack`/`hstack` are shortcuts; `column_stack` promotes 1D arrays to columns.
```python
np.vstack([a, b])                    # same cols required
np.column_stack([votes, ratings])    # two 1D → 2D table
```

**Advanced** — `stack` creates a **NEW** axis, so all inputs must have *identical* shape. Common confusion: you want `stack` for "5 same-shape images → tensor of shape `(5, H, W, 3)`", not `concatenate`.
```python
np.stack([img1, img2, img3], axis=0) # (3, H, W, 3) — new axis 0
np.concatenate([img1, img2], axis=0) # (2H, W, 3)  — existing axis 0
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Vectorization | Apply a function across an array in one call (in C, ideally) |
| `np.vectorize` | Convenience wrapper — **not** true speed-up; loops in Python |
| Broadcasting | Auto-stretch smaller array to match larger; **no copy** |
| Broadcasting rules (4) | See below |
| `np.tile` | Explicit replication along given axes |
| `np.split(arr, N, axis)` | Even split — errors on uneven |
| `np.array_split` | Same idea but tolerates uneven counts |
| `np.hsplit` / `vsplit` | Shortcut for `split(axis=1)` / `split(axis=0)` |
| `np.vstack` | Row stack — same number of **cols** required |
| `np.hstack` | Column stack — same number of **rows** required |
| `np.concatenate` | Glue along an **existing** axis |
| `np.stack` | Glue along a **new** axis |

### 📐 The 4 broadcasting rules

1. If arrays don't have the same number of dimensions, **prepend 1s** to the smaller shape.
2. Dimensions are compatible if they're **equal** or one of them is **1**.
3. Output shape is the **max** along each dimension.
4. If neither rule 2 holds nor padding fixes it, **broadcasting fails**.

Visualize from the **right**:

```
A: (5, 2)
B:    (2,)        ← right-align; matches → OK → result (5, 2)

A: (5, 2)
B: (5, 1)         ← 1 stretches → OK → result (5, 2)

A: (5, 2)
B: (3,)           ← mismatch → ERROR
```

### ⚙️ Top APIs

```python
np.vectorize(fn)
np.tile(arr, reps)
np.split(arr, N, axis=0), np.hsplit(arr, N), np.vsplit(arr, N)
np.array_split(arr, N, axis=0)               # tolerates uneven
np.vstack([a, b]), np.hstack([a, b])
np.concatenate([a, b], axis=0)               # existing axis
np.stack([a, b], axis=0)                     # new axis
np.column_stack([a, b])                      # 1D → cols
```

### 🧩 Code patterns

```python
# 1. Broadcasting — subtract per-column mean
data - data.mean(axis=0)                     # (n, k) - (k,) → broadcasts

# 2. Per-row mean with keepdims
data - data.mean(axis=1, keepdims=True)      # (n, k) - (n, 1)

# 3. Tile manually if broadcasting won't help
np.tile(arr, (3, 1))                         # repeat 3× along axis 0

# 4. Vectorize a Python function (convenience, not speed)
@np.vectorize
def label(rate): return "High" if rate >= 4 else "Low"
labels = label(ratings)

# 5. Split data into batches
batches = np.array_split(data, 5, axis=0)    # 5 batches; uneven OK

# 6. Stack — append new restaurants (rows)
np.vstack([restaurants_df, new_rows])
```

### 🎯 Advanced Q&A — Module 3

> Mix of original drills and questions adapted from `chiphuyen/ml-interviews-book`, `rougier/numpy-100`, and `khanhnamle1994/cracking-the-data-science-interview`.

1. **Why isn't `np.vectorize` true vectorization?** *(common performance trap)*
   It loops in Python under the hood — it's a *convenience* wrapper for elementwise application. Real vectorization means a single C/SIMD call. For speed, replace `np.vectorize` with native NumPy ops (`np.where`, masking) or write a true ufunc with Numba.

2. **You have `(5, 2)` and want to subtract the per-row mean — why does `keepdims=True` matter?**
   Without it, `data.mean(axis=1)` returns shape `(5,)`. Broadcasting `(5, 2) - (5,)` fails (right-align gives `(2,)` vs `(5,)`). With `keepdims=True` you get `(5, 1)`, which broadcasts cleanly against `(5, 2)`.

3. **Does broadcasting copy memory?**
   No — it's "virtual" stretching via stride=0 tricks. The output is allocated; the broadcast itself doesn't duplicate input.

4. **`np.split` vs `np.array_split`?**
   `split` requires the array to divide evenly into N pieces and raises otherwise. `array_split` allows uneven splits — last chunks absorb the remainder. For training batches of arbitrary size, use `array_split`.

5. **`np.concatenate` vs `np.stack` — when which?**
   `concatenate` joins along an **existing** axis (shapes match on every other axis). `stack` creates a **new** axis (all inputs must have identical shape). If you want to "add a row to a 2D table" use `concatenate`/`vstack`. If you want to stack 5 same-shaped images into a `(5, H, W, 3)` tensor, use `stack`.

6. **For `np.vstack([a, b])`, what shape constraint must hold?**
   Same number of **columns** (last axis). 1D arrays are promoted to row vectors first.

7. **When would broadcasting *fail* on `(5, 2)` and `(5,)`?**
   Right-align: `(5, 2)` vs `(5,)` → compare `2` with `5` — mismatch, neither is 1. Fails. To fix, reshape `(5,)` to `(5, 1)` with `[:, None]`.

8. **What's a "broadcast view"?**
   You can create a virtual broadcast with `np.broadcast_to(arr, shape)` — same data, expanded view. Useful for read-only computations on a stretched shape without paying memory.

[🔝 Back to top](#top)

---

<a id="4-pandas-moved"></a>
## 4. Pandas — moved to its own guide

The Pandas module (Series, DataFrame, `iloc`/`loc`, dtype cleanup, `.str` accessor, etc.) is now maintained separately so each library can be drilled in isolation.

👉 **See [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md)** for cheat sheets, top APIs, gotchas, advanced Q&A, and a 50-question drill — all focused on Pandas.

For applied / project-level pandas (joins, groupby, apply, reshape, datetime, plotting), use [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md).

[🔝 Back to top](#top)

---

<a id="5-conceptmap"></a>
## 5. Cross-module concept map

How the three NumPy notebooks build on each other (Pandas lives in its own guide):

```
Module 1: WHAT IS DATA?
  └── ML motivation → EDA → why NumPy → ndarray basics
       └── shape, dtype, type priority, astype
            └── indexing & slicing (with view/copy hint)

Module 2: WHAT CAN I DO TO ONE ARRAY?
  ├── filter (boolean mask, fancy)
  ├── reshape (-1 trick, axis intuition)
  ├── aggregate (sum/mean/std along axis)
  ├── condition (np.where, any/all)
  ├── sort (sort/argsort)
  └── multiply (elementwise vs dot/matmul/@)

Module 3: WHAT CAN I DO WITH MULTIPLE ARRAYS?
  ├── broadcast (4 rules, no-copy stretching)
  ├── vectorize (np.vectorize convenience)
  ├── split (split/array_split/hsplit/vsplit)
  └── stack (vstack/hstack/concatenate/stack)

Next step: REAL TABULAR DATA → Pandas_Revision_Guide.md
```

[🔝 Back to top](#top)

---

<a id="6-anchors"></a>
## 6. The 5 mental anchors (memorize these)

Five examples that, if internalized, give you ~70% of NumPy interviews.

1. **`[1,2,3] * 2` → `[1,2,3,1,2,3]` but `np.array([1,2,3]) * 2` → `[2,4,6]`.**
   This is why every data scientist uses NumPy.

2. **Tea-room analogy.** Lists scatter ingredients across the room (pointer-chase). NumPy lines them up on one shelf (contiguous memory) — the entire perf story.

3. **The race ends at the finish line, not after it.** `arr[0:5]` is 5 elements (indices 0–4). End is **exclusive**.

4. **"Axis that disappears."** For aggregations, `axis=k` is the dimension that *collapses*. `axis=0` collapses rows → one value per column.

5. **Right-align shapes for broadcasting.** `(5, 2)` vs `(2,)` → align right → matches → broadcast OK.

If you can teach a friend each of these in 60 seconds, you've internalized the foundation.

[🔝 Back to top](#top)

---

<a id="7-zomato"></a>
## 7. Zomato dataset cheat sheet

> The cleanup steps below use Pandas. For the underlying API mechanics (`pd.to_numeric`, `.str` accessor, `replace`, etc.) see [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md).

Columns you'll see across the notebooks and what to remember about them:

| Column | What it is | Cleaning needed |
|---|---|---|
| `name` | Restaurant name | None |
| `location` | Area in Bangalore (e.g., Koramangala) | Standardize case |
| `cuisines` | Comma-separated cuisines | Split / count |
| `rate` / `Ratings` | Rating like `"4.1/5"` or `"NEW"` | Strip `/5`, replace `NEW`→NaN, `to_numeric` |
| `approx_cost(for two people)` | Cost like `"1,200"` | Remove comma, `astype(float)` |
| `votes` | Integer count | Usually clean |
| `listed_in(type)` | Delivery / Dine-out / etc. | None |

Common questions answered with this data:
- Top areas by listing count → `df['location'].value_counts()`
- Top cuisines → split `cuisines` then count
- Rating vs cost → scatter / correlation
- Best-value restaurants → high rating ÷ cost
- Most-popular restaurant → max votes

[🔝 Back to top](#top)

---

<a id="8-businessmap"></a>
## 8. Common business questions → which API

> Most of these use Pandas. NumPy is involved when you compose with `np.log`/`np.where` or hand work off to ndarrays. For the full Pandas business map see [`Pandas_Revision_Guide.md` §8](./Pandas_Revision_Guide.md#8-businessmap).

| Business question | API |
|---|---|
| "Which restaurants are above 4★?" | `df[df['rating'] > 4]` |
| "Top 10 locations by restaurant count?" | `df['location'].value_counts().head(10)` |
| "Cost per person?" | `df['cost_for_two'] / 2` |
| "Average rating per cuisine?" | `df.groupby('cuisines')['rating'].mean()` |
| "Filter high-vote and low-cost?" | `df[(df['votes']>500) & (df['cost']<500)]` |
| "Replace 'NEW' ratings with NaN?" | `df['rating'].replace('NEW', np.nan)` |
| "How many distinct locations?" | `df['location'].nunique()` |
| "Sort by votes descending?" | `df.sort_values('votes', ascending=False)` |
| "Composite score = rating × log(votes)?" | `df['score'] = df['rating'] * np.log(df['votes'])` |
| "Save cleaned data?" | `df.to_csv('clean.csv', index=False)` |

[🔝 Back to top](#top)

---

<a id="9-terms"></a>
## 9. 📚 Master terms glossary

All key terms across the four notebooks — alphabetical for quick lookup.

| Term | Definition |
|---|---|
| `argsort` | Returns the indices that would sort an array |
| Aligned filter | Filtering one array by a boolean mask derived from another (same length) |
| Axis | The dimension along which an op runs. `axis=0` rows-collapse, `axis=1` cols-collapse |
| Boolean mask | Array of True/False used to select elements |
| Broadcasting | Auto shape-alignment for math on different-shaped arrays |
| Coercion | Forcing one type into another (e.g., `errors='coerce'` makes bad values NaN) |
| Contiguous memory | One continuous RAM block — NumPy's secret weapon |
| Copy | New buffer; mutations don't affect the source |
| Determinant | Scalar from a square matrix; 0 = singular |
| `dtype` | Element type (`int64`, `float64`, `<U7`, `object`) |
| EDA | Exploratory Data Analysis — understand the data first |
| Fancy indexing | Selecting by a list/array of positions; returns **copy** |
| Matrix multiplication | (m,k)×(k,n)→(m,n); use `@` |
| `ndarray` | NumPy's n-dimensional array — the core data structure |
| NaN | Floating-point "not a number" sentinel |
| Ravel | 1D view-when-possible; `flatten` always copies |
| Reshape | Same data, new shape; element count must match |
| SIMD | Single Instruction Multiple Data — CPU parallelism NumPy uses |
| Slicing | `start:end:step`; end exclusive; returns **view** |
| Sort | Ordered values; `argsort` gives ordered indices |
| Stack | Combine arrays along a **new** axis |
| Concatenate | Combine arrays along an **existing** axis |
| Stride | Bytes to step per axis — under the hood of views/transpose |
| Type priority | String > Float > Int > Bool |
| Vectorization | Apply op across an array in one C call |
| View | Window into the same memory; mutations propagate |
| `np.where` | Conditional replacement or index lookup |

[🔝 Back to top](#top)

---

<a id="10-apis"></a>
## 10. ⚙️ API cheat sheet — every method, one table

The single table to scan five minutes before any interview.

### NumPy creation
| Call | Returns |
|---|---|
| `np.array(list)` | ndarray from a list |
| `np.zeros(shape)` / `np.ones(shape)` | Filled with 0 / 1 |
| `np.full(shape, v)` | Filled with v |
| `np.arange(s, e, step)` | Like range; end exclusive; floats OK |
| `np.linspace(s, e, n)` | N evenly spaced points; end inclusive |
| `np.eye(n)` | Identity matrix |
| `np.empty(shape)` | Uninitialized — fastest |
| `np.random.rand/randn/randint` | Uniform / normal / int |
| `np.random.seed(42)` | Reproducibility |

### NumPy attributes
| Attribute | Meaning |
|---|---|
| `.shape` | Tuple of dims |
| `.ndim` | Number of dims |
| `.size` | Total elements |
| `.dtype` | Element type |
| `.itemsize` / `.nbytes` | Bytes per element / total |
| `.T` | Transpose view |
| `.flags` | C/F-contiguous, writeable, etc. |

### NumPy ops
| Call | Purpose |
|---|---|
| `arr.astype(t)` | Convert dtype (copy) |
| `arr[bool_mask]` | Boolean filter (copy) |
| `arr[[i, j]]` | Fancy index (copy) |
| `arr[i:j:k]` | Slice (view) |
| `arr.reshape(r, c)` / `(-1, c)` | Restructure |
| `arr.flatten()` / `arr.ravel()` | 1D copy / 1D view-if-possible |
| `arr.sum/mean/max/min/std(axis=k)` | Aggregate |
| `np.where(cond, a, b)` / `np.where(cond)` | If-else / indices |
| `np.any/all(cond, axis=k)` | Reduce to bool |
| `np.sort(arr) / arr.sort()` | Sorted copy / in-place |
| `np.argsort(arr)` | Permutation indices |
| `A * B` / `A @ B` | Elementwise / matmul |
| `np.dot(A, B)` / `np.matmul(A, B)` | Matmul (dot also for scalars) |
| `np.tile(arr, reps)` | Replicate |
| `np.split / hsplit / vsplit / array_split` | Split (last tolerates uneven) |
| `np.vstack / hstack / concatenate / stack` | Combine |
| `np.column_stack` | 1D arrays → columns |
| `np.clip(arr, lo, hi)` | Bound into a range |
| `np.percentile / np.quantile` | Distribution stats |
| `np.isnan / np.nan*` | NaN detection / NaN-aware aggregations |
| `np.linalg.norm/solve/inv/det/eig/svd` | Linear algebra |
| `np.save / np.load / np.savez` | Binary I/O |

> **Pandas APIs?** See [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) for the dedicated cheat sheet (creation, I/O, selection, info, cleanup).

[🔝 Back to top](#top)

---

<a id="11-gotchas"></a>
## 11. ⚠️ Gotchas & traps (all in one place)

The full collection — re-read this before any technical interview.

### NumPy

1. **`arr == np.nan` is always False.** Use `np.isnan(arr)`.
2. **`astype(int)` truncates, doesn't round.** `1.9 → 1`. Use `np.round(arr).astype(int)`.
3. **`astype` returns a new array.** Assign it back: `arr = arr.astype(float)`.
4. **A single string in a mixed list coerces the whole array to strings** — silent and severe.
5. **`np.arange` end is exclusive.** And it can drift with floats — prefer `np.linspace` for floats.
6. **Reshape needs the element count to match exactly.** Otherwise: `ValueError`.
7. **Axis sense is reversed from intuition.** `axis=0` aggregates **down** rows → one per column.
8. **Slicing returns a view; fancy/boolean indexing returns a copy.** Mutations behave differently. When in doubt, `.copy()`.
9. **`&`/`|`/`~` for arrays, `and`/`or`/`not` for scalars.** Wrap each comparison in parens: `(a > 0) & (a < 5)`.
10. **`A * B` is elementwise, `A @ B` is matmul.** Conflating them is a silent bug.
11. **`np.dot` allows scalar/vector tricks `np.matmul` doesn't.** Stick to `@` for clarity.
12. **`np.vectorize` is not fast** — it's a convenience wrapper that loops in Python.
13. **`np.split` errors on uneven splits; use `np.array_split` for tolerant splits.**
14. **`vstack` needs same columns; `hstack` needs same rows.**
15. **`mean()` on a NumPy array containing NaN returns NaN.** Use `np.nanmean`.
16. **Broadcasting failure example:** `(5, 2) - (5,)` fails — reshape `(5,)` to `(5, 1)` first.
17. **`keepdims=True` is what lets a reduction be subtracted back from the original.**
18. **Integer overflow with small dtypes.** `int8` wraps at 127.
19. **Transpose is a view, not a copy** — strides are swapped, no data move.

> **Pandas gotchas?** See [`Pandas_Revision_Guide.md` §6](./Pandas_Revision_Guide.md#6-gotchas) for the dedicated list (`SettingWithCopyWarning`, `.iloc`/`.loc` mismatches, dtype inference, NaN handling, etc.).

[🔝 Back to top](#top)

---

<a id="12-advanced"></a>
## 12. 🎯 Advanced interview Q&A

Cross-module questions a senior interviewer would actually ask. Read the question, formulate an answer, then peek.

### Performance & internals

**Q1. Walk me through three reasons NumPy is faster than a Python loop, ranked by impact.**
(1) **Vectorized C kernels with SIMD** — biggest factor; one C call replaces N Python opcodes. (2) **Contiguous memory** — cache-friendly streaming, no pointer-chase. (3) **Homogeneous dtype** — no per-element type dispatch. Bonus: NumPy **releases the GIL** during C ops, allowing thread-level parallelism.

**Q2. What's a stride and why does that explain why transpose is free?**
Strides are the byte offsets to step one element along each axis. NumPy's array is a buffer + shape + strides. Transposing only swaps the `strides` tuple — no data movement. That's why `.T` is O(1).

**Q3. When does `reshape` return a copy?**
When the desired strides aren't expressible on the existing buffer — most commonly after a transpose (C-contiguous → F-contiguous). NumPy falls back to allocating a new contiguous buffer and copying.

**Q4. `np.vectorize` — is it true vectorization?**
No. It's a thin Python loop with array I/O. For real speed, use ufuncs (`np.where`, masking, arithmetic) or write a compiled kernel with Numba/Cython.

### View vs copy

**Q5. Predict the output and explain.**
```python
arr = np.array([1., 2., 3., 4., 5.])
sub = arr[1:4]
sub[0] = 999
print(arr)
```
Answer: `[1., 999., 3., 4., 5.]`. Basic slicing returns a **view** — mutating the view mutates the source.

**Q6. Same setup but with `sub = arr[[1, 2, 3]]`?**
Fancy indexing returns a **copy**. The original is unchanged.

**Q7. How do you check programmatically if `b` is a view of `a`?**
`b.base is a` (or `np.shares_memory(a, b)`).

### Broadcasting

**Q8. Centre a `(5, 2)` array — subtract per-column mean.**
`data - data.mean(axis=0)` → `(5,2) - (2,)` → broadcasts cleanly.

**Q9. Subtract per-row mean from `(5, 2)`.**
`data - data.mean(axis=1, keepdims=True)` → `(5,2) - (5,1)` → broadcasts. Without `keepdims`, you'd get `(5,)` which doesn't broadcast against `(5, 2)`.

**Q10. Outer product without `np.outer`?**
```python
a[:, None] * b[None, :]      # (m,1) × (1,n) → (m,n)
```

### Numerical stability & algorithmic

**Q11. Why might `np.linalg.inv(A) @ b` be a bad idea even if mathematically equivalent to `solve(A, b)`?**
`inv` is slower (LU is one-shot) and *numerically less stable* — it amplifies floating-point error. `np.linalg.solve(A, b)` is the production choice.

**Q12. Sort a 2D NumPy array by column 0, keep rows together.**
```python
data[data[:, 0].argsort()]
```

**Q13. Compute moving average of length 3 on a 1D array (no Pandas).**
```python
np.convolve(arr, np.ones(3)/3, mode='valid')
```

**Q14. How would you bucket continuous values into [low, medium, high]?**
```python
np.select(
    [arr < 200, arr < 600],
    ['low', 'medium'],
    default='high',
)
```

**Q15. Compute pairwise Euclidean distances between rows of `(n, d)` matrix without a Python loop.**
```python
diff = X[:, None, :] - X[None, :, :]       # (n, n, d)
dist = np.sqrt((diff**2).sum(axis=-1))     # (n, n)
```

> **Pandas Q&A?** See [`Pandas_Revision_Guide.md` §7](./Pandas_Revision_Guide.md#7-advanced) for `iloc`/`loc`, `SettingWithCopyWarning`, dtype coercion, cleanup pipelines, etc.

[🔝 Back to top](#top)

---

<a id="sourced-bank"></a>
## 🌐 Sourced interview questions

> **Real questions paraphrased from canonical interview-prep sources.** Use this as your standalone practice bank — no internet required. Each batch keeps the source's original numbering for traceability.

### Batch 1 — from [`rougier/numpy-100`](https://github.com/rougier/numpy-100) (★ easy / ★★ medium / ★★★ hard)

| # | Question | Answer |
|---|---|---|
| 1 (★, #3) | Create a zero-filled array of size 10. | `np.zeros(10)` |
| 2 (★, #4) | Determine memory size of an array. | `arr.nbytes` or `arr.size * arr.itemsize` |
| 3 (★, #6) | Vector of length 10 with the 5th element set to 1. | `z = np.zeros(10); z[4] = 1` |
| 4 (★, #7) | Vector with values 10–49. | `np.arange(10, 50)` |
| 5 (★, #8) | Reverse a vector. | `arr[::-1]` |
| 6 (★, #9) | 3×3 matrix with values 0–8. | `np.arange(9).reshape(3, 3)` |
| 7 (★, #10) | Find indices of non-zero elements in `[1,2,0,0,4,0]`. | `np.nonzero([1,2,0,0,4,0])` → `(array([0,1,4]),)` |
| 8 (★, #11) | 3×3 identity matrix. | `np.eye(3)` |
| 9 (★, #13) | Min/max of a 10×10 random array. | `a.min(), a.max()` |
| 10 (★, #15) | 2D array with 1 on borders, 0 inside. | `a = np.ones((n,n)); a[1:-1, 1:-1] = 0` |
| 11 (★, #17) | What do `0 * np.nan` and `np.nan == np.nan` return? | `nan` and `False` — NaN propagates and is never equal to itself |
| 12 (★, #23) | Define a custom dtype for RGBA. | `np.dtype([('r','u1'),('g','u1'),('b','u1'),('a','u1')])` |
| 13 (★, #24) | Real matrix product `(5,3) × (3,2)`. | `A @ B` |
| 14 (★, #25) | Negate elements between 3 and 8 in a 1D array, in-place. | `arr[(arr>=3) & (arr<=8)] *= -1` |
| 15 (★, #29) | Round away from zero (not toward). | `np.where(arr >= 0, np.ceil(arr), np.floor(arr))` |
| 16 (★, #30) | Common values between two arrays. | `np.intersect1d(a, b)` |
| 17 (★★, #44) | Convert Cartesian to polar coordinates. | `r = np.hypot(x, y); θ = np.arctan2(y, x)` |

### Batch 2 — from [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) (NumPy-relevant theory)

| # | Question | One-liner answer |
|---|---|---|
| 18 | Difference between a Python list and a NumPy array? | NumPy = contiguous memory + homogeneous dtype + vectorized C/SIMD → 10–100× faster on numeric work. |
| 19 | Why is broadcasting essentially free in memory? | Stride trick — the smaller array is "stretched" via a stride of 0 on the broadcast axis; no copy. |
| 20 | View vs copy — when does each occur? | Basic slicing → **view** (shared memory). Fancy/boolean indexing → **copy**. Verify with `np.shares_memory(a, b)`. |

### Batch 3 — common FAANG/senior NumPy patterns

| # | Question | One-liner answer |
|---|---|---|
| 21 | Compute all pairwise Euclidean distances of an `(n, d)` matrix without a Python loop. | `np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))` |
| 22 | Why prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`? | Faster (one LU factorization) and *numerically more stable* — `inv` amplifies floating-point error. |
| 23 | Add a new axis to a 1D vector for broadcasting. | `arr[:, None]` → column; `arr[None, :]` → row. |
| 24 | How to bucket continuous values into `[low, medium, high]` without `pd.cut`? | `np.select([arr<200, arr<600], ['low','medium'], default='high')` |
| 25 | Why does `arr.sum(axis=0)` give per-column results? | The axis you pass is the axis that **collapses** — `axis=0` collapses rows → one value per column. |

### Citations & where to drill more
- 🎯 [`rougier/numpy-100`](https://github.com/rougier/numpy-100) — full 100 drills with graded difficulty.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — theory + coding.
- 🎯 [`kojino/120-Data-Science-Interview-Questions`](https://github.com/kojino/120-Data-Science-Interview-Questions) — broader DS bank.

[🔝 Back to top](#top)

---

<a id="13-drill"></a>
## 13. 🔁 NumPy revision drill (70 questions)

Designed as a **timed pre-interview tool**. Read each question, answer in your head, peek. Aim for under 15 seconds per question — finish all 70 in under 20 minutes. For Pandas drill, see [`Pandas_Revision_Guide.md` §9](./Pandas_Revision_Guide.md#9-drill).

### Block A — NumPy basics (Q1–25)

1. NumPy stands for? → **Numerical Python**
2. Core data structure? → **ndarray**
3. Reasons NumPy is faster (3)? → **Contiguous memory + homogeneous dtype + vectorized C/SIMD**
4. `[1,2,3] * 2` in Python? → **`[1,2,3,1,2,3]`**
5. Same with `np.array`? → **`[2,4,6]`**
6. Type priority? → **String > Float > Int > Bool**
7. Shape of 1D 10-element array? → **`(10,)`**
8. `dtype='<U5'` means? → **Unicode string max 5 chars**
9. Bytes for `int64`? → **8**
10. `arr.size` returns? → **Total element count**
11. Last element? → **`arr[-1]`**
12. Reverse array? → **`arr[::-1]`**
13. 2D index pattern? → **`arr[row, col]`**
14. Shape of `arr[:, 0]` for `(m, n)`? → **`(m,)`**
15. Shape of `arr[:, 0:1]`? → **`(m, 1)`**
16. Slicing end inclusive? → **No — exclusive**
17. Basic slicing returns? → **View**
18. Fancy indexing returns? → **Copy**
19. Boolean indexing returns? → **Copy**
20. Force a copy? → **`.copy()`**
21. Check if view? → **`b.base is a`**
22. `astype(int)` does? → **Truncate**
23. Convert string `"1,200"` → number? → **`replace(',', '').astype(float)`**
24. `np.array([True, 6])` dtype? → **int**
25. Modify original via `astype`? → **No, new array**

### Block B — NumPy intermediate (Q26–50)

26. `np.arange` end inclusive? → **No**
27. Reshape with `-1` means? → **Infer that dim**
28. Element count must match in reshape? → **Yes**
29. Axis mantra? → **C, R — axis=0 collapses rows (per col); axis=1 collapses cols (per row)**
30. `axis=0` on `(3,4)` sum gives shape? → **`(4,)`**
31. `axis=1` on `(3,4)` sum gives shape? → **`(3,)`**
32. `np.any(cond)` returns? → **Scalar bool (no axis)**
33. `np.all` purpose? → **All elements satisfy condition?**
34. `np.where(c, a, b)` does? → **If-else, elementwise**
35. `np.where(c)` only? → **Indices where True**
36. `np.sort` returns? → **Sorted copy**
37. `arr.sort()` returns? → **None — in place**
38. `np.argsort` returns? → **Permutation indices**
39. Descending sort? → **`np.sort(arr)[::-1]`**
40. Sort 2D by col 0? → **`data[data[:,0].argsort()]`**
41. `A * B` vs `A @ B`? → **Elementwise vs matmul**
42. `np.dot` vs `np.matmul`? → **dot allows scalar; matmul strict**
43. Matmul shape rule? → **(m,k) × (k,n) → (m,n)**
44. Mask filter syntax? → **`arr[arr > x]`**
45. Mask combine? → **`(a > 0) & (a < 5)`** with parens
46. Why parens? → **`&` has higher precedence than `>`**
47. Mask AND/OR? → **`&`, `|`, `~`**
48. Aligned filter? → **`b[mask_from_a]` works if same length**
49. `np.column_stack` purpose? → **Stack 1D as cols of 2D**
50. `sorted(['10','2','1'])`? → **Lexicographic, not numeric**

### Block C — Broadcasting, vectorize, stack/split (Q51–70)

51. Broadcasting rule (4)? → **(1) Prepend 1s. (2) Equal or 1. (3) Max per dim. (4) Else fail**
52. Right-align shapes? → **Yes — start from the right**
53. `keepdims=True` purpose? → **Keep size-1 axis after reduction**
54. Broadcast `(5,2) - (2,)`? → **OK → `(5,2)`**
55. Broadcast `(5,2) - (5,)`? → **Fails — reshape to `(5,1)`**
56. Memory cost of broadcasting? → **None — virtual stretch**
57. `np.tile` purpose? → **Explicit replication**
58. `np.vectorize` truly vectorized? → **No — Python loop wrapper**
59. `np.split` even? → **Must divide evenly or errors**
60. `np.array_split`? → **Tolerates uneven**
61. `hsplit` shortcut? → **`split(axis=1)`**
62. `vsplit` shortcut? → **`split(axis=0)`**
63. `vstack` constraint? → **Same columns**
64. `hstack` constraint? → **Same rows**
65. `np.concatenate` what axis? → **Existing axis**
66. `np.stack` what axis? → **NEW axis**
67. Outer product? → **`a[:, None] * b[None, :]`**
68. Stretch via stride trick? → **`np.broadcast_to(arr, shape)`**
69. Two arrays for matmul shapes? → **Inner dims must match**
70. Composite score formula? → **`data @ weights`**

**Score yourself**: 63+ = strong, 53–62 = solid, 42–52 = revise, <42 = re-read modules.

> **Continue with Pandas?** [`Pandas_Revision_Guide.md` §9 drill](./Pandas_Revision_Guide.md#9-drill) — 50 more questions on Series, DataFrame, `iloc`/`loc`, cleanup, and traps.

[🔝 Back to top](#top)

---

<a id="14-bestpractices"></a>
## 14. ✅ Best practices

Crystallized NumPy do's and don'ts. For Pandas best practices, see [`Pandas_Revision_Guide.md` §10](./Pandas_Revision_Guide.md#10-bestpractices).

### Performance

1. **Vectorize.** Replace Python loops with array ops. Even `np.where` beats a `for`.
2. **Preallocate, don't append.** `np.empty(N)` + index assignment, not repeated `np.append`.
3. **Right dtype.** `int32`/`float32` halves memory if range/precision fits.
4. **Use `@` for matmul** for clarity; reserve `np.dot` for the scalar edge cases.
5. **`np.linalg.solve(A, b)`** over `np.linalg.inv(A) @ b` — faster, more stable.
6. **`np.empty` over `np.zeros`** when you'll overwrite everything anyway.

### Correctness

7. **Mind the view-vs-copy boundary.** `.copy()` when unsure; verify with `np.shares_memory`.
8. **`np.isnan` not `== np.nan`** (NaN is never equal to anything).
9. **`keepdims=True`** when broadcasting a reduction back.
10. **Wrap mask conditions in parens:** `(a > 0) & (a < 5)`.
11. **Match `*` for elementwise, `@` for matmul.** Silent bugs otherwise.

### Workflow

12. **Save intermediate data** with `np.save` — recomputation is wasteful.
13. **Set a `np.random.seed`** when generating reproducible test arrays.

### Interview-day reminders

14. **Walk through dimensions out loud.** "Shape `(5, 2)` — 5 rows, 2 columns."
15. **Right-align shapes** for broadcasting questions — on paper if needed.
16. **Axis mnemonic: "axis 0 disappears = one per column"**.
17. **For view/copy: basic slice = view, fancy/boolean = copy.** Memorize the trio.
18. **When unsure, show you'd test it.** Quoting `np.shares_memory(a, b)` or `arr.base` signals senior judgment.

[🔝 Back to top](#top)

---

<a id="15-mapping"></a>
## 15. 📦 What's in each notebook (mapping)

| Notebook | Title | Covers | This guide |
|---|---|---|---|
| 1 | Food Delivery Data Exploration and analysis 1 | ML motivation, EDA, lists vs arrays, why NumPy fast, array creation, shape/dtype, type coercion, astype, indexing, slicing | §1, plus the full [NumPy_EDA_Interview_Prep_Guide](./Food%20Delivery%20Data%20Exploration%20and%20analysis%201/NumPy_EDA_Interview_Prep_Guide.md) |
| 2 | Food Delivery Data Exploration and analysis 2 | `np.arange`, fancy indexing, boolean masking, 2D reshape, axis & aggregations, `np.where`/`any`/`all`, sort/argsort, matrix multiply | §2 |
| 3 | Food Delivery Data Exploration and Analysis 3 | Vectorization, broadcasting (4 rules), `np.tile`, `split`/`hsplit`/`vsplit`/`array_split`, `vstack`/`hstack`/`concatenate`/`stack` | §3 |
| 4 | Food Delivery Data Exploration and analysis 4 | Pandas intro — Series, DataFrame, `read_csv`, `.iloc`/`.loc`, `set_index`/`reset_index`, `.info()`/`.describe()`, column ops, rename, `astype`, `pd.to_numeric`, `.str` accessor, `replace`, `unique`/`nunique`/`value_counts`, `to_csv` | → [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) |
| 5 | Amazon sales data analysis 1 | Joins (`pd.merge`, inner/outer/left/right), groupby (single/multi key, agg), filter/apply on groups, sorting, duplicates, `.concat`, `.apply` (row/column/lambda), `.isnull().sum()` | → [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) §1 |
| 6 | Amazon sales data analysis 2 | Missing values (`isna`/`dropna`/`fillna`), `pd.melt`/`pivot_table`, `pd.cut` binning, `.str.contains`/`.str.extract`, `pd.to_datetime` & `.dt` accessor, univariate viz (hist/count) | → [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) §2 |
| 7 | Amazon sales data analysis 3 | Bivariate viz (scatter/line/box/violin/grouped bar), multivariate viz (pairplot/heatmap/stacked), `.corr()`, hue & subplots | → [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) §3 |
| 8 | Analyzing Sachin Tendulkar's ODI Career | Probability fundamentals (sample space, events, set ops), addition/multiplication/complement rules, marginal vs joint, conditional probability, Bayes' Theorem — computed empirically via pandas boolean filtering | → [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) §4 |

[🔝 Back to top](#top)
