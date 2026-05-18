<a id="top"></a>
# Data Foundation — NumPy Master Revision Guide

> **Consolidated quick-revision notes for the three NumPy notebooks** (Food Delivery EDA 1–3). Built for fast review and advanced interview prep. Every concept, every API, every gotcha — in scannable form, with deep Q&A and a revision drill at the end.

**Companion guides (Pandas pulled out for focused study):**
- 🐼 [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) — Series, DataFrame, `iloc`/`loc`, dtype cleanup, `.str` accessor, etc.
- 🛒 [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) — applied pandas (joins, groupby, apply, reshape, datetime), visualization, and probability for the Amazon + Sachin notebooks.

**How to use:**
- **First-time learning a concept:** open the module's **📖 Guided concept walkthrough** ([M1](#1g-guided) · [M2](#2g-guided) · [M3](#3g-guided)). Each concept is introduced *what → why → how → where → related → code → gotcha* — no follow-up search needed.
- **Pre-interview revision:** topic finder → cheat sheet → Q&A.
- **Just before a coding round:** run the [§13 Revision Drill](#13-drill).
- **Quick term lookup:** [§9 Master glossary](#9-terms) — every term has a 2–4 sentence beginner-friendly definition.
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
| 📖 First-time intro to a concept (what / why / how / where / related) | [M1 walkthrough](#1g-guided), [M2 walkthrough](#2g-guided), [M3 walkthrough](#3g-guided) |
| ML, EDA, lists vs arrays, why NumPy fast | [Module 1](#1-module1) → [walkthrough](#1g-guided) |
| Array creation, shape, dtype, indexing, slicing, astype, view-vs-copy | [Module 1](#1-module1) → [walkthrough](#1g-guided) / [cheat](#1c-cheat) |
| Boolean masking, fancy indexing, reshape, axis, aggregation | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| `np.where`, `np.any`/`all`, `np.sort`/`argsort`, matrix multiply | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| Vectorization, broadcasting (4 rules), `np.tile` | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
| `split`/`hsplit`/`vsplit`, `vstack`/`hstack`/`concatenate`/`stack` | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
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

1. [Module 1 — NumPy Foundation](#1-module1) · [📖 Guided walkthrough](#1g-guided) · [🧠 Cheat sheet](#1c-cheat)
2. [Module 2 — Filtering, Reshape, Aggregation, Matrix Multiply](#2-module2) · [📖 Guided walkthrough](#2g-guided)
3. [Module 3 — Broadcasting, Vectorization, Stack & Split](#3-module3) · [📖 Guided walkthrough](#3g-guided)
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

<a id="1g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 1 concept. Read this top-to-bottom on a first pass; the cheat sheet below is the recap.

#### Machine Learning (ML)

> **🪜 Mental model:** *Teaching by example.* Instead of writing rules by hand, you show a program thousands of past examples and let it find the rules itself.

**What it is.** Machine Learning is a way of building software where, rather than a programmer writing the rules ("if email contains 'lottery' → spam"), an algorithm **learns the rules from data**. You feed it many labelled examples (emails marked spam or not), and it produces a model that can label new, unseen emails. ML is a subfield of Artificial Intelligence (AI); the term *learning* specifically means "improving performance on a task as more data is seen."

**Why it matters.** Most interesting real-world problems (image recognition, fraud detection, recommendations, search ranking) have rules too complicated to write by hand. ML lets us solve them by collecting examples instead of writing logic. In an interview, "why ML over rule-based code" is a recurring opener — the honest answer is *"because the rules are unknown, change over time, or are too many to enumerate."*

**How it works.**
1. **Collect data** — input examples (`X`) and the right answers (`y`).
2. **Choose a model family** — e.g., linear regression, decision tree, neural network.
3. **Train** — the algorithm tunes internal numbers (parameters/weights) to fit the examples by minimising a *loss* (a measure of how wrong its predictions are).
4. **Evaluate** on data the model has *not* seen, to check it actually generalises.
5. **Predict** — use the trained model on brand-new inputs.

**Where it's used.** Spam filters, Netflix/YouTube recommendations, credit-card fraud detection, voice assistants, ChatGPT-style large language models, self-driving perception, medical imaging. In this repo, every later module (Supervised Learning, NLP, CV, GenAI) is a specific kind of ML.

**Related terms.**
- **AI** — the broader umbrella; ML is one approach to AI.
- **Supervised learning** — ML with labelled examples (the most common kind).
- **Unsupervised learning** — finding structure (e.g., clusters) without labels.
- **Deep learning** — ML using deep neural networks; the workhorse for images/text.
- **Model** — the trained "rule-finder" that ML produces.

**Gotcha.** ML is *not* magic — bad data in → bad model out. Most real-world ML time is spent on data cleaning and EDA, not on tuning the model.

#### Exploratory Data Analysis (EDA)

> **🪜 Mental model:** *Meet the dataset before you marry it.* Look at, summarise, and visualise the data before doing anything fancy with it.

**What it is.** EDA is the **first hands-on phase** of any data project: you load the raw data, peek at samples, summarise each column, check for missing values, plot distributions, and form initial hypotheses. The goal is to understand *what you have* — types, ranges, quirks, outliers — before you build any model on top of it. The term was coined by John Tukey in the 1970s.

**Why it matters.** Skipping EDA is the most common reason models fail in production. You miss that a column secretly contains the string `"NEW"` mixed with numbers, or that 30% of rows have a missing target, or that the test set was accidentally leaked into training. EDA catches all of this *cheaply*, before you've spent days on a model. Interviewers ask about EDA precisely because it separates people who model real data from those who only do textbook problems.

**How it works.** A typical pass goes:
1. **Shape & dtypes** — `df.shape`, `df.info()`.
2. **Sample inspection** — `df.head()`, `df.sample(5)`.
3. **Per-column summary** — `df.describe()` for numerics, `df['col'].value_counts()` for categories.
4. **Missing values** — `df.isna().sum()`.
5. **Distributions** — histograms for numerics, bar charts for categories.
6. **Relationships** — scatter plots, group-by aggregations, correlation heatmaps.
7. **Clean as you go** — fix dtypes, drop bad rows, document oddities.

**Where it's used.** Every notebook in this repo opens with EDA. In FAANG interviews, a "data analysis case study" is *almost always* an EDA prompt in disguise — "you're given the Zomato dataset, find the top 5 areas by restaurant count" is an EDA question.

**Related terms.**
- **Data cleaning** — the part of EDA where you fix what's broken.
- **Feature engineering** — *after* EDA: building new columns the model can use.
- **Outlier** — a value far from the rest; EDA is where you find it.
- **Missing value (NaN)** — empty cell; EDA decides whether to drop, fill, or flag it.

**Gotcha.** EDA isn't "look once and move on." On real datasets you loop back to EDA *after* every modelling surprise.

#### `ndarray` — the NumPy array

> **🪜 Mental model:** *One shelf of identical jars.* All elements live in one continuous strip of memory and all have the same type — that's why operations are fast.

**What it is.** An `ndarray` is NumPy's core data structure: an **N-dimensional array** of values with a *single* element type (`dtype`). Under the hood it's a triple — a flat memory buffer + a `shape` tuple (how it's organised into dimensions) + `strides` (how many bytes to step to move along each axis). 1D `ndarray` ≈ vector, 2D ≈ matrix, 3D+ ≈ tensor.

**Why it matters.** Almost every numerical Python library (Pandas, scikit-learn, PyTorch, TensorFlow) either *is* built on ndarrays or speaks the same shape/dtype dialect. If you understand ndarrays, you understand 80% of how Python ML libraries move data around.

**How it works.** When you write `arr = np.array([1, 2, 3])`:
1. NumPy allocates one contiguous block of memory (here, 3 × 8 bytes for `int64`).
2. It stores `shape=(3,)`, `dtype=int64`, and `strides=(8,)` ("step 8 bytes to reach the next element").
3. Any operation (`arr + 1`, `arr.sum()`) is dispatched to a precompiled C function that streams through the buffer.

**Where it's used.** As the input to literally every scikit-learn model (`fit(X, y)` expects `X` to be an ndarray or convertible). As the under-the-hood storage of every Pandas column. As the GPU-side data structure (with tweaks) in PyTorch tensors.

**Related terms.**
- **Vector / matrix / tensor** — 1D / 2D / N-D arrays in math language.
- **`shape`** — tuple of dimensions; the array's "size in each direction."
- **`dtype`** — element type (see below).
- **Stride** — bytes to step per axis; how NumPy walks the memory.
- **Tensor** (in PyTorch/TF) — basically an ndarray that can also live on a GPU.

**Gotcha.** Once you create an ndarray, its `dtype` is fixed — you can't store a string in an int array; NumPy will silently coerce or raise.

#### `shape`, `ndim`, `size`

> **🪜 Mental model:** *The label on the jar.* `shape` tells you the dimensions; `ndim` how many dimensions; `size` how many elements total.

**What it is.** Three attributes every ndarray exposes:
- **`shape`** — a tuple like `(rows, cols)` describing dimensions. `(5,)` is 1D with 5 elements; `(5, 2)` is 5 rows × 2 cols.
- **`ndim`** — the number of dimensions (= `len(shape)`).
- **`size`** — the total number of elements (= product of shape entries).

**Why it matters.** Almost every NumPy bug — broadcasting failures, matmul mismatches, reshape errors — traces back to a shape misunderstanding. Saying shapes out loud while reading code is the single highest-leverage habit you can build.

**How it works.** These are O(1) lookups on the array's metadata (no scan of the data). So `arr.shape` on a 10-million-element array still returns instantly.

**Where it's used.** In every error message NumPy raises ("could not broadcast `(5,2)` against `(5,)`"). In matmul rules ("inner dims must match"). In reshape (`reshape(-1, 3)` works only if `size` is divisible by 3).

**Related terms.**
- **Axis** — a single dimension; `axis=0` is the first dim of `shape`.
- **1D / 2D / N-D** — informal names for arrays of `ndim` 1, 2, N.
- **`(n,)` vs `(n, 1)`** — 1D vector vs 2D column vector; behave differently under broadcasting.

**Gotcha.** `(3,)` (1D) and `(3, 1)` (2D column) print the same numbers but are different shapes — and matmul/broadcasting treats them differently.

#### `dtype` and type priority

> **🪜 Mental model:** *Everyone in the jar must be the same flavour.* If you put in mixed flavours, NumPy promotes everything to the most powerful flavour present.

**What it is.** `dtype` (data type) declares the element type of an array — common ones are `int64`, `float64`, `bool`, `<U7` (Unicode string up to 7 chars), and `object` (Python objects, opt-out from speed). When you create an array from a mixed list, NumPy chooses **one** dtype that can hold every element, using a priority order: **string > float > int > bool**.

**Why it matters.** dtype determines memory cost (`int8` = 1 byte, `float64` = 8 bytes), what operations are legal (you can't `arr.mean()` a string array), and whether you accidentally end up with an `object` array (which silently loses all speed). One stray string in a numeric column will quietly turn the whole array into strings.

**How it works.** Each NumPy ufunc (universal function) has separate C implementations for each dtype. When you call `arr + 1`, NumPy looks up the version compiled for `arr.dtype`. If two arrays have different dtypes, NumPy "promotes" both to a common type before computing.

**Where it's used.** Every Pandas column has a dtype (inherited from NumPy plus a few extras like `datetime64`). Every neural-network library cares deeply about whether your tensors are `float32` vs `float64`. EDA always starts with `df.info()` to see dtypes.

**Related terms.**
- **Type coercion / promotion** — automatic upcast to a common dtype.
- **`astype(t)`** — explicit conversion, returns a new array.
- **`object` dtype** — fallback dtype for Python objects; slow and to be avoided.
- **`NaN`** — only legal in `float` dtype, not `int`.

**Gotcha.** `np.array([1, 2, 'oops']).dtype` is `<U21` — the whole array is now strings, and `+`/`mean` will fail or behave bizarrely.

#### `astype` — explicit conversion

> **🪜 Mental model:** *Pour into a different jar.* `astype` returns a new array of the requested dtype; it never modifies the original.

**What it is.** `arr.astype(dtype)` produces a **new** array where every element has been converted to the requested dtype. For float → int it **truncates** (chops the decimal, does not round). For numeric → string it serializes. For string → numeric it raises if any element doesn't parse.

**Why it matters.** Real-world data is rarely the right dtype on arrival ("1,200" instead of `1200`, `"4.1/5"` instead of `4.1`). `astype` is the standard cleanup step. The truncation-vs-rounding gotcha is a perennial interview question and a common source of off-by-one bugs.

**How it works.** NumPy allocates a new buffer of the right size for the target dtype, walks the source array, and converts element-by-element using the C-level cast rules. The result is independent memory — mutating it does not affect the source.

**Where it's used.** Right after `pd.to_numeric` to lock in a precise dtype. Before feeding data to scikit-learn (which usually wants `float64`). To shrink memory: `arr.astype('float32')` halves memory if you can afford the precision.

**Related terms.**
- **`np.round` + `astype(int)`** — the correct way to round-then-cast.
- **`pd.to_numeric(..., errors='coerce')`** — pandas equivalent for messy strings (bad values → NaN).
- **Downcasting** — using a smaller dtype (`int32`) to save memory.

**Gotcha.** `astype(int)` on `1.9` returns `1`, not `2`. To round properly: `np.round(arr).astype(int)`. Also: `astype` returns a new array — `arr.astype(float)` alone changes nothing; you must reassign.

#### Indexing & slicing

> **🪜 Mental model:** *The race ends at the finish line, not after it.* Slice end is exclusive — `arr[0:5]` is 5 elements (indices 0–4), not 6.

**What it is.** Indexing pulls out a single element (`arr[3]`); slicing pulls out a contiguous range (`arr[1:5]`) or a stride (`arr[::2]`). For 2D arrays, the idiomatic form is `arr[row, col]` (comma-separated), which is one C call. Negative indices count from the end (`arr[-1]` is the last element).

**Why it matters.** Indexing is how you extract every subset, every filter, every "first 100 rows" you'll ever write. Getting the **end-exclusive** rule wrong causes off-by-one bugs; mixing up `arr[r][c]` and `arr[r, c]` works but the latter is faster and more idiomatic.

**How it works.** Basic slicing (`arr[i:j:k]`) returns a **view**: a new array object pointing at the *same* memory with different shape/strides. No data is copied. 2D `arr[r, c]` uses the array's strides to jump directly to the requested element in one step.

**Where it's used.** Every train/test split (`X[:80]`, `X[80:]`). Every "first N rows," every "last M columns." Every Pandas `.iloc` under the hood.

**Related terms.**
- **View vs copy** — see below; basic slice = view.
- **Fancy indexing** — `arr[[2, 5, 7]]`; returns a *copy*.
- **Boolean indexing** — `arr[arr > 0]`; also returns a *copy*.
- **Negative indices** — `arr[-1]` is last element.

**Gotcha.** Slicing end is **exclusive**: `arr[0:5]` gives indices 0,1,2,3,4 — five elements. People learning Pandas later get burned when `.loc[0:5]` *includes* index 5 (because `.loc` is label-based and inclusive).

#### View vs copy

> **🪜 Mental model:** *View = window into the same room. Copy = a photo of the room.* Editing the view changes what everyone sees; editing the photo doesn't.

**What it is.** A NumPy *view* is a new array object that shares its memory buffer with the original — mutations propagate both ways. A *copy* is an independent buffer — mutations don't affect the source. The rule:
- **Basic slicing** (`arr[i:j]`, `arr[:, 0]`) → **view**.
- **Fancy indexing** (`arr[[1, 2, 3]]`) → **copy**.
- **Boolean indexing** (`arr[mask]`) → **copy**.
- Explicit `arr.copy()` → **copy**.

**Why it matters.** Confusing view and copy is the #1 source of "mysterious mutation" bugs — you write `sub = arr[1:4]; sub[0] = 999` thinking you've made a local change, but `arr` now also has a 999 in it. In interviews, "predict the output of this slice-then-mutate snippet" is a common probe of whether you've actually used NumPy seriously.

**How it works.** Basic slicing only needs to record new shape/strides/offset pointing into the same buffer — that's why it's free. Fancy/boolean indexing picks *arbitrary* positions, which can't be expressed as stride math, so NumPy materialises a new buffer.

**Where it's used.** Memory-sensitive code (large arrays) often deliberately uses views to avoid copying. Pandas' `SettingWithCopyWarning` is the higher-level version of this same trap.

**Related terms.**
- **`b.base is a`** — checks if `b` is a view of `a` (returns `True` for views).
- **`np.shares_memory(a, b)`** — robust check for any memory overlap.
- **`.copy()`** — explicitly force a copy when you want isolation.
- **`SettingWithCopyWarning`** — the Pandas-side cousin of this gotcha.

**Gotcha.** When in doubt, `.copy()`. The performance cost of an unnecessary copy is small; the debugging cost of an accidental view is high.

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
### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#1g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **ML** | Building rules from examples instead of writing them by hand. A trained model maps inputs `X` to predicted outputs `ŷ`. | Whenever the rules are too many, too unknown, or changing too fast to enumerate manually (spam, fraud, recommendations). |
| **EDA** | The first hands-on pass over a dataset — shape, dtypes, missing values, distributions, quick plots. | Every new dataset, before *any* modelling. Without it you'll model garbage. |
| **List vs array** | Python list scatters objects via pointers and overloads `*` as repeat (`[1,2]*2 → [1,2,1,2]`). NumPy array lives in one contiguous buffer and treats `*` as elementwise math (`np.array([1,2])*2 → [2,4]`). | Any numeric work — use arrays. Lists are fine for mixed-type / variable-length collections. |
| **Why NumPy is fast** | Three reasons: contiguous memory (cache-friendly), homogeneous dtype (no per-element type check), vectorized C kernels with SIMD. | Always — this is the answer to "why use NumPy" in every interview. |
| **`ndarray`** | NumPy's core N-dimensional array — flat buffer + `shape` + `strides` + one `dtype`. | The default container for any numeric data; underlies every Pandas column and scikit-learn input. |
| **`shape`** | Tuple of dimensions, e.g. `(5, 2)` = 5 rows × 2 cols. `(5,)` is 1D, `(5, 1)` is 2D column. | Every broadcasting/matmul/reshape question. Say shapes out loud. |
| **`ndim`** | Number of dimensions = `len(shape)`. | Quick sanity check; `arr.ndim == 1` distinguishes vector from matrix. |
| **`size`** | Total element count = product of shape entries. | When checking memory cost or whether a reshape will fit. |
| **`dtype`** | Element type — `int64`, `float64`, `bool`, `<U7`, `object`. Fixed once the array is created. | Always check after loading messy data — wrong dtype = wrong behaviour. |
| **Type priority** | When mixing types, NumPy picks one dtype that holds all: **string > float > int > bool**. | Explains "why did my int array become strings?" — one stray string promoted everything. |
| **`astype`** | Returns a **new** array of a different dtype. Float→int **truncates** (does not round). | Cleanup after parsing (`replace(',', '').astype(float)`); shrinking memory (`astype('float32')`). |
| **Indexing** | `arr[i]` (single), `arr[-1]` (negative from end), `arr[r, c]` (2D), `arr[[i,j,k]]` (fancy). Comma form is one C call; faster and idiomatic. | Pulling individual elements or arbitrary positions. |
| **Slicing** | `arr[start:end:step]`. End is **exclusive**. Returns a **view** of the same memory. | Train/test splits, "first N rows," any contiguous range. |
| **View vs copy** | Basic slice = view (shared memory, mutations propagate). Fancy/boolean indexing = copy (independent). | When you're about to mutate — `.copy()` if unsure. |
| **Reverse** | `arr[::-1]` — step of `-1` walks backward. Returns a view. | Descending sorts (`np.sort(arr)[::-1]`), reversing time series. |

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

<a id="2g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 2 concept. The cheat sheet below is the recap surface.

#### `np.arange` — array version of `range`

> **🪜 Mental model:** *Same as Python `range`, but it produces an array — and the end value is excluded.*

**What it is.** `np.arange(start, stop, step)` returns a 1D ndarray of evenly spaced values from `start` (inclusive) to `stop` (exclusive), incrementing by `step`. Unlike Python's `range`, all three arguments can be floats. The output dtype is inferred from the inputs (int → `int64`, float → `float64`).

**Why it matters.** `np.arange` is the workhorse for generating index arrays, x-axis values for plots, and toy data for examples. The **end-exclusive** convention matches Python slicing and is consistent across the NumPy world.

**How it works.** NumPy precomputes the number of elements as `ceil((stop - start) / step)`, allocates a buffer that size, and fills it. For float steps the result can be one element off due to floating-point drift — that's why for floats you should prefer `np.linspace(start, stop, n)`, which lets you specify the *count* exactly.

**Where it's used.** Index generators: `np.arange(n)` → `[0, 1, …, n-1]`. Plot x-axes. Reshape sources: `np.arange(12).reshape(3, 4)`. Time series helpers.

**Related terms.**
- **`np.linspace(s, e, n)`** — `n` evenly spaced points; end is **inclusive**. Use for plotting/float ranges.
- **Python `range`** — same idea but returns a lazy iterator, not an array.
- **`np.arange(n)`** — the most common one-arg form; equivalent to `np.arange(0, n, 1)`.

**Gotcha.** Floats + `arange` can produce surprising counts: `np.arange(0, 1, 0.1)` gives 10 elements, but the last is `0.9` not `1.0`, and rounding can occasionally give 11. Use `np.linspace` when you care about the count or the endpoint.

#### Boolean masking

> **🪜 Mental model:** *A True/False stencil.* Lay an array of booleans over your data; only elements above a True are kept.

**What it is.** A *boolean mask* is a boolean array the same length as the data you're filtering. `arr[mask]` returns a new array containing only the elements where `mask` is True. The mask itself is usually built from a comparison: `arr > 500` produces a boolean array element-wise.

**Why it matters.** This is the canonical "filter rows of data" operation in NumPy and Pandas. It's vectorised (no Python loop), readable (looks like the math), and combines with `&` / `|` / `~` for multi-condition filtering. In interviews, "give me all rows where X > 500 and Y < 200" expects boolean masking, not a `for` loop.

**How it works.** `arr > 500` creates a temporary boolean ndarray by walking `arr` in C. `arr[mask]` then walks `mask`, copying matching elements from `arr` into a new buffer. Note: boolean indexing always returns a **copy**, not a view.

**Where it's used.** Filtering: `arr[arr > 0]`. Aligned filtering: `costs[votes >= 500]` — masks built on one array can filter another, as long as lengths match. In-place updates: `arr[arr < 0] = 0` (clamp negatives to zero).

**Related terms.**
- **Fancy indexing** — sibling technique that picks by *positions* (`arr[[1, 3, 5]]`) instead of by a True/False mask.
- **`np.where(cond, a, b)`** — pick *between two values* by condition, instead of just keeping the True ones.
- **Aligned filter** — using a mask from array A to filter array B (requires equal length).
- **`&` / `|` / `~`** — bitwise AND / OR / NOT for combining masks (note: parens are required around each comparison).

**Gotcha.** You **must** use `&`, `|`, `~` (bitwise) for masks, not `and`, `or`, `not`. And every comparison must be parenthesised: `(a > 0) & (a < 5)` — without parens, `&` binds tighter than `>` and you get nonsense.

#### Fancy indexing

> **🪜 Mental model:** *Pick specific seats by row number.* Pass a list of positions; NumPy gives you back exactly those elements, in that order.

**What it is.** Fancy indexing is selecting elements by an *array of positions* — `arr[[2, 5, 7]]` returns elements at indices 2, 5, and 7, in that order. For 2D arrays, you can fancy-index rows (`arr[[0, 2], :]`) or both axes (`arr[[0, 1], [3, 4]]` picks two specific cells).

**Why it matters.** Fancy indexing is how you reorder, sample, or gather scattered elements without a loop. It's what powers `arr[argsort_indices]` (re-arrange after sorting), random sampling, and arbitrary lookups. The fact that fancy indexing returns a **copy** (not a view) is a frequent interview question.

**How it works.** NumPy walks the index array, looks up each position in the source array, and writes the result into a new buffer. Because the indices can be arbitrary (non-contiguous, repeated), there's no stride math that can describe the result — so it has to materialise a copy.

**Where it's used.** Sorting rows by a column: `data[data[:, 0].argsort()]`. Random batches: `data[np.random.permutation(n)[:32]]`. Lookup tables: `vocab_embeddings[token_ids]`.

**Related terms.**
- **Basic slicing** — sibling that uses `start:end:step`; returns a *view*, not a copy.
- **Boolean masking** — sibling that uses True/False, not positions.
- **`np.take(arr, indices)`** — explicit function-form of fancy indexing.

**Gotcha.** Repeated indices are allowed (`arr[[1, 1, 1]]` returns three copies of `arr[1]`). And the result is always a fresh copy — mutating it does not change the source.

#### `reshape` and the `-1` trick

> **🪜 Mental model:** *Same data, different box.* Reshape changes the shape tuple; the underlying values are unchanged. `-1` means "you figure this dimension out for me."

**What it is.** `arr.reshape(new_shape)` returns an array with the same elements but a new shape. The total element count must stay constant: a `(12,)` array can become `(3, 4)` or `(2, 6)` but not `(3, 5)`. You can replace **one** dimension with `-1` and NumPy will infer it from the size: `arr.reshape(-1, 4)` says "4 columns, you compute the rows."

**Why it matters.** Reshape is the bridge between 1D pipelines (`np.arange`, file reads) and 2D/3D structures (matrices, images). The `-1` trick lets you write reshape code that survives batch-size changes — you don't have to recompute the row count by hand.

**How it works.** If the new shape can be expressed with new strides over the *same* memory buffer, reshape returns a **view** (no copy). If not — most commonly after a transpose, when the data is no longer C-contiguous — NumPy falls back to allocating a new contiguous buffer and copying.

**Where it's used.** 1D → 2D for plotting/matmul: `np.arange(10).reshape(2, 5)`. Flattening: `arr.reshape(-1)` to go N-D → 1D. Image batching: `images.reshape(-1, 28*28)` to flatten each image in a batch. Adding a feature dimension: `x.reshape(-1, 1)` for scikit-learn (which needs 2D `X`).

**Related terms.**
- **`ravel`** — view-when-possible 1D flattening (cheaper than `flatten`).
- **`flatten`** — always-copy 1D flattening.
- **`np.newaxis` / `None`** — `arr[:, None]` adds a size-1 axis (alternative to `reshape(-1, 1)`).
- **Transpose `.T`** — swaps axes; followed by reshape may force a copy.

**Gotcha.** Only **one** `-1` per call. `reshape(-1, -1)` raises a `ValueError`. And reshape after transpose may silently copy — check `arr.flags['C_CONTIGUOUS']` if you care.

#### Axis — the dimension that disappears

> **🪜 Mental model:** *The axis you pass is the axis you kill.* `arr.sum(axis=0)` collapses axis 0; what survives is "one value per remaining axis."

**What it is.** Most aggregations (`sum`, `mean`, `max`, `std`, `argmax`, etc.) accept an `axis=` argument. The axis you pass is the one that **collapses** to a single value. For a 2D array of shape `(rows, cols)`:
- `axis=0` collapses **rows** → one number per column → result shape `(cols,)` → "per-column" results.
- `axis=1` collapses **cols** → one number per row → result shape `(rows,)` → "per-row" results.
- No axis → reduce everything to a scalar.

**Why it matters.** Axis is the single most confusing thing in NumPy for beginners — and it shows up in every aggregation, every reshape, every concat. Internalise the mantra "axis kills the axis you pass" and you'll never re-derive it.

**How it works.** Under the hood, NumPy iterates over all index combinations *except* the named axis, runs the reduction along that axis for each combination, and stores the result in an array with that axis removed.

**Where it's used.** Per-column statistics: `data.mean(axis=0)`. Per-row totals: `data.sum(axis=1)`. Argmax over classes (predicted label per sample): `probs.argmax(axis=1)`. Most Pandas operations also accept `axis=` and follow the same convention.

**Related terms.**
- **`keepdims=True`** — keeps the collapsed axis as size 1 (`(n, 1)` instead of `(n,)`); useful for broadcasting the reduction back.
- **Reduction** — generic name for "operation that collapses an axis."
- **`np.argmax(axis=k)`** — returns *indices* of the max along axis `k`.

**Gotcha.** "axis=0 = rows" sounds like it should give per-row results, but the opposite is true — axis=0 *removes* the row axis, leaving you with per-column results. The mantra fixes it: "the axis you pass is the axis that disappears."

#### `np.where` — vectorised if/else

> **🪜 Mental model:** *Array-friendly if-statement.* At each position, pick value A if the condition is true, value B if false — in one call, no loop.

**What it is.** `np.where(cond, a, b)` walks `cond` element-by-element and, at each position, picks the value from `a` (if True) or `b` (if False). The result is a new array of the same shape. With **only one argument** — `np.where(cond)` — it instead returns the **indices** of True positions (a tuple of index arrays, one per dimension).

**Why it matters.** Whenever you'd write a Python loop with a per-element `if/else`, `np.where` does it 10–100× faster and in one line. It's the canonical "create a label column" tool: `np.where(rating >= 4, 'good', 'bad')`. The single-argument form is a separate idiom (basically `np.nonzero(cond)`) and confusing the two is a popular interview trap.

**How it works.** NumPy broadcasts `cond`, `a`, and `b` to a common shape, then runs a compiled C loop that picks per element. With one argument, it scans `cond` and collects the True positions into index arrays.

**Where it's used.** Labelling: `np.where(score > 0.5, 1, 0)`. Imputation: `np.where(np.isnan(x), median, x)`. Conditional clipping: `np.where(arr < 0, 0, arr)`. Finding positions: `np.where(arr == target)[0]`.

**Related terms.**
- **`np.select`** — multi-branch version (more than two outcomes).
- **`np.clip`** — shortcut for "clamp to [lo, hi]."
- **Boolean masking** — keeps only matching elements (different goal — pick *one or the other*, vs. *keep or drop*).
- **`pd.Series.where`** — pandas variant; *keeps* where True, replaces where False — opposite default from `np.where`.

**Gotcha.** Single-argument `np.where(cond)` returns **indices**, not values, as a *tuple*. Most code wants `np.where(cond)[0]` to unpack the index array.

#### `np.any` / `np.all`

> **🪜 Mental model:** *"Is there any?" vs "are they all?"* Reduces a boolean array to a single yes/no — or one per axis if you pass `axis=`.

**What it is.** `np.any(cond)` returns `True` if **any** element is True; `np.all(cond)` returns `True` only if **every** element is True. With `axis=k`, the reduction happens along axis `k`, yielding a per-axis answer.

**Why it matters.** They're the standard "fast check" operations — "do any rows contain a NaN?" (`np.isnan(data).any()`), "are all elements positive?" (`(arr > 0).all()`). Idiomatic, fast, and short-circuit-friendly under the hood.

**How it works.** Walks the boolean array, OR-ing (`any`) or AND-ing (`all`) elements. With an axis argument, performs the reduction along that axis, leaving the others.

**Where it's used.** Data validation: `df.isna().any()` flags columns with any missing values. Sanity checks before/after a transformation. Equality checks for unit tests: `(arr1 == arr2).all()`.

**Related terms.**
- **`np.isnan(arr).any()`** — common NaN-detection idiom.
- **`(a == b).all()`** — common array-equality idiom (don't use `==` directly because that returns an array).
- **`np.count_nonzero(cond)`** — counts True elements, instead of just yes/no.

**Gotcha.** With no `axis=`, the result is a scalar bool. Beginners try `if arr:` on a multi-element array — Python can't determine the truthiness of a whole array and raises `ValueError: The truth value of an array... is ambiguous`. Use `.any()` or `.all()` explicitly.

#### `np.sort` and `np.argsort`

> **🪜 Mental model:** *Sort gives you the sorted values; argsort gives you the indices that would sort them.* Argsort is the more useful one when you need to keep rows aligned.

**What it is.** `np.sort(arr)` returns a new sorted array (ascending by default). `arr.sort()` sorts **in-place** and returns `None`. `np.argsort(arr)` returns the **indices** that would sort `arr` — i.e., `arr[np.argsort(arr)]` equals `np.sort(arr)`.

**Why it matters.** Sorting values alone is rarely enough — you usually need to keep an *aligned* table sorted by one column. That's where `argsort` shines: `data[data[:, 0].argsort()]` sorts the whole 2D array by column 0 while keeping rows intact. Asking when to use `sort` vs `argsort` is a classic interview probe.

**How it works.** Default algorithm is quicksort (`kind='quicksort'`). For stable sorts (preserving relative order of equal elements), use `kind='stable'`. Both are O(n log n).

**Where it's used.** Top-k: `arr.argsort()[-k:]` gives indices of the k largest. Sorting a DataFrame on a column without using pandas: `data[data[:, col].argsort()]`. Building rankings: `ranks = np.empty_like(idx); ranks[idx] = np.arange(len(idx))`.

**Related terms.**
- **`np.argmax` / `np.argmin`** — index of the single max / min (faster than full sort when you only need one).
- **`np.partition`** — partial sort; cheaper if you only need the top-k.
- **Stable sort** — `kind='stable'` keeps equal elements in their original order.
- **Descending sort** — there's no built-in flag; use `np.sort(arr)[::-1]` or `arr.argsort()[::-1]`.

**Gotcha.** `arr.sort()` returns `None` because it sorts in-place. Writing `x = arr.sort()` makes `x` be `None` — a common bug for newcomers.

#### Matrix multiplication — element-wise vs `@`/`dot`/`matmul`

> **🪜 Mental model:** *`A * B` is "multiply each cell"; `A @ B` is "row dot column."* They look similar in code but compute totally different things.

**What it is.** Three multiplication concepts in NumPy:
- **Element-wise (`A * B`)** — same shape (or broadcastable); each cell is `A[i, j] * B[i, j]`.
- **Matrix multiplication (`A @ B`)** — inner dimensions must match; `(m, k) @ (k, n) → (m, n)`. Each output cell is a dot product of an A-row with a B-column.
- **`np.dot(A, B)`** — for 2D inputs it's the same as `@`. It also accepts scalar/vector mixes that `@` doesn't.
- **`np.matmul(A, B)`** — function form of `@`. Strict matrix multiplication; rejects scalars.

**Why it matters.** The single most common silent bug in numerical code is writing `*` when you meant `@` (or vice versa). They both run — they just compute the wrong thing. ML models, linear regression, neural networks, attention layers — all are essentially long chains of matrix multiplications.

**How it works.** For `(m, k) @ (k, n)`, the output has shape `(m, n)`. Each `out[i, j] = sum(A[i, :] * B[:, j])` — i.e., row *i* of A dot-producted with column *j* of B. NumPy dispatches to a BLAS library (highly optimised C/Fortran) for speed.

**Where it's used.** Linear regression: `y_hat = X @ w`. Neural network layers: `h = X @ W + b`. Composite scores: `score = data @ weights` for weighted sums per row. Every PyTorch / TensorFlow forward pass.

**Related terms.**
- **Inner product / dot product** — 1D × 1D version of matmul, returns a scalar.
- **Hadamard product** — formal name for element-wise multiplication.
- **Outer product** — `a[:, None] * b[None, :]` → `(m, n)` from two 1D arrays.
- **Broadcasting** — applies to `*` (element-wise), not to `@`.
- **BLAS** — the C/Fortran library doing the heavy lifting behind `@`.

**Gotcha.** Matmul requires **inner dims to match**: `(m, k) @ (k, n)`. The most common error is forgetting to transpose: `X @ w` works if `w` is shape `(k,)` or `(k, n)`, but not if `w` is `(n, k)`. Also: `@` doesn't broadcast — element-wise `*` does.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — full definitions are in [the guided walkthrough above](#2g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **`np.arange`** | Array version of Python `range`; `start:stop:step`, **end exclusive**, accepts floats. | Generating index arrays, plot x-axes, reshape sources. Prefer `np.linspace` for floats. |
| **Boolean mask** | A True/False stencil same length as the data; `arr[mask]` keeps elements where True. Returns a **copy**. | Filtering rows by a condition without a loop. |
| **Fancy indexing** | Pick by a list of positions: `arr[[2, 5, 7]]`. Returns a **copy** (not a view). | Reordering, sampling, gathering arbitrary positions, sort-with-aligned-rows. |
| **Aligned filter** | Mask from array A used to filter array B — works if both are the same length. `costs[votes >= 500]`. | Filtering one column by another's condition (common Pandas pattern). |
| **`reshape(r, c)`** | Restructure same data into a new shape; element count must stay the same. `-1` means "infer this dim." | 1D ↔ 2D transitions, image flattening for sklearn, adding/removing axes. |
| **`reshape` vs transpose** | Reshape *re-walks* the same data into a new shape; transpose *swaps axes* (changes which dim is rows vs cols). | Reshape: change layout. Transpose: change orientation. They're not interchangeable. |
| **Axis mantra** | The axis you pass is the axis that **disappears**. `axis=0` collapses rows → per-column result. `axis=1` → per-row. | Every aggregation. Internalise this and you'll stop guessing. |
| **`np.any` / `np.all`** | Reduce a boolean array to one bool (or one per axis if `axis=` is passed). | "Any NaN in this column?" `arr.isna().any()`. "All positive?" `(arr > 0).all()`. |
| **`np.where(cond, a, b)`** | Vectorised if/else — pick `a` where True, `b` where False, in one C call. | Creating labels, imputing values, conditional transforms. |
| **`np.where(cond)`** | Single-arg form: returns **indices** where condition is True (as a tuple). | Finding positions of matches. Usually you want `np.where(cond)[0]`. |
| **`np.sort`** | Returns a new sorted array. `arr.sort()` sorts in-place and returns `None`. | When you want the sorted values themselves. |
| **`np.argsort`** | Returns the **indices** that would sort the array. | When you need to keep aligned rows together: `data[data[:, 0].argsort()]`. |
| **Element-wise `*`** | Multiplies cell-by-cell; same shape or broadcastable. | Scaling, masking-by-multiply (`arr * (mask)`), Hadamard products. |
| **`np.dot`** | Matrix multiplication for 2D inputs; also accepts scalars and 1D vectors. | Legacy code; for new code prefer `@`. |
| **`np.matmul` / `@`** | Strict matrix multiplication. Inner dims must match: `(m, k) @ (k, n) → (m, n)`. | Linear regression `X @ w`, NN layers `X @ W + b`, weighted scores `data @ weights`. |
| **Matmul shape rule** | `(m, k) @ (k, n) → (m, n)`. The two inner `k`s must match. | Sanity-check every matmul by saying shapes out loud. |

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

<a id="3g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 3 concept. The cheat sheet below is the recap.

#### Vectorization

> **🪜 Mental model:** *No Python loop.* Replace a `for` loop with a single array operation; let compiled C do the heavy lifting.

**What it is.** Vectorization means **applying an operation to an entire array in one go**, instead of looping over elements in Python. `arr + 1` is vectorized; `[x + 1 for x in arr]` is not. The vectorized version dispatches to a precompiled C function that streams through the array using SIMD (Single Instruction Multiple Data) CPU instructions.

**Why it matters.** Vectorized code is 10–100× faster than the loop version, and shorter to read. "Vectorize the inner loop" is the #1 NumPy optimisation tip. It also unlocks broadcasting (which only works on vectorized ops). Interviewers love asking "how would you rewrite this Python loop using NumPy?" — they're testing whether you reach for vectorization automatically.

**How it works.** When you write `arr + 1`, NumPy calls a **ufunc** (universal function) — a precompiled C kernel that knows how to apply `+1` element-by-element over any-dtype array. The CPU's SIMD lanes can process 4–8 elements per clock. NumPy also releases the Python GIL during ufunc execution, allowing thread-level parallelism.

**Where it's used.** Everywhere. Any time you see a Python `for x in arr` doing math, it can almost always be replaced with vectorized ops. ML pipelines, image processing, signal processing — all are chains of vectorized operations.

**Related terms.**
- **ufunc (universal function)** — the compiled element-wise kernel behind every `+`, `*`, `np.sin`, etc.
- **SIMD** — CPU feature: do the same arithmetic op on multiple values in one instruction.
- **`np.vectorize`** — a *convenience wrapper*, **not** real vectorization (see below).
- **Broadcasting** — the rule that lets vectorized ops mix differently-shaped arrays.

**Gotcha.** `np.vectorize(fn)` is not actually vectorized — it's a thin Python loop with array I/O. Real speed comes from native NumPy ops (`np.where`, `+`, `*`) or compiled code (Numba, Cython).

#### `np.vectorize` — convenience wrapper

> **🪜 Mental model:** *Sugar, not speed.* Lets you call a Python function on an array, but underneath it loops in Python.

**What it is.** `np.vectorize(fn)` takes a Python function `fn` that works on scalars and returns a callable that accepts arrays. It's syntactic convenience — you don't have to write the loop yourself. But it does **not** make your function fast: it still calls `fn` once per element from Python.

**Why it matters.** It's a beginner trap. Reading the name, you'd expect speed; you get none. Recognising this in interview/review is a senior-judgment signal.

**How it works.** Internally it's basically `np.array([fn(x) for x in arr])`. NumPy adds output-dtype inference and a bit of broadcasting bookkeeping, but the bottleneck is still the per-element Python call.

**Where it's used.** Quick prototyping when you have a complicated Python function and don't yet care about speed. Replace with native NumPy / Pandas ops before shipping.

**Related terms.**
- **`np.frompyfunc`** — even thinner wrapper; also slow.
- **Numba `@njit`** — real JIT compilation, gives you genuine vectorization.
- **`np.where`, masking, ufuncs** — the actual fast tools.

**Gotcha.** Don't use it inside a hot loop. If you find yourself reaching for `np.vectorize` for performance, you're reaching for the wrong tool.

#### Broadcasting

> **🪜 Mental model:** *Right-align shapes.* Stretch the smaller array against the larger one for free (no copy) — but only if shapes are compatible position-by-position.

**What it is.** Broadcasting is NumPy's rule for letting differently-shaped arrays interact in a single op. When you do `A + B`, NumPy tries to make their shapes compatible by **stretching size-1 axes** of the smaller one to match the larger one — virtually, with no memory copy. If shapes can't be made compatible, you get a `ValueError`.

**Why it matters.** Broadcasting is what lets you write `data - data.mean(axis=0)` (a `(n, k)` array minus a `(k,)` row) without manually reshaping or looping. It's the most common source of subtle shape bugs ("the code ran but the result is wrong because broadcasting did something I didn't expect").

**How it works — the 4 rules.**
1. If arrays have different numbers of dimensions, **prepend 1s** to the smaller shape until both have the same `ndim`.
2. Compare shapes **dimension-by-dimension, right-to-left**. Two dimensions are compatible if they're **equal**, or one of them is **1**.
3. The output shape is the **max** along each dimension.
4. If neither rule 2 holds nor padding fixes it, **broadcasting fails**.

The "stretch" itself is virtual — implemented with a stride of 0 on the broadcast axis — so no memory is allocated for the stretched view.

**Where it's used.** Centering data: `data - data.mean(axis=0)`. Per-row scaling: `data / row_norms[:, None]`. Outer products: `a[:, None] * b[None, :]`. Adding a bias to a batch: `X @ W + b` (where `b` is `(n_out,)` and broadcasts across the batch).

**Related terms.**
- **`keepdims=True`** — preserve the size-1 axis after a reduction so broadcasting works on the result.
- **`np.newaxis` / `None`** — `arr[:, None]` adds a size-1 axis at that position.
- **`np.broadcast_to(arr, shape)`** — explicit virtual stretch (read-only).
- **Stride trick** — the implementation: stride=0 means "don't advance the pointer."

**Gotcha.** `(5, 2) - (5,)` **fails** because right-aligning gives `(5, 2)` vs `(5,)` → compare `2` with `5` → mismatch. The fix is to make it `(5, 1)`: `data - data.mean(axis=1, keepdims=True)` or `[:, None]`.

#### `np.tile` — explicit replication

> **🪜 Mental model:** *Photocopy and arrange.* Repeat an array N times along given axes — really copy memory, unlike broadcasting.

**What it is.** `np.tile(arr, reps)` returns a new array where `arr` has been **repeated** `reps` times along each axis. `np.tile([1, 2, 3], 2)` → `[1, 2, 3, 1, 2, 3]`. `np.tile([[1, 2]], (3, 1))` stacks the row three times.

**Why it matters.** Most of the time you should reach for **broadcasting** instead (no memory cost). `np.tile` is the fallback when you genuinely need a *materialised* repeated array (e.g., for passing to a function that doesn't broadcast properly, or to debug a shape mismatch by making the shapes explicitly equal).

**How it works.** Allocates a new buffer of size `arr.size * prod(reps)` and copies the original into each tile slot.

**Where it's used.** Building test patterns. Building grid-like structures when broadcasting won't reach. As a thinking tool — if you can't broadcast, ask "what shape would I need if I tiled this explicitly?" — and once you know, broadcasting will usually work.

**Related terms.**
- **`np.repeat`** — repeats each element individually (`np.repeat([1,2,3], 2)` → `[1,1,2,2,3,3]`), versus tile which repeats the whole array.
- **Broadcasting** — the free, preferred alternative when the operation allows.
- **`np.broadcast_to`** — explicit broadcast view; like tile but with no memory cost.

**Gotcha.** Don't tile when broadcasting works — you're paying full memory cost for the same result.

#### `split` family — `split` / `array_split` / `hsplit` / `vsplit`

> **🪜 Mental model:** *Cut a cake.* `split` insists on equal slices; `array_split` tolerates a smaller last slice.

**What it is.** `np.split(arr, N, axis=k)` divides `arr` into `N` equal pieces along axis `k` and returns a list of sub-arrays. If `arr` doesn't divide evenly into `N`, it **raises** a `ValueError`. `np.array_split` does the same but **tolerates** uneven splits — the last few pieces absorb the remainder. `np.hsplit(arr, N)` is shorthand for `split(arr, N, axis=1)`; `np.vsplit` is `axis=0`.

**Why it matters.** Splitting is how you create batches, train/test partitions, or break a long array into manageable chunks. The `split` vs `array_split` distinction matters because real-world arrays rarely divide evenly.

**How it works.** Both produce **views** of the original (no copy) — each sub-array points into the same buffer. Mutating a sub-array mutates the original.

**Where it's used.** Mini-batch generation for training: `batches = np.array_split(data, n_batches)`. Splitting an image into tiles. Breaking a long time series into windows.

**Related terms.**
- **`np.split(arr, [i, j, k])`** — pass a list of indices instead of a count for *non-uniform* splits.
- **`np.hsplit` / `vsplit`** — axis-specific shortcuts.
- **`np.array_split`** — uneven-tolerant version.

**Gotcha.** `np.split` raises on uneven splits. For mini-batches of arbitrary size, always use `np.array_split`.

#### Stack family — `vstack` / `hstack` / `concatenate` / `stack`

> **🪜 Mental model:** *`concatenate` joins along an existing axis; `stack` creates a new axis.* If you want to add a row, concatenate. If you want to combine 5 images into a tensor, stack.

**What it is.** Four ways to combine arrays:
- **`np.vstack([a, b])`** — vertical stack; same number of **columns** required. (Special case of `concatenate(axis=0)`.)
- **`np.hstack([a, b])`** — horizontal stack; same number of **rows** required. (Special case of `concatenate(axis=1)` for 2D.)
- **`np.concatenate([a, b], axis=k)`** — general join along an **existing** axis `k`. All other axes must match in size.
- **`np.stack([a, b], axis=k)`** — creates a **NEW** axis at position `k`. All inputs must have *identical* shape.

**Why it matters.** Concatenate-vs-stack is the most-confused pair in NumPy. The output shape rules differ: `concatenate` keeps `ndim` the same; `stack` adds one. You'll need one or the other every time you assemble training data from multiple sources.

**How it works.** All four allocate a new buffer big enough to hold the combined data and copy each input into its slot. They are not views.

**Where it's used.**
- Adding new rows to a table: `np.vstack([data, new_rows])`.
- Joining columns: `np.hstack([X, np.ones((n, 1))])` (adding a bias column).
- Building a batch tensor from N same-shape arrays: `np.stack([img1, img2, …], axis=0)` → `(N, H, W, 3)`.
- 1D → 2D columns: `np.column_stack([votes, ratings])` → table with two columns.

**Related terms.**
- **`np.column_stack`** — like `hstack` but auto-promotes 1D arrays into columns of a 2D matrix.
- **`np.row_stack`** — alias for `vstack`.
- **`np.dstack`** — depth-stack; stacks along axis 2.
- **`pd.concat`** — the Pandas equivalent (works on DataFrames).

**Gotcha.** `np.stack` and `np.concatenate` have *different* shape requirements. `stack` insists on **identical** input shapes (it's adding an axis). `concatenate` allows different sizes along the joined axis only.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — full definitions are in [the guided walkthrough above](#3g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Vectorization** | Applying an op to a whole array in one compiled C call — no Python loop. | Any time you'd write `for x in arr` doing math. 10–100× speedup. |
| **`np.vectorize`** | Convenience wrapper around a Python function; **not** real speed-up — still loops in Python. | Prototyping only. Replace with native ops or Numba before shipping. |
| **Broadcasting** | Auto-stretching of smaller arrays to match larger ones for elementwise ops, with **no memory copy**. | Centering, scaling, outer products — any time shapes don't quite match but should. |
| **4 broadcasting rules** | (1) Prepend 1s to smaller shape. (2) Each dim must be equal or 1. (3) Output dim = max. (4) Else fail. Right-align shapes. | Whenever a `(5,2) op (?,?)` happens — work the rules right-to-left. |
| **`np.tile`** | Explicit physical replication of an array along given axes. Allocates memory. | When broadcasting can't reach the layout you need — or for debugging shape mismatches. |
| **`np.split(arr, N, axis)`** | Even split into `N` pieces; raises `ValueError` if `arr` doesn't divide evenly. | When you *know* the array divides evenly and want to assert it. |
| **`np.array_split`** | Same as `split` but tolerates uneven counts — last piece absorbs the remainder. | Mini-batch generation, splitting arrays of arbitrary length. |
| **`np.hsplit` / `vsplit`** | Shortcut for `split(axis=1)` / `split(axis=0)`. Both require even division. | Quick horizontal/vertical cuts when you know the shape divides. |
| **`np.vstack`** | Stack arrays as rows; requires same **number of columns** in each input. | Appending new rows to a 2D table. |
| **`np.hstack`** | Stack arrays as columns; requires same **number of rows** in each input. | Appending new columns; adding a bias column. |
| **`np.concatenate(axis=k)`** | General join along an **existing** axis `k`. Keeps `ndim` the same. | Any "merge along this dim" — generalises `vstack`/`hstack`. |
| **`np.stack(axis=k)`** | Joins along a **NEW** axis. All inputs must have *identical* shape. `ndim` increases by 1. | Building a batch tensor from N same-shape items (e.g., images → `(N, H, W, 3)`). |
| **`np.column_stack`** | Like `hstack` but auto-promotes 1D arrays into columns. | Turning a list of 1D feature vectors into a 2D feature matrix. |

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

All key terms across the four notebooks — alphabetical for quick lookup. Each entry is a **2–4 sentence beginner-friendly definition**. For full walkthroughs, follow the links into the module sections.

| Term | Definition |
|---|---|
| **Aligned filter** | A filtering pattern where the boolean mask comes from *one* array but is applied to *another* (e.g., `costs[votes >= 500]`). It works only when both arrays have the same length — NumPy walks them in lockstep. Common in Pandas: select rows of column A based on a condition on column B. ([walkthrough](#2g-guided)) |
| **`argsort`** | `np.argsort(arr)` returns the **indices** that would sort the array, not the sorted values themselves. The "argument-sort" view is useful when you need to sort one array while keeping a sibling array aligned — e.g., `data[data[:, 0].argsort()]` sorts whole 2D rows by column 0. ([walkthrough](#2g-guided)) |
| **Axis** | A single dimension of an N-dimensional array (`shape=(rows, cols)` → axis 0 is rows, axis 1 is cols). When an op accepts `axis=k`, the axis you pass is the one that **collapses**: `arr.sum(axis=0)` removes the row axis, leaving per-column sums. Internalising this mantra eliminates 90% of aggregation confusion. ([walkthrough](#2g-guided)) |
| **Boolean mask** | A True/False array of the same length as your data, used to filter or assign: `arr[mask]` keeps elements where True. Masks are usually built from comparisons (`arr > 5`) and can be combined with bitwise operators (`(a > 0) & (a < 10)`). Returns a **copy**, not a view. ([walkthrough](#2g-guided)) |
| **Broadcasting** | NumPy's rule that lets differently-shaped arrays be combined in a single op by **virtually stretching** size-1 axes to match — no memory copy. The 4 rules: prepend 1s to the smaller shape, each dim must be equal or 1, output is max per dim, else fail. It's the magic behind `data - data.mean(axis=0)` and most ML matrix math. ([walkthrough](#3g-guided)) |
| **Coercion (type)** | Automatic or explicit conversion of a value from one type to another. NumPy *automatically* coerces when you create an array with mixed types (priority: string > float > int > bool). Explicit coercion uses `astype` or `pd.to_numeric(..., errors='coerce')` (the latter turns bad strings into NaN). Coercion is where data silently goes wrong — always inspect dtypes after loading. |
| **Concatenate** | Joining arrays along an **existing** axis. `np.concatenate([a, b], axis=0)` glues `b` below `a` (more rows); `axis=1` glues `b` to the right (more cols). All other axes must match in size, and the output has the same `ndim` as the inputs. Contrast with **stack**, which adds a new axis. ([walkthrough](#3g-guided)) |
| **Contiguous memory** | When an array's elements are stored in one unbroken strip of RAM, in row-major (C) order. This is why NumPy is fast: the CPU can stream through the buffer with maximal cache use and SIMD. Most operations preserve contiguity; transpose breaks it (the data is now stored "column-first"), which may force a copy on a subsequent reshape. |
| **Copy** | An array that lives in its own memory buffer — mutations don't affect the source. Fancy indexing (`arr[[1, 2, 3]]`), boolean masking, `astype`, and `.copy()` all return copies. Use copies when you're going to mutate and don't want side effects on the original. ([walkthrough](#1g-guided)) |
| **Determinant** | A scalar derived from a square matrix that summarises a key property: if the determinant is 0, the matrix is **singular** (non-invertible, columns linearly dependent). Used in linear algebra to check whether a system of equations has a unique solution. NumPy: `np.linalg.det(A)`. |
| **`dtype`** | The element type of an ndarray — `int64`, `float64`, `bool`, `<U7` (Unicode string ≤7 chars), `object` (Python objects). Fixed when the array is created; mixing types at creation triggers promotion to the most-general type. Wrong dtype is the silent cause of most NumPy/Pandas surprises. ([walkthrough](#1g-guided)) |
| **EDA (Exploratory Data Analysis)** | The first hands-on phase of any data project: inspect shape and dtypes, summarise per column, plot distributions, find missing values and outliers, form hypotheses. Coined by John Tukey in the 1970s; skipping EDA is the #1 reason ML models fail in production. Every notebook in this repo opens with EDA. ([walkthrough](#1g-guided)) |
| **Fancy indexing** | Selecting elements by an explicit array of positions: `arr[[2, 5, 7]]` returns three elements in that order. Sibling of basic slicing and boolean masking; returns a **copy** (not a view) because the picks can be arbitrary. Used for reordering, gathering, sorting-with-alignment. ([walkthrough](#2g-guided)) |
| **Matrix multiplication** | The dot-product-based way to combine two matrices: `(m, k) @ (k, n) → (m, n)`, where each output cell is a row-of-A · column-of-B dot product. Use `@` (or `np.matmul`); reserve `np.dot` for legacy/scalar code. Confusing it with element-wise `*` is the #1 silent bug in numerical code. ([walkthrough](#2g-guided)) |
| **NaN (Not a Number)** | A special floating-point value meaning "missing" or "undefined." Only legal in `float` dtype (an `int` array can't hold NaN — try and pandas will promote to float). Key gotcha: `NaN == NaN` is `False`, so use `np.isnan(arr)` to detect, and `np.nanmean` / `np.nansum` to aggregate while ignoring NaNs. |
| **`ndarray`** | NumPy's core N-dimensional array — a flat memory buffer plus a `shape` tuple, `strides`, and one `dtype`. Underlies every Pandas column, every scikit-learn input, and (in adapted form) every PyTorch tensor. The single most important data structure in scientific Python. ([walkthrough](#1g-guided)) |
| **Ravel** | `arr.ravel()` flattens an N-D array to 1D, returning a **view** when memory layout allows (and a copy otherwise). Cheaper than `.flatten()`, which always copies. Use ravel when you need a flat view for indexing or iteration. |
| **Reshape** | Restructure same data into a new shape; element count must match exactly. `-1` in any one dim means "you figure it out from the size." Usually a view; falls back to a copy when the new strides aren't expressible (often after a transpose). ([walkthrough](#2g-guided)) |
| **SIMD (Single Instruction Multiple Data)** | A CPU feature: one instruction processes multiple values (typically 4–8 floats) in parallel. NumPy's ufuncs use SIMD, which is part of why vectorized code is much faster than Python loops. You don't program SIMD directly; you get it for free by using NumPy ops. |
| **Slicing** | Extracting a contiguous range or stride from an array: `arr[start:end:step]`, with **end exclusive**. Returns a **view** of the same memory — mutations propagate to the source. Negative indices count from the end. ([walkthrough](#1g-guided)) |
| **Sort** | Ordering an array. `np.sort(arr)` returns a sorted copy; `arr.sort()` sorts in-place and returns `None`. For aligned-row sorting (e.g., sort a 2D array by one column), use `argsort` to get the permutation. Default algorithm is quicksort (`kind='stable'` for a stable variant). ([walkthrough](#2g-guided)) |
| **Stack** | Joining arrays along a **NEW** axis — `np.stack([a, b], axis=0)` adds a dim and requires all inputs to have *identical* shape. Output `ndim` is one larger than inputs. Use for assembling batches: 5 same-shape images → `(5, H, W, 3)` tensor. Contrast with **concatenate**, which joins along an existing axis. ([walkthrough](#3g-guided)) |
| **Stride** | The number of bytes to step in memory to move one position along a given axis. NumPy stores `shape`, `dtype`, *and* `strides` for every array. Transpose just swaps strides → O(1), no data movement. Broadcasting uses a stride of 0 to virtually "stretch" a size-1 axis. |
| **Type priority** | The order NumPy uses when promoting mixed types to one dtype: **string > float > int > bool**. One stray string in a numeric list silently turns the whole array into a string array. Defend with `dtype=` at creation, or clean the input first. ([walkthrough](#1g-guided)) |
| **Vectorization** | Applying an operation to a whole array in one compiled C call, with no Python loop. NumPy ufuncs (`+`, `*`, `np.where`, `np.sin`, …) are vectorized; `np.vectorize` is *not* (despite the name). The single biggest performance lever in NumPy. ([walkthrough](#3g-guided)) |
| **View** | An array object that shares its memory with another. Created by basic slicing (`arr[1:5]`), transpose (`.T`), reshape (when strides allow), and `np.broadcast_to`. Mutating a view mutates the source. Check with `b.base is a` or `np.shares_memory(a, b)`. ([walkthrough](#1g-guided)) |
| **`np.where`** | The vectorized if/else: `np.where(cond, a, b)` returns an array picking from `a` where True and `b` where False, element-by-element. With a single argument, it returns the **indices** where the condition is True (as a tuple) — a different return type entirely. ([walkthrough](#2g-guided)) |

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
