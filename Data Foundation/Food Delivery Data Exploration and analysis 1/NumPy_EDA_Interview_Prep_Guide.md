<a id="top"></a>
# NumPy & EDA Interview Prep — Food Delivery Notebook Companion

> Built from your **"Food Delivery Data Exploration and Analysis 1"** Colab notebook. Every topic in your notebook is here, plus the related interview questions the notebook doesn't get to (broadcasting, views vs copies, fancy indexing, etc.) that *always* come up.

**How to use this:** For each topic there are four blocks:
- **🧠 Mental model** — the intuition behind why it exists
- **📐 The mechanics** — the precise definition and syntax
- **❓ Basic questions** — what gets asked first
- **🎯 Advanced & trick questions** — what separates a strong candidate

The Zomato/food-delivery context from your notebook is woven into the examples wherever it fits — that way the interview answers stick to a concrete business story.

---

## Table of contents

**Core (from your notebook)**
1. [Machine Learning — the motivation](#1-ml-motivation)
2. [EDA & DAV — why we explore before we model](#2-eda-dav)
3. [Python lists vs NumPy arrays](#3-list-vs-array)
4. [Why NumPy is faster (the real reasons)](#4-why-numpy-fast)
5. [Creating arrays — every way you should know](#5-creating-arrays)
6. [Dimensions, shape, ndim, size, dtype](#6-shape-dtype)
7. [Type coercion & priority rules](#7-coercion)
8. [`astype()` — type conversion](#8-astype)
9. [Indexing — positive, negative, fancy](#9-indexing)
10. [Slicing — 1D and 2D](#10-slicing)

**Advanced (interview must-haves)**

11. [⚠️ Views vs copies — the #1 NumPy gotcha](#11-views-copies)
12. [Boolean indexing](#12-boolean)
13. [Broadcasting](#13-broadcasting)
14. [Aggregations and axis](#14-aggregations)
15. [Reshape, flatten, ravel, transpose](#15-reshape)
16. [Stacking, splitting, concatenation](#16-stacking)
17. [Sorting, unique, set operations, searching](#17-sorting)
18. [NaN & missing data handling](#18-nan)
19. [`np.where`, `clip`, `select`, percentiles, conditional ops](#19-where-clip)
20. [Linear algebra essentials](#20-linalg)
21. [Strides, memory layout & performance internals](#21-strides)
22. [I/O — saving and loading](#22-io)
23. [Datetime & timedelta arrays](#23-datetime)

**Practice & revision**

24. [Common trick / multi-correct questions](#24-tricks)
25. [Rapid-fire flashcards](#25-flashcards)
26. [📚 Today's learning — quick summary](#26-summary)
27. [🔁 Revision drill — 10-minute speed-run](#27-revision)
28. [✅ Best practices cheat sheet](#28-bestpractices)
29. [Final checklist before any data-science interview](#29-checklist)
30. [Mapping back to your notebook](#30-mapping)

---

<a id="1-ml-motivation"></a>
## 1. Machine Learning — the motivation

### 🧠 Mental model

Machine learning is the ability of a machine to learn from **experience, history, and patterns** — just like a human. The notebook frames this with two classic stories:

- **The Vijay Mallya loan example.** Two candidates apply for a ₹20 lakh loan. Looking only at *current capacity* (income), you'd pick C1. Then you learn C1 has a history of defaulting. Decision flips. **Lesson:** good decisions depend on *patterns from history*, not just present-day numbers.
- **The salary prediction example.** 1 yr → ₹10k, 2 yr → ₹20k, 3 yr → ₹30k. What about 6 years? Easy — ₹60k. You spotted a pattern: `salary = years × 10k`. Humans can spot patterns in 3 rows. ML systems do it across 10 million.

This is the entire reason for everything that follows: **before a model can learn patterns, you have to understand the data.** That's what EDA is.

### ❓ Basic questions

1. **What is machine learning in one sentence?**
   A machine learning the relationship between inputs and outputs from historical data, then applying that relationship to new inputs.

2. **What's the difference between rule-based programming and ML?**
   Rule-based: a human writes the rules. ML: the machine discovers the rules from examples. ML wins when the rules are too complex or numerous for a human to write down.

3. **What are the three main types of ML?**
   - **Supervised** — labels available (price, fraud/not, rating). Most of business ML.
   - **Unsupervised** — no labels; discover structure (clustering, anomaly detection).
   - **Reinforcement** — learn by trial-and-error rewards (robotics, game playing, recommendation).

### 🎯 Advanced questions

- **Why can't a bank just use rules like "if income > X then approve"?**
  Real fraud signals are interactions of many features — income, transaction patterns, geolocation, device, history. A handful of rules can't capture that combinatorial space; an ML model can.
- **Difference between ML and AI and Deep Learning?**
  AI ⊃ ML ⊃ Deep Learning. AI is the broad goal (machines acting intelligently). ML is one approach (learning from data). DL is one subfield of ML (neural networks with many layers).
- **What is the "bias-variance" tradeoff in one sentence?**
  A model that's too simple **underfits** (high bias); a model that's too complex **overfits** (high variance). Good ML balances both.

[🔝 Back to top](#top)

---

<a id="2-eda-dav"></a>
## 2. EDA & DAV — why we explore before we model

### 🧠 Mental model

**EDA** = Exploratory Data Analysis. **DAV** = Data Analysis and Visualization (the broader skillset built on NumPy, Pandas, Matplotlib, Seaborn).

Your notebook makes the case with a clear analogy: **before watching a movie, you check ratings, reviews, actors, director, genre, price.** You don't decide based on one factor. Same with data — before modelling, you look at every feature, every distribution, every relationship.

Two reasons EDA is necessary (also straight from the notebook):

1. **Remove irrelevant patterns.** A naive machine might think the "Serial Number" column predicts salary because it correlates with row order. A human catches that absurdity in seconds.
2. **Clean corrupt data.** Real-world data is messy. Costs stored as `"'800.0'"` (a quoted string). Ratings stored as `"4.1/5"`. Costs with commas like `"1,200"`. Math doesn't work on strings — you have to clean first.

### ❓ Basic questions

4. **What is EDA?**
   The process of exploring a dataset's features, distributions, and relationships before modelling — to understand what you have, what's wrong with it, and what's worth modelling.

5. **What are the four libraries in the standard Python data stack?**
   - **NumPy** — numerical arrays and math
   - **Pandas** — tabular data (DataFrames), text, grouping
   - **Matplotlib** — base plotting
   - **Seaborn** — statistical plots built on top of Matplotlib

6. **What kinds of business questions does EDA on a food delivery dataset answer?**
   - Which restaurants/areas have the highest ratings or votes?
   - How does cost correlate with rating?
   - Which cuisines drive the most orders?
   - Where should we expand or run promotions?

### 🎯 Advanced questions

- **Give three concrete EDA cleaning tasks for the Zomato dataset.**
  - Convert `approx_cost(for two people)` from `"1,200"` (string with comma) to a clean number.
  - Drop or impute rows where `rating` is missing or stored as `"NEW"`.
  - Standardize cuisine strings (`"North Indian"` vs `"north indian"`).
- **EDA vs feature engineering — same thing?**
  No. EDA is *understanding*. Feature engineering is *creating new columns* to help a model (e.g., `cost_per_vote = cost / votes`). EDA usually surfaces ideas for feature engineering.
- **Walk me through a 60-second EDA checklist.**
  1. `.shape` and `.dtypes` 2. `.isna().sum()` 3. `.describe()` for numerics 4. `.value_counts()` for categoricals 5. histograms / boxplots for outliers 6. correlation heatmap 7. target-vs-feature scatters.

[🔝 Back to top](#top)

---

<a id="3-list-vs-array"></a>
## 3. Python lists vs NumPy arrays

### 🧠 Mental model

Python lists are flexible containers — they can hold mixed types and were never designed for math. NumPy arrays are numeric containers — they hold one type and were built specifically for vectorized math.

The notebook nails it with one example:

```python
[4, 5, 6] * 2          # → [4, 5, 6, 4, 5, 6]   (replicate)
np.array([4,5,6]) * 2  # → array([8, 10, 12])    (multiply each)
```

Same operator, completely different behavior. That single example explains why every data scientist reaches for NumPy.

### 📐 The mechanics

```python
import numpy as np

votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
costs = np.array(["'800.0'", "'800.0'", "'800.0'", "'300.0'", "'600.0'",
                  "'600.0'", "'600.0'", "'700.0'", "'550.0'", "'500.0'"])
```

`type(votes)` → `numpy.ndarray`. The `ndarray` (n-dimensional array) is **the** NumPy data structure — everything else is built around it.

### ❓ Basic questions

7. **What does NumPy stand for?**
   Numerical Python.

8. **What is `np.array()`?**
   The constructor that converts a list (or other iterable) into a NumPy `ndarray`.

9. **What does `[4, 5, 6] * 2` do vs `np.array([4, 5, 6]) * 2`?**
   List: replication (`[4,5,6,4,5,6]`). Array: element-wise multiplication (`[8,10,12]`).

10. **Can a NumPy array hold mixed types like `[1, "Akash", 3.5]`?**
    Technically yes, but it will coerce everything to the highest-priority type (here, strings). See section 7.

### 🎯 Advanced questions

- **When would you choose a list over a NumPy array?**
  - When you need heterogeneous types and don't need math
  - When you're frequently appending (lists are O(1) for append; arrays aren't)
  - For very small data where the NumPy overhead isn't worth it
- **Pandas Series uses NumPy underneath — why does it exist then?**
  Pandas adds *labeled* indices, missing-value handling (`NaN`), heterogeneous columns, and methods like `groupby`. NumPy is the engine; Pandas is the tabular interface.
- **Is `np.array` the only way to create an array? When would you avoid it?**
  No — `np.zeros`, `np.empty`, `np.arange`, `np.fromiter`, etc. Avoid `np.array(big_list)` if you can pre-allocate with `np.empty(shape)` and fill in-place; it skips the intermediate list copy.

[🔝 Back to top](#top)

---

<a id="4-why-numpy-fast"></a>
## 4. Why NumPy is faster (the real reasons)

### 🧠 Mental model — the "Tea Room" analogy from your notebook

- **Room 1 (Python list):** ingredients scattered all over the room. Every operation has to chase down where the next item lives.
- **Room 2 (NumPy array):** ingredients lined up in order on one shelf. Pick them up in one sweep.

That shelf-vs-scatter image is the entire performance story. Three concrete reasons:

1. **Contiguous memory.** A NumPy array stores its values in one continuous block of RAM. A Python list stores *pointers* to scattered objects elsewhere in memory. CPUs love contiguous blocks (cache-friendly); they hate pointer chasing.

2. **Homogeneous data type.** Every element is the same type and the same byte width, so the CPU knows the stride and can stream through with no per-element type checks.

3. **Vectorization in C.** When you write `votes * 2`, NumPy doesn't run a Python `for` loop — it dispatches to a tight C loop with SIMD instructions. That's 10×–100× faster than the Python interpreter walking a list.

### ❓ Basic questions

11. **Give two reasons NumPy is faster than Python lists.**
    Contiguous memory + homogeneous types enable vectorized C-level operations with cache-friendly access patterns.

12. **What is vectorization?**
    Applying an operation to an entire array at once (using one C call) instead of looping in Python. `arr * 2` is vectorized; `[x*2 for x in arr]` is not.

13. **Does NumPy store the addresses of elements or the actual values in its contiguous block?**
    The **actual values** (this is exactly the doubt your notebook flags). Python lists store addresses; that's why they're slower.

### 🎯 Advanced questions

- **What is a "ufunc"?**
  A *universal function* — a NumPy function that operates element-wise on arrays with broadcasting, type handling, and vectorization built in. `np.add`, `np.sqrt`, `np.exp`, `np.maximum` are all ufuncs.
- **Is there ever a case where a Python loop beats NumPy?**
  Yes — for very small arrays (a handful of elements) the NumPy dispatch overhead can exceed the savings. Also when the operation is inherently sequential (one step depends on the previous).
- **What is SIMD?**
  Single Instruction Multiple Data — CPU instructions that process 4, 8, or 16 numbers in one cycle. NumPy's C kernels use SIMD; Python interpreters don't.
- **What is the GIL and does NumPy bypass it?**
  Python's Global Interpreter Lock prevents true threading. NumPy *releases* the GIL during C-level operations, so heavy compute can use multiple threads even from Python.
- **`np.vectorize` — is it really vectorization?**
  No. It's a convenience wrapper that loops in Python under the hood. It's for code clarity, not speed. True vectorization means a single C call over the whole array.

[🔝 Back to top](#top)

---

<a id="5-creating-arrays"></a>
## 5. Creating arrays — every way you should know

### 📐 The mechanics

```python
np.array([1, 2, 3])           # from a list
np.zeros((3, 4))              # 3×4 array of zeros
np.ones((2, 5))               # 2×5 array of ones
np.full((2, 2), 7)            # 2×2 array of sevens
np.arange(0, 10, 2)           # [0, 2, 4, 6, 8] — like range()
np.linspace(0, 1, 5)          # [0., 0.25, 0.5, 0.75, 1.] — N evenly spaced
np.logspace(0, 2, 3)          # [1., 10., 100.] — logarithmic spacing
np.eye(3)                     # 3×3 identity matrix
np.identity(3)                # same thing
np.diag([1, 2, 3])            # diagonal matrix
np.random.rand(3, 4)          # uniform random [0, 1)
np.random.randn(3, 4)         # standard normal (mean 0, std 1)
np.random.randint(0, 10, 5)   # 5 random ints in [0, 10)
np.empty((3, 3))              # uninitialized — fastest
np.zeros_like(other_arr)      # zeros, same shape/dtype as another array
np.ones_like(other_arr)       # ones, same shape/dtype as another array
```

### ❓ Basic questions

14. **What's the difference between `np.arange` and `np.linspace`?**
    `arange(start, stop, step)` — you control the **step size**, length is whatever falls out. `linspace(start, stop, num)` — you control the **number of points**; step size is computed for you. `linspace` includes the endpoint by default; `arange` does not.

15. **Difference between `np.zeros` and `np.empty`?**
    `zeros` initializes to 0. `empty` allocates memory without initializing — values are whatever garbage was in that memory. Use `empty` when you're about to overwrite everything anyway (faster).

### 🎯 Advanced questions

- **`np.random.rand` vs `np.random.randn`?**
  `rand` → uniform [0, 1). `randn` → standard normal (mean 0, std 1). The `n` stands for normal.
- **How to make NumPy reproducible?**
  `np.random.seed(42)` before any random call. Or in modern NumPy: `rng = np.random.default_rng(42); rng.standard_normal(...)`.
- **What's `np.meshgrid` and when do you use it?**
  Generates 2D coordinate matrices from two 1D vectors. Used for evaluating functions on a 2D grid — heatmaps, contour plots, gradient computation.
- **Why use `zeros_like(x)` instead of `zeros(x.shape)`?**
  Preserves *dtype* and memory layout (C/F-order) of `x`. Important when downstream code expects matching dtype.

[🔝 Back to top](#top)

---

<a id="6-shape-dtype"></a>
## 6. Dimensions, shape, ndim, size, dtype

### 🧠 Mental model

Four attributes describe everything about an array's structure:

| Attribute | Tells you |
|---|---|
| `.shape` | Tuple of dimensions, e.g. `(10,)` or `(5, 2)` |
| `.ndim` | Number of dimensions (1, 2, 3, ...) |
| `.size` | Total number of elements (product of shape) |
| `.dtype` | The element data type (`int64`, `float64`, `<U7`, etc.) |
| `.itemsize` | Bytes per element (4 for int32, 8 for int64/float64) |
| `.nbytes` | Total bytes = `size × itemsize` |

**The bracket shortcut from your notebook:** count opening brackets at the start of an array.
- `[1, 2, 3]` → 1D
- `[[1, 2], [3, 4]]` → 2D
- `[[[...]]]` → 3D

### 📐 The mechanics

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])

votes.shape   # (10,)        — 1D, 10 elements
votes.ndim    # 1
votes.size    # 10
votes.dtype   # int64
votes.nbytes  # 80           — 10 elements × 8 bytes
```

For a 2D array of 5 restaurants × 2 columns (votes, cost):

```python
two_d_data.shape   # (5, 2)   — 5 rows × 2 cols
two_d_data.ndim    # 2
two_d_data.size    # 10       — 5 × 2
```

### ❓ Basic questions

16. **What does the shape `(10,)` mean? Why the trailing comma?**
    1D array with 10 elements. The trailing comma is Python's way of distinguishing a 1-element tuple `(10,)` from the integer in parentheses `(10)`.

17. **What's the difference between `shape` and `size`?**
    `shape` is the structure (rows, cols, etc.). `size` is the total element count. For a `(5, 2)` array, shape is `(5, 2)` and size is `10`.

18. **What does `dtype='<U32'` mean?**
    Little-endian Unicode string, up to 32 characters per element. The `U` means Unicode (string). NumPy uses this dtype when any element in the array is a string.

### 🎯 Advanced questions

- **What's the difference between `shape (3,)` and `shape (3, 1)` and `shape (1, 3)`?**
  - `(3,)` → 1D vector with 3 elements
  - `(3, 1)` → 2D column vector (3 rows × 1 col)
  - `(1, 3)` → 2D row vector (1 row × 3 cols)
  They contain the same values but behave differently in broadcasting and matrix multiplication.
- **What is `dtype=object`?**
  A NumPy array of Python objects (pointers). Loses all the speed advantages — it's essentially a Python list with NumPy syntax. Avoid unless you really need heterogeneity.
- **Common dtype byte sizes — should know cold.**
  `int8`=1, `int16`=2, `int32`=4, `int64`=8, `float32`=4, `float64`=8 (default), `bool`=1, `complex128`=16.
- **How do you check actual memory footprint?**
  `arr.nbytes` for the raw buffer. For *total* including Python overhead use `sys.getsizeof(arr)`.

[🔝 Back to top](#top)

---

<a id="7-coercion"></a>
## 7. Type coercion & priority rules

### 🧠 Mental model

NumPy arrays are **homogeneous** — every element must be the same type. When you create an array from a list with mixed types, NumPy promotes everything to the **highest priority** type:

> **String > Float > Integer > Boolean**

### 📐 The mechanics

```python
np.array([1, 2, 3.5])         # → [1.0, 2.0, 3.5]      all float
np.array([1, "Akash", 3.5])   # → ['1', 'Akash', '3.5'] all string
np.array([True, 6])           # → [1, 6]                bool → int (True=1, False=0)
np.array([True, 6, 3.14])     # → [1.0, 6.0, 3.14]      all float
```

### ❓ Basic questions

19. **What is type priority in NumPy and why does it exist?**
    String > Float > Int > Bool. It exists because NumPy arrays must be homogeneous — when the input is mixed, NumPy picks the "most general" type so no information is lost.

20. **What happens to `np.array([True, 6])`?**
    Both become integers: `[1, 6]`. Booleans are silently upgraded to 0/1.

21. **Why does `np.array([1, "hello"])` turn the integer into a string?**
    String is highest priority; integers can be safely represented as their string form (`"1"`), but not vice versa.

### 🎯 Advanced questions

- **Why is this a real-world problem? Give a Zomato example.**
  In the Zomato dataset, `approx_cost` is sometimes scraped as `"1,200"` (a string). If a single cost is messy, NumPy may coerce the *whole* cost column to strings — and then `mean()` fails. You have to clean and `astype(float)` first.
- **What's integer overflow in NumPy and how to avoid it?**
  `np.array([200, 100], dtype=np.int8)` can't hold values > 127 — it wraps around. Use a bigger dtype (`int32`, `int64`) when in doubt. Pandas defaults to 64-bit, but explicitly-set dtypes can bite you.
- **What is "upcasting" during arithmetic?**
  `int32 + float64 → float64`. NumPy promotes to the wider type to avoid info loss. You can force back down with `.astype()` if needed.

[🔝 Back to top](#top)

---

<a id="8-astype"></a>
## 8. `astype()` — type conversion

### 🧠 Mental model

`astype()` returns a **new array** with the requested dtype. It does not modify the original.

### 📐 The mechanics

```python
a = np.array(['1.2', '2.5', '3.6', '4.8'])
a.dtype          # dtype('<U3')   — strings, up to 3 chars

b = a.astype(float)
b                # array([1.2, 2.5, 3.6, 4.8])
b.dtype          # dtype('float64')

# Float → int truncates (does NOT round!)
np.array([1.7, 2.8, 3.9]).astype(int)   # → [1, 2, 3]   (not [2, 3, 4])
```

### ❓ Basic questions (matches the MCQ in your notebook)

22. **Which method converts a float array to integers?**
    `arr.astype(int)` ✓
    *Not* `arr.toInt()`, *not* `int(arr)`, *not* `astype(arr, int)`. Only the first is valid NumPy syntax.

23. **Does `astype(int)` round or truncate?**
    **Truncate** — `1.7 → 1`, `2.8 → 2`. To round properly, use `np.round(arr).astype(int)`.

24. **Why convert `"1,200"` to a number?**
    To do math. You can't take the mean of strings. The full pipeline is: strip commas/quotes → `astype(float)` → compute.

### 🎯 Advanced questions

- **Does `astype` modify the original array?**
  No. It returns a new array. Assign it back: `arr = arr.astype(float)`.
- **What happens if you `astype(int)` on a string array `"hello"`?**
  `ValueError`. The string must be parseable as a number.
- **How would you convert `"1,200"` to a float?**
  ```python
  arr = np.array(["1,200", "800", "550"])      # all strings
  cleaned = np.char.replace(arr, ",", "")      # strip commas
  cleaned.astype(float)                        # → [1200., 800., 550.]
  ```
  Or with Pandas: `df['cost'].str.replace(',', '').astype(float)`.
- **Difference between `astype` and `view`?**
  `astype` *reinterprets and copies* the data with the new dtype. `view` *re-reads the same bytes* through a different dtype lens (no copy, but values look very different). `astype` is what you want 99% of the time.

[🔝 Back to top](#top)

---

<a id="9-indexing"></a>
## 9. Indexing — positive, negative, fancy

### 🧠 Mental model

Indexing = grabbing a specific element. Three flavors:

1. **Positive indexing** — start at the front. `arr[0]` is the first.
2. **Negative indexing** — start at the back. `arr[-1]` is the last.
3. **Fancy indexing** — pass a *list* of indices to grab multiple elements at once.

For 2D arrays: **(row, column)** — always row first, column second.

### 📐 The mechanics

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])

votes[0]              # 775         first
votes[-1]             # 402         last
votes[-2]             # 504         second last
votes[[2, 3, 4]]      # [918, 88, 166]   fancy — pick multiple
votes[[2, 3, 4, 1, 2, 2]]  # [918, 88, 166, 787, 918, 918]  — repeats allowed!

# 2D — (row, col)
two_d_data[1, 0]      # row 1, col 0 — the second restaurant's vote count
two_d_data[0]         # row 0 — entire first restaurant row
```

### ❓ Basic questions

25. **What does `votes[-1]` return?**
    The last element. Negative indexing counts from the end.

26. **What's the rule for indexing a 2D array?**
    `arr[row, column]`. Row always comes first.

27. **Can fancy indexing repeat indices?**
    Yes — `votes[[2, 2, 2]]` returns the same element three times. Useful for resampling.

### 🎯 Advanced questions

- **Does `arr[2]` return a view or a copy?**
  A scalar (or for 2D, a view of that row). See section 11 — this is the views-vs-copies trap.
- **What's the difference between `arr[1, 2]` and `arr[1][2]`?**
  Both return the same scalar for basic 2D indexing, but `arr[1, 2]` is one operation while `arr[1][2]` is two (first row 1, then col 2 of that row). The single-bracket form is faster and is the NumPy idiom.
- **Fancy indexing returns a view or a copy?**
  Always a **copy**. (Plain slicing returns a view. This distinction matters in section 11.)
- **Can you assign through fancy indexing?**
  Yes: `arr[[0, 2, 4]] = 99` writes into those positions of the original. (The lookup returns a copy, but the assignment writes back.)

[🔝 Back to top](#top)

---

<a id="10-slicing"></a>
## 10. Slicing — 1D and 2D

### 🧠 Mental model

Slicing = grabbing a *range* of elements. The notebook calls this the **Race Analogy**: you start at `start` and stop *exactly one step before* `end`. **The end index is exclusive.**

### 📐 The mechanics

```python
arr[start : end : step]
```

```python
votes[:5]        # first 5      → [775, 787, 918, 88, 166]
votes[5:]        # from 5 to end → [286, 2556, 324, 504, 402]
votes[2:7]       # indices 2,3,4,5,6
votes[::2]       # every 2nd element
votes[::-1]      # reversed
```

2D slicing — **R, C** pattern:

```python
two_d_data[:3, :]       # first 3 rows, all columns
two_d_data[:, 0]        # all rows, column 0  → 1D shape (m,)
two_d_data[:, 0:1]      # all rows, column 0  → 2D shape (m, 1)
two_d_data[0:2, 1:3]    # rows 0–1, cols 1–2  → 2×2 block
```

### ❓ Basic questions

28. **Why does `votes[0:3]` return 3 elements, not 4?**
    The end index is exclusive — slice goes up to *but not including* index 3. So you get indices 0, 1, 2.

29. **What does `votes[:]` return?**
    A copy-feel slice of the entire array. *(Technically a view — see section 11.)*

30. **For a 2D array of shape `(m, n)`, what shape does `arr[:, 0]` return?**
    `(m,)` — a 1D array. Selecting a single column **collapses** that dimension. ✓ (This matches the MCQ in your notebook.)

31. **How to reverse a NumPy array?**
    `arr[::-1]`. The negative step reverses traversal direction.

### 🎯 Advanced questions

- **`arr[:, 0]` vs `arr[:, 0:1]` — what's the difference?**
  - `arr[:, 0]` → shape `(m,)` — 1D, the dimension collapses.
  - `arr[:, 0:1]` → shape `(m, 1)` — still 2D, dimension preserved.
  When you need column-vector arithmetic with broadcasting later, you usually want the second form.
- **What does `arr[..., 0]` mean?**
  The `...` (Ellipsis) is shorthand for "as many `:` as needed." For a 3D array `arr[..., 0]` ≡ `arr[:, :, 0]`. Convenient for code that handles arrays of unknown dimensionality.
- **Does slicing create a copy?**
  No — **slicing returns a view**. Modifying the slice modifies the original. See next section.
- **What does `np.newaxis` do?**
  Adds a new dimension of size 1: `arr[:, np.newaxis]` turns a `(m,)` into `(m, 1)`. Same as `arr[:, None]`.

[🔝 Back to top](#top)

---

<a id="11-views-copies"></a>
## 11. ⚠️ Views vs copies — the #1 NumPy gotcha

This is **not in your notebook**, but it's the most common NumPy bug in real interviews and real code. Master it.

### 🧠 Mental model

When you slice a NumPy array, you don't get a fresh array — you get a **view** into the original's memory. The two share the same underlying data. Modify one, the other changes too.

### 📐 Demonstration

```python
arr = np.array([1, 2, 3, 4, 5])
sub = arr[1:4]          # view — shares memory with arr
sub[0] = 999            # modify the view...
print(arr)              # → [1, 999, 3, 4, 5]   ← original changed!

# Compare to:
sub = arr[1:4].copy()   # explicit copy
sub[0] = 999
print(arr)              # → [1, 2, 3, 4, 5]     ← unchanged
```

### The rules

| Operation | Returns |
|---|---|
| Basic slicing (`arr[1:4]`) | **View** |
| Fancy indexing (`arr[[0, 2, 4]]`) | **Copy** |
| Boolean indexing (`arr[arr > 5]`) | **Copy** |
| `arr.reshape(...)` | View if possible, copy otherwise |
| `arr.copy()` | Always a copy |
| `arr.flatten()` | Always a copy |
| `arr.ravel()` | View if possible, copy otherwise |
| `arr.T` (transpose) | View |
| `astype()` | Always a copy |

### ❓ Basic questions

32. **Does slicing return a view or a copy?**
    Basic slicing → view. Fancy/boolean indexing → copy.

33. **How do you check if `b` is a view of `a`?**
    `b.base is a` returns `True` if `b` is a view of `a`. Or `np.shares_memory(a, b)`.

34. **How do you force a copy?**
    `arr[1:4].copy()` or `np.copy(arr[1:4])`.

### 🎯 Advanced questions

- **Why does NumPy default to views? Isn't that dangerous?**
  Performance. Copying every slice would be wasteful — most slices are read-only inspections. Views are basically free (just a new pointer + new shape/stride). Copies are expensive.
- **Subtle bug: explain what's wrong here.**
  ```python
  def normalize(arr):
      arr -= arr.mean()       # in-place subtraction
      arr /= arr.std()
      return arr

  x = np.array([1., 2., 3., 4., 5.])
  y = normalize(x[1:4])
  print(x)   # → x has been modified! Because x[1:4] is a view.
  ```
  Fix: `arr = arr.astype(float).copy()` at the top of the function, or use out-of-place operations (`arr = arr - arr.mean()`).
- **`a += 1` vs `a = a + 1` — why does it matter for views?**
  `a += 1` is in-place — it writes to the existing buffer (so if `a` is a view, the underlying array is mutated). `a = a + 1` allocates a new array and rebinds the name (the original is untouched).

[🔝 Back to top](#top)

---

<a id="12-boolean"></a>
## 12. Boolean indexing

Also not in your notebook yet but standard interview material.

### 🧠 Mental model

Pass an array of booleans (same shape) and you get back only the elements where the boolean is `True`. Combined with comparison operators, this is how you do filtering in NumPy.

### 📐 The mechanics

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])

mask = votes > 500                # → [True, True, True, False, False, False, True, False, True, False]
popular = votes[mask]             # → [775, 787, 918, 2556, 504]

# Inline:
votes[votes > 500]                # same thing, one-liner

# Multiple conditions (use & | not and/or)
votes[(votes > 200) & (votes < 1000)]   # → [775, 787, 918, 286, 324, 504, 402]

# Negation
votes[~(votes > 500)]             # everything NOT > 500

# Counting matches
(votes > 500).sum()               # 5  — True is 1, False is 0
(votes > 500).any()               # True if at least one match
(votes > 500).all()               # True if every element matches
```

### ❓ Basic questions

35. **How would you get all restaurants with more than 500 votes?**
    `votes[votes > 500]`.

36. **Why do you have to use `&` and `|` instead of `and` and `or`?**
    `and`/`or` work on single booleans; they're not element-wise. `&` and `|` are the element-wise versions and operate on the whole boolean array at once.

37. **Why do you need parentheses around each condition?**
    `&` has higher operator precedence than `>`, so `a > 0 & a < 5` parses wrong. Wrap each comparison: `(a > 0) & (a < 5)`.

### 🎯 Advanced questions

- **`np.where` vs boolean indexing — when to use which?**
  Boolean indexing **extracts** matching elements: `votes[votes > 500]`. `np.where` is more flexible — it returns indices or lets you do *conditional replacement*:
  ```python
  np.where(votes > 500, votes, 0)   # keep big votes, replace small with 0
  ```
- **How do you count matches without building the filtered array?**
  `(votes > 500).sum()` — far cheaper than `len(votes[votes > 500])` because there's no extraction.
- **Assignment via boolean mask?**
  ```python
  votes[votes < 100] = 0     # zero-out tiny-vote restaurants
  ```

[🔝 Back to top](#top)

---

<a id="13-broadcasting"></a>
## 13. Broadcasting

Also not in your notebook but **always** asked. Worth a section.

### 🧠 Mental model

When you do math between arrays of different shapes, NumPy automatically "stretches" the smaller one to match the larger one — *without actually copying data*. This is **broadcasting**.

The simplest case is array × scalar:
```python
votes * 2     # the scalar 2 is broadcast to every element of votes
```

The general rule: shapes are compared from the **right**. Dimensions are compatible if they're **equal** or one of them is **1**.

### 📐 Worked example

```python
# Restaurants × features (5 restaurants, 2 cols: votes, cost)
data = np.array([[100, 800],
                 [200, 600],
                 [150, 500],
                 [300, 700],
                 [250, 550]])         # shape (5, 2)

means = data.mean(axis=0)              # shape (2,)  → [200, 630]
centered = data - means                # broadcast! (5,2) - (2,) → (5,2)
```

Visualizing the alignment:
```
data.shape:    (5, 2)
means.shape:      (2,)         ← right-align
                  ↑
                 same → OK
```

### Common broadcastable pairs

| Shape A | Shape B | Result |
|---|---|---|
| `(5, 2)` | `(2,)` | `(5, 2)` ✓ |
| `(5, 2)` | `(5, 1)` | `(5, 2)` ✓ |
| `(5, 2)` | `(1, 2)` | `(5, 2)` ✓ |
| `(5, 2)` | `(3,)` | ❌ Error |
| `(5, 2)` | `(5,)` | ❌ Error (right-align mismatch) |

### ❓ Basic questions

38. **What is broadcasting?**
    Automatic shape alignment in NumPy operations — small arrays "stretch" to match larger ones without copying memory.

39. **Why does `votes * 2` work even though `2` is a scalar?**
    The scalar is broadcast to match the array's shape; conceptually it becomes `[2, 2, 2, ...]`.

### 🎯 Advanced questions

- **You have a `(5, 2)` data array and want to subtract the per-column mean. How?**
  ```python
  data - data.mean(axis=0)      # (5,2) - (2,) → broadcasts
  ```
- **You have a `(5, 2)` data array and want to subtract the per-row mean. How?**
  ```python
  data - data.mean(axis=1, keepdims=True)   # (5,2) - (5,1) → broadcasts
  # Without keepdims you'd get (5,) which won't broadcast against (5,2)!
  ```
- **What does `keepdims=True` do?**
  Keeps the reduced dimension as size 1 instead of dropping it. Crucial for broadcasting reductions back against the original.
- **Outer product via broadcasting?**
  ```python
  a = np.array([1, 2, 3])          # (3,)
  b = np.array([10, 20, 30, 40])   # (4,)
  a[:, None] * b[None, :]          # (3, 1) × (1, 4) → (3, 4)
  ```
- **Does broadcasting allocate memory?**
  No — it's "virtual" stretching during the operation. The output is allocated, but the broadcast itself does not duplicate the input.

[🔝 Back to top](#top)

---

<a id="14-aggregations"></a>
## 14. Aggregations and axis

### 🧠 Mental model

Aggregations summarize an array into fewer numbers. The `axis` argument tells NumPy *which dimension to collapse*.

- `axis=0` → collapse rows → one value per column
- `axis=1` → collapse columns → one value per row
- No axis → collapse everything → one scalar

### 📐 The mechanics

```python
data = np.array([[775, 800],
                 [787, 600],
                 [918, 500]])         # shape (3, 2)

data.sum()              # 4380       total
data.sum(axis=0)        # [2480, 1900]   col sums (3 rows collapsed)
data.sum(axis=1)        # [1575, 1387, 1418]   row sums (2 cols collapsed)

data.mean(axis=0)       # mean per column
data.max(axis=1)        # max per row
data.argmax()           # index of overall max (flat index)
```

Common aggregations: `sum`, `mean`, `min`, `max`, `std`, `var`, `median`, `argmin`, `argmax`, `cumsum`, `cumprod`, `prod`, `ptp` (peak-to-peak / range).

### ❓ Basic questions

40. **What's the difference between `axis=0` and `axis=1`?**
    `axis=0` collapses rows (column-wise op); `axis=1` collapses columns (row-wise op). Memory aid: "axis 0 disappears, columns remain → column-wise."

41. **For a `(3, 4)` array, what's the shape after `.sum(axis=0)`? After `.sum(axis=1)`?**
    `sum(axis=0)` → shape `(4,)` (4 column sums).
    `sum(axis=1)` → shape `(3,)` (3 row sums).

### 🎯 Advanced questions

- **What does `argmax()` return without an axis?**
  The flat (1D) index of the maximum. Use `np.unravel_index(arr.argmax(), arr.shape)` to convert to a (row, col) coordinate.
- **Difference between `np.sum(arr)` and `arr.sum()`?**
  Functionally identical. The method form is slightly more pythonic and slightly faster (no function lookup). Use either.
- **`std` and `var` — what's the default `ddof`?**
  `ddof=0` (population std). For a *sample* std use `ddof=1` (matches Pandas default).
- **`cumsum` vs `sum`?**
  `sum` returns one number. `cumsum` returns a running total — same shape as input. Useful for time series cumulative metrics.
- **How to compute a weighted mean?**
  `np.average(arr, weights=w)`. Don't confuse with `np.mean`, which has no weights argument.

[🔝 Back to top](#top)

---

<a id="15-reshape"></a>
## 15. Reshape, flatten, ravel, transpose

### 🧠 Mental model

Same data, different shape. The total number of elements (`size`) must stay the same.

### 📐 The mechanics

```python
arr = np.arange(12)           # shape (12,)

arr.reshape(3, 4)             # shape (3, 4) — view if possible
arr.reshape(3, -1)            # NumPy infers the -1 → (3, 4)
arr.reshape(-1, 2)            # → (6, 2)

mat = arr.reshape(3, 4)
mat.flatten()                 # → 1D (12,) — always a copy
mat.ravel()                   # → 1D (12,) — view if possible

mat.T                         # transpose — swaps axes, view
mat.transpose()               # same as .T for 2D
mat.transpose(1, 0)           # explicit axis order — for >2D

np.expand_dims(arr, axis=0)   # add a dimension
np.squeeze(arr)               # remove size-1 dimensions
```

### ❓ Basic questions

42. **What's the `-1` trick in reshape?**
    NumPy infers the missing dimension. `arr.reshape(3, -1)` says "3 rows, you figure out the columns."

43. **Flatten vs ravel?**
    Both flatten an array to 1D. `flatten` always returns a copy. `ravel` returns a view when possible (faster but mutating affects the original).

44. **What does the transpose attribute `.T` do?**
    Swaps axes. For a 2D array, swaps rows ↔ columns. For higher-D, reverses the axis order. It's a view, not a copy.

### 🎯 Advanced questions

- **Why does reshape sometimes return a copy?**
  When the new shape can't be represented by just adjusting strides (e.g., reshaping a transposed array). NumPy falls back to copying.
- **What is C-order vs F-order?**
  C-order (row-major, default in NumPy and C/Python) stores rows contiguously. F-order (column-major, used by Fortran/MATLAB/R) stores columns contiguously. Matters for performance — iterate the direction your data is stored.
- **`squeeze` vs `expand_dims`?**
  `squeeze` removes axes of length 1. `expand_dims` inserts a new axis of length 1. Both are common for fixing shape mismatches before broadcasting/matmul.
- **For a 3D `(2, 3, 4)` array, what does `.T` give you?**
  `(4, 3, 2)` — reverses *all* axes. For more control use `transpose(0, 2, 1)` etc.

[🔝 Back to top](#top)

---

<a id="16-stacking"></a>
## 16. Stacking, splitting, concatenation

### 🧠 Mental model

You'll often need to glue arrays together (stacking/concatenating) or break one apart (splitting). The three keywords to remember are **concatenate, stack, split** — plus their `v*`/`h*` axis-fixed cousins.

### 📐 The mechanics

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])              # → [1,2,3,4,5,6]   joins along existing axis
np.stack([a, b])                    # → [[1,2,3],[4,5,6]]   NEW axis (shape (2,3))
np.stack([a, b], axis=1)            # → [[1,4],[2,5],[3,6]] (shape (3,2))
np.vstack([a, b])                   # vertical stack — adds rows
np.hstack([a, b])                   # horizontal — flat for 1D, side-by-side for 2D
np.column_stack([a, b])             # 1D → columns of a 2D matrix
np.dstack([a, b])                   # depth (3rd axis) stack

# Splitting
arr = np.arange(12).reshape(3, 4)
np.split(arr, 2, axis=1)            # split into 2 along cols → list of 2 arrays
np.hsplit(arr, 2)                   # same as split axis=1
np.vsplit(arr, 3)                   # split into 3 along rows
np.array_split(arr, 3, axis=1)      # tolerates uneven splits (split errors)
```

### Key distinction

- **`concatenate`** joins along an **existing** axis (shapes must match on every other axis).
- **`stack`** creates a **new** axis (all inputs must have identical shape).

### ❓ Basic questions

45. **`concatenate` vs `stack` — what's the difference?**
    `concatenate` joins along an *existing* axis; `stack` creates a *new* axis.

46. **What does `vstack` do for two 1D arrays of shape `(3,)`?**
    Stacks them as rows → shape `(2, 3)`.

47. **What does `hstack` do for two 1D arrays of shape `(3,)`?**
    Flat join → shape `(6,)`. (For 2D inputs, it joins side-by-side.)

### 🎯 Advanced questions

- **`split` vs `array_split`?**
  `split` requires the array to divide evenly into N pieces; otherwise raises. `array_split` allows uneven splits — last chunk(s) absorb the remainder. Use `array_split` for "give me N chunks no matter what."
- **What's `column_stack` good for?**
  Building a 2D feature matrix from a list of 1D columns. Each input becomes one column. Common pattern when assembling features for ML.
- **How do you append a single row to a 2D array?**
  ```python
  new_row = np.array([1, 2, 3])
  np.vstack([data, new_row])         # easy & explicit
  ```
  Note: `np.append` exists but always copies — preallocate when you can.

[🔝 Back to top](#top)

---

<a id="17-sorting"></a>
## 17. Sorting, unique, set operations, searching

### 🧠 Mental model

Sorting and searching are bread-and-butter NumPy operations — and *every* one of them comes in two flavors: returning the **values** (`sort`) or returning the **indices** (`argsort`).

### 📐 The mechanics

```python
votes = np.array([775, 787, 918, 88, 166])

# Sorting
np.sort(votes)                     # → [88, 166, 775, 787, 918]  (out-of-place)
votes.sort()                       # in-place — modifies votes
np.argsort(votes)                  # → indices that would sort: [3, 4, 0, 1, 2]
votes[np.argsort(votes)]           # → sorted array (same as np.sort)

# Descending sort — no direct flag, slice with [::-1]
np.sort(votes)[::-1]

# Sort by one column, keep rows together
data = np.array([[3, 100], [1, 200], [2, 150]])
data[data[:, 0].argsort()]         # sort rows by first column

# Unique
np.unique(np.array([1, 2, 2, 3, 3, 3]))            # → [1, 2, 3]
np.unique(arr, return_counts=True)                  # → ([1,2,3], [1,2,3])  (value_counts)

# Set operations
np.intersect1d(a, b)               # values in both
np.union1d(a, b)                   # values in either
np.setdiff1d(a, b)                 # in a but not b
np.in1d(a, b)                      # boolean mask: which a's are in b

# Searching
np.searchsorted(sorted_arr, val)   # binary search — where would val go?
np.argmin(arr)                     # index of min
np.argmax(arr)                     # index of max
np.nonzero(arr > 500)              # indices where condition is true
```

### ❓ Basic questions

48. **How do you sort an array descending?**
    `np.sort(arr)[::-1]`. There's no `reverse=True` arg.

49. **What's the difference between `sort` and `argsort`?**
    `sort` returns sorted *values*; `argsort` returns the *indices* that would sort. `arr[arr.argsort()] == np.sort(arr)`.

50. **How do you get unique values and their counts in one call?**
    `np.unique(arr, return_counts=True)`. This is the NumPy equivalent of Pandas' `value_counts()`.

### 🎯 Advanced questions

- **How would you sort a 2D array by one of its columns and keep the rows together?**
  ```python
  data[data[:, 0].argsort()]         # sort by column 0
  ```
- **What's `np.searchsorted` and when is it useful?**
  Binary search in a sorted array — returns where a value *would* be inserted. Used for bucketing/binning continuous values, e.g., assigning cost to a price band.
- **What's the time complexity of `np.sort`?**
  O(n log n). Default is quicksort; `kind='mergesort'` is stable; `kind='stable'` is alias for mergesort.
- **`np.unique` on a 2D array — what does it return?**
  By default, flattens. Pass `axis=0` for unique rows or `axis=1` for unique columns.

[🔝 Back to top](#top)

---

<a id="18-nan"></a>
## 18. NaN & missing data handling

### 🧠 Mental model

`np.nan` is a **floating-point sentinel** for "not a number". It propagates: any arithmetic involving `NaN` returns `NaN`. Aggregations like `mean` return `NaN` if any element is `NaN` — unless you use the `nan*` variant.

### 📐 The mechanics

```python
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

# Detection
np.isnan(arr)                  # → [F, F, T, F, T]
np.isnan(arr).sum()            # 2 — count of NaNs

# DON'T do this — NaN != NaN!
arr == np.nan                  # → all False  ❌

# NaN-aware aggregations
arr.mean()                     # nan ❌
np.nanmean(arr)                # 2.33 ✓ ignores NaN
np.nansum(arr)                 # 7.0
np.nanstd(arr)
np.nanmedian(arr)
np.nanmin(arr), np.nanmax(arr)

# Replacement
clean = np.where(np.isnan(arr), 0, arr)   # replace NaN with 0
arr[np.isnan(arr)] = 0                    # in-place

# Drop NaN
arr[~np.isnan(arr)]                       # → [1.0, 2.0, 4.0]

# Infinity
np.isinf(arr)
np.isfinite(arr)                          # not NaN and not inf
```

### ❓ Basic questions

51. **Why doesn't `arr == np.nan` work?**
    Because `NaN` compared to anything (including itself) is `False`. Use `np.isnan(arr)`.

52. **What does `mean()` return if the array contains a NaN?**
    `NaN`. Use `np.nanmean()` to skip NaNs.

53. **How do you count missing values in an array?**
    `np.isnan(arr).sum()`.

### 🎯 Advanced questions

- **Why is `np.nan` a float, not an int?**
  Integer types have no representation for NaN. If you have integer data with missing values, NumPy will silently upcast to float when you introduce a NaN. Pandas has a separate "nullable int" dtype to avoid this.
- **`np.isnan` vs `pd.isna` — when to use which?**
  `np.isnan` only works for floats (errors on object/string). `pd.isna` works for *anything* — floats, strings, datetimes, objects. Use `pd.isna` when types are mixed.
- **Difference between `inf` and `nan`?**
  `inf` = mathematical infinity (e.g., `1/0` → `inf`). `nan` = undefined (e.g., `0/0` → `nan`). Both are floats. `np.isfinite` filters both out.
- **Best practice for imputation in NumPy?**
  Compute the replacement statistic on the *non-NaN* portion: `np.nan_to_num(arr, nan=np.nanmean(arr))`. Don't compute mean on the whole array (it'll be NaN).

[🔝 Back to top](#top)

---

<a id="19-where-clip"></a>
## 19. `np.where`, `clip`, `select`, percentiles, conditional ops

### 🧠 Mental model

These are the **conditional / threshold** workhorses. `where` and `select` for "if X then Y else Z" logic; `clip` for bounding values; `percentile`/`quantile` for distributional summaries.

### 📐 The mechanics

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])

# np.where — conditional replacement
np.where(votes > 500, "popular", "niche")
np.where(votes > 500, votes, 0)              # keep big votes, zero else

# Multi-condition labelling
conditions = [votes < 200, votes < 600, votes < 1500]
choices    = ["low",     "medium",   "high"]
np.select(conditions, choices, default="viral")

# np.clip — bound values into a range
np.clip(votes, 100, 1000)                    # values < 100 → 100, > 1000 → 1000

# Percentiles & quantiles
np.percentile(votes, 50)                     # median
np.percentile(votes, [25, 50, 75])           # Q1, median, Q3 — gives quartiles
np.quantile(votes, 0.95)                     # same idea, 0–1 scale

# Other useful ones
np.diff(votes)                               # consecutive differences
np.cumsum(votes)                             # running total
np.maximum(a, b)                             # elementwise max
np.minimum(a, b)                             # elementwise min
np.sign(arr)                                 # -1, 0, +1
np.abs(arr)
```

### ❓ Basic questions

54. **What does `np.where(cond, a, b)` return?**
    For each element: `a` where `cond` is True, `b` where it's False. Same shape as broadcast.

55. **How do you cap (clip) values into [0, 100]?**
    `np.clip(arr, 0, 100)`.

56. **How do you compute the 95th percentile?**
    `np.percentile(arr, 95)` or `np.quantile(arr, 0.95)`.

### 🎯 Advanced questions

- **`np.where(cond)` with only one argument — what does it return?**
  The *indices* where `cond` is True. Equivalent to `np.nonzero(cond)`. Useful when you need positions, not values.
- **`np.maximum` vs `np.max`?**
  `np.max(arr)` aggregates one array into a scalar (or along an axis). `np.maximum(a, b)` is **element-wise** comparison of two arrays.
- **`np.percentile` vs `np.quantile`?**
  Same function, different scale: percentile takes 0–100, quantile takes 0–1.
- **What's `interpolation` (or `method` in newer NumPy) in `np.percentile`?**
  Controls how percentile is computed when the index falls between elements. Default `'linear'`. Options: `nearest`, `lower`, `higher`, `midpoint`. Matters when you want an *actual* value vs an interpolated one.

[🔝 Back to top](#top)

---

<a id="20-linalg"></a>
## 20. Linear algebra essentials

### 🧠 Mental model

NumPy's linear algebra module (`np.linalg`) is the foundation of basically every ML algorithm under the hood. You need to be fluent in matrix multiplication, dot products, norms, inverses, and decompositions.

### 📐 The mechanics

```python
# Dot product / matrix multiplication
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a @ b                              # 32 — dot product (1*4 + 2*5 + 3*6)
np.dot(a, b)                       # same
a.dot(b)                           # same

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                              # 2D matrix multiply
A * B                              # ❗ element-wise, NOT matmul

# Properties
A.T                                # transpose
np.linalg.det(A)                   # determinant
np.linalg.inv(A)                   # inverse
np.linalg.matrix_rank(A)           # rank
np.trace(A)                        # sum of diagonal

# Norms
np.linalg.norm(a)                  # L2 norm (default) — sqrt(sum(x^2))
np.linalg.norm(a, ord=1)           # L1 norm — sum(|x|)
np.linalg.norm(a, ord=np.inf)      # max(|x|)

# Solve Ax = b
x = np.linalg.solve(A, b)          # faster & more stable than inv(A) @ b

# Eigenvalues / decompositions (used in PCA, etc.)
np.linalg.eig(A)                   # eigenvalues, eigenvectors
np.linalg.svd(A)                   # singular value decomposition
```

### ❓ Basic questions

57. **What's the difference between `A * B` and `A @ B`?**
    `*` is **element-wise** (broadcasts); `@` is **matrix multiplication**. Confusing them is a classic bug.

58. **What's the dot product of two vectors?**
    Sum of element-wise products. `[1,2,3] · [4,5,6] = 1·4 + 2·5 + 3·6 = 32`.

59. **How do you compute the L2 norm of a vector?**
    `np.linalg.norm(v)`. Equivalent to `np.sqrt((v**2).sum())`.

### 🎯 Advanced questions

- **Why prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`?**
  `solve` is faster (uses LU decomposition) and more numerically stable. Inverting is slower and amplifies floating-point error.
- **Shape rule for `A @ B`?**
  If `A` is `(m, k)` and `B` is `(k, n)`, then `A @ B` is `(m, n)`. The inner dimensions must match.
- **What does SVD give you and where is it used?**
  Factorizes `A = U Σ Vᵀ`. Used in PCA, recommender systems, image compression, latent semantic indexing.
- **Element-wise `**2` vs `A @ A` for a matrix?**
  `A ** 2` squares each element; `A @ A` is true matrix self-multiplication. Completely different results.

[🔝 Back to top](#top)

---

<a id="21-strides"></a>
## 21. Strides, memory layout & performance internals

### 🧠 Mental model

Under the hood, every NumPy array is just **one contiguous buffer** of bytes plus a `shape` and a `strides` tuple. The shape tells you "how many in each dimension"; strides tell you **how many bytes to move to step one element along each axis**.

This is the secret behind why slicing is free (just adjust strides), why transposing is free (swap strides), and why some reshapes need to copy (the desired strides aren't expressible without rearranging memory).

### 📐 The mechanics

```python
arr = np.arange(12, dtype=np.int64).reshape(3, 4)

arr.shape         # (3, 4)
arr.strides       # (32, 8)   — row jump is 4*8=32 bytes; col jump is 8
arr.itemsize      # 8         — int64 = 8 bytes
arr.flags         # info on layout: C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA, WRITEABLE
arr.flags['C_CONTIGUOUS']    # True (row-major default)

# Transposing just swaps strides — no data movement
arr.T.strides     # (8, 32)
arr.T.flags['C_CONTIGUOUS']  # False — it's F-contiguous now
```

### ❓ Basic questions

60. **What are strides?**
    Byte offsets to step one position along each axis. They describe how NumPy walks the underlying buffer.

61. **Is transpose a view or a copy?**
    A view. Transposing only swaps the entries of `strides` — the buffer is unchanged.

### 🎯 Advanced questions

- **Why does `arr.T.reshape(...)` sometimes return a copy?**
  After transpose the array is no longer C-contiguous. If the requested reshape can't be expressed with new strides on the current buffer, NumPy has to allocate a new contiguous buffer and copy.
- **What does `np.ascontiguousarray` do?**
  Returns a C-contiguous version of the array (copies if necessary). Used before passing to C code that expects row-major layout.
- **C-order vs F-order — which is faster to iterate?**
  Depends on the access pattern. **Always iterate the inner stride.** For C-order, that's column index changes fastest → `for row: for col`. For F-order, swap. Wrong order = cache miss every step.
- **What's a "broadcasting view"?**
  Some broadcasts can be implemented by setting a stride to 0 — that axis "repeats" without copying. Inspect with `np.broadcast_to(a, shape)`.

[🔝 Back to top](#top)

---

<a id="22-io"></a>
## 22. I/O — saving and loading

### 🧠 Mental model

For pure NumPy arrays, use the binary `.npy` / `.npz` formats — they round-trip dtype and shape exactly and are fast. For text files use `loadtxt`/`savetxt` (or better, Pandas).

### 📐 The mechanics

```python
# Binary — preferred for arrays
np.save('votes.npy', votes)              # saves one array
loaded = np.load('votes.npy')

np.savez('multi.npz', v=votes, c=costs)  # multiple arrays in one file
data = np.load('multi.npz')
data['v']                                # access by name

np.savez_compressed('big.npz', a=big)    # compressed — slower but smaller

# Text
np.savetxt('votes.csv', votes, delimiter=',', fmt='%d')
np.loadtxt('votes.csv', delimiter=',')
np.genfromtxt('messy.csv', delimiter=',', skip_header=1, missing_values='NA', filling_values=0)
```

### ❓ Basic questions

62. **What's the difference between `.npy` and `.npz`?**
    `.npy` stores a single array; `.npz` is a zip of multiple `.npy`s (you access by keyword name).

63. **Why use `np.save` over `np.savetxt`?**
    Binary `.npy` round-trips dtype and shape exactly and is much faster. Text is for human-readable interchange only.

### 🎯 Advanced questions

- **`loadtxt` vs `genfromtxt` — when to use which?**
  `loadtxt` is fast but strict (no missing values). `genfromtxt` handles missing values, mixed types, and irregular formats — slower but flexible.
- **For very large arrays, what's `np.memmap`?**
  Memory-map a file as an array — operate on it without loading the whole thing into RAM. Used for out-of-core processing of multi-GB datasets.

[🔝 Back to top](#top)

---

<a id="23-datetime"></a>
## 23. Datetime & timedelta arrays

### 🧠 Mental model

NumPy has native datetime support via `datetime64` and `timedelta64`. In practice, you'll use Pandas for most date work (richer API), but knowing the NumPy primitives helps when reading docs or debugging.

### 📐 The mechanics

```python
dates = np.array(['2026-01-01', '2026-02-15', '2026-03-30'], dtype='datetime64')

dates.dtype                                  # dtype('<M8[D]')  — day-precision
dates + np.timedelta64(7, 'D')               # add 7 days
dates[1] - dates[0]                          # timedelta64(45, 'D')

# Different precisions
np.datetime64('2026-01-01')                  # day
np.datetime64('2026-01-01T12:00')            # minute
np.datetime64('2026-01-01T12:00:00.001')     # millisecond

# Range
np.arange('2026-01', '2026-04', dtype='datetime64[M]')   # monthly range
```

### ❓ Basic questions

64. **What dtype does NumPy use for dates?**
    `datetime64` (and `timedelta64` for durations).

### 🎯 Advanced questions

- **Why prefer Pandas for date work?**
  Pandas has `DatetimeIndex` with timezone support, `.dt` accessor (`.dt.year`, `.dt.dayofweek`), business-day calendars, frequency strings (`'D'`, `'B'`, `'M'`, `'W-MON'`), resampling, and full integration with grouping/plotting. NumPy's `datetime64` is the primitive; Pandas is the workhorse.

[🔝 Back to top](#top)

---

<a id="24-tricks"></a>
## 24. Common trick / multi-correct questions

Memorize these patterns — they appear in tests verbatim.

### Q1. Which method converts a float array to integers?
```python
arr = np.array([1.7, 2.8, 3.9])
```
- ✅ `arr.astype(int)`
- ❌ `arr.toInt()` (no such method)
- ❌ `int(arr)` (works for single numbers, not arrays)
- ❌ `astype(arr, int)` (wrong calling convention)

### Q2. In a `(m, n)` array, what shape is returned by `arr[:, 0]`?
- ❌ `(1, n)`
- ✅ **`(m,)`**
- ❌ `(m, 1)`
- ❌ `(n,)`

Selecting a single column with an integer index **collapses** that dimension to 1D.

### Q3. NumPy type priority — what becomes of `np.array([True, 6, 3.14, "x"])`?
All become strings. Priority: **String > Float > Int > Bool**.

### Q4. What's the result of `np.array([4, 5, 6]) * 2`?
- ✅ `array([8, 10, 12])` (element-wise multiplication)
- ❌ `array([4, 5, 6, 4, 5, 6])` (that's the Python *list* behavior)

### Q5. What does `arr[::-1]` do?
Reverses the array (negative step traversal).

### Q6. `arr[:5]` vs `arr[0:5]` — same or different?
**Same.** Omitted start defaults to 0.

### Q7. Slicing returns a view or a copy? Boolean indexing returns a view or a copy?
- Basic slicing → **view**
- Boolean indexing → **copy**
- Fancy indexing (list of indices) → **copy**

### Q8. For a `(3, 4)` array, what's the shape after `.sum(axis=0)`?
`(4,)` — 4 column sums. Axis 0 (rows) gets collapsed.

### Q9. What does `dtype='<U7'` indicate?
Little-endian Unicode string, max 7 characters. Tells you the array got coerced to strings (almost always because input had a string).

### Q10. Does `astype(int)` round or truncate?
**Truncates.** `1.9 → 1`, not 2.

### Q11. `A * B` vs `A @ B` for two 2D arrays?
`*` is element-wise; `@` is matrix multiplication. Different math.

### Q12. `arr == np.nan` — what does it return?
**All False** — `NaN` is never equal to anything, including itself. Use `np.isnan(arr)`.

### Q13. `np.stack` vs `np.concatenate`?
`stack` creates a **new** axis; `concatenate` joins along an **existing** axis.

### Q14. `np.sort` vs `np.argsort`?
`sort` returns sorted values; `argsort` returns the indices that would sort.

### Q15. `np.maximum(a, b)` vs `np.max(a)`?
`maximum` is element-wise (two arrays); `max` is an aggregator (one array → scalar/axis).

[🔝 Back to top](#top)

---

<a id="25-flashcards"></a>
## 25. Rapid-fire flashcards

| Q | A |
|---|---|
| What does NumPy stand for? | Numerical Python |
| What's the core data structure? | `ndarray` (n-dimensional array) |
| Two reasons NumPy is faster than lists? | Contiguous memory + vectorized C operations |
| `[4,5,6] * 2` vs `np.array([4,5,6]) * 2`? | `[4,5,6,4,5,6]` vs `[8,10,12]` |
| Type priority? | String > Float > Int > Bool |
| `arr.shape` of a 10-element 1D array? | `(10,)` |
| What does `.ndim` return? | Number of dimensions |
| What does `.size` return? | Total element count |
| What does `dtype='<U3'` mean? | Unicode string, max 3 chars |
| Convert float → int? | `arr.astype(int)` (truncates!) |
| Negative index for last element? | `arr[-1]` |
| Slicing end index — inclusive or exclusive? | **Exclusive** |
| Reverse an array? | `arr[::-1]` |
| 2D indexing pattern? | `arr[row, col]` |
| Shape of `arr[:, 0]` for `(m, n)`? | `(m,)` |
| Shape of `arr[:, 0:1]` for `(m, n)`? | `(m, 1)` |
| Basic slicing returns view or copy? | **View** |
| Fancy/boolean indexing returns view or copy? | **Copy** |
| How to force a copy? | `arr.copy()` |
| Filter elements > 500? | `arr[arr > 500]` |
| Why `&` not `and` in boolean masks? | `and` isn't element-wise |
| What is broadcasting? | Auto shape-matching for math between different-shaped arrays |
| `axis=0` collapses what? | Rows (one value per column) |
| `axis=1` collapses what? | Columns (one value per row) |
| `-1` in reshape? | Infer that dimension |
| Flatten vs ravel? | Flatten = copy, ravel = view when possible |
| Transpose syntax? | `arr.T` |
| `concatenate` vs `stack`? | Existing axis vs new axis |
| `np.unique` with counts? | `np.unique(arr, return_counts=True)` |
| Count NaNs? | `np.isnan(arr).sum()` |
| Mean ignoring NaNs? | `np.nanmean(arr)` |
| Cap values into a range? | `np.clip(arr, lo, hi)` |
| Conditional replace? | `np.where(cond, a, b)` |
| `A * B` vs `A @ B`? | Elementwise vs matmul |
| Strides? | Bytes to step along each axis |
| Reproducible randomness? | `np.random.seed(42)` |
| Save array binary? | `np.save('x.npy', arr)` |
| 95th percentile? | `np.percentile(arr, 95)` |

[🔝 Back to top](#top)

---

<a id="26-summary"></a>
## 26. 📚 Today's learning — quick summary

A condensed recap of everything covered in the notebook + this guide. Read this once at the end of every revision pass — it's your "what did I learn today" memory anchor.

### The big picture

You started with the **why of ML** (the loan story, the salary pattern) and ended with the **how of NumPy** — the numerical engine every ML/EDA workflow runs on. The thread tying it together: **ML needs patterns from data → data must be clean and explored → NumPy is the foundation that makes that fast.**

### Core takeaways

1. **ML = learning patterns from history.** Rule-based logic fails when the rules are too numerous; ML succeeds by learning them from examples. Supervised, unsupervised, reinforcement are the three flavors.

2. **EDA is non-negotiable.** Before any model, you check the data: shape, dtypes, missingness, distributions, relationships. The Zomato example: messy strings like `"'800.0'"`, `"1,200"`, `"4.1/5"` — all need cleaning before math.

3. **Python list ≠ NumPy array.** Same `* 2` operator, different behaviour. Lists replicate; arrays multiply element-wise. Arrays win for math, lists win for heterogeneity and append-heavy code.

4. **NumPy is fast because of three things:** (a) contiguous memory, (b) homogeneous types, (c) vectorized C/SIMD operations. The tea-room analogy: ingredients on one shelf, not scattered.

5. **Arrays have four key attributes:** `shape`, `ndim`, `size`, `dtype`. Plus `nbytes` and `itemsize` for memory.

6. **Type priority:** `String > Float > Int > Bool`. Mixed-type lists get coerced to the highest. This causes silent bugs when a single stray string makes your whole numeric column into strings.

7. **`astype` returns a new array, doesn't modify in place. Truncates floats — does not round.** Clean strings first (`np.char.replace`), then convert.

8. **Indexing:** positive, negative, fancy (list of indices), 2D = `[row, col]`. Fancy indexing → copy. Basic slicing → view.

9. **Slicing end index is exclusive.** `arr[:5]` is 5 elements (indices 0–4). For 2D: `arr[:, 0]` gives `(m,)` (1D), `arr[:, 0:1]` gives `(m, 1)` (2D).

10. **Views vs copies** is the #1 NumPy gotcha. Basic slice = view (mutating affects the original). Fancy/boolean indexing = copy. Use `.copy()` when in doubt.

11. **Boolean indexing** filters: `arr[arr > 500]`. Combine with `&`, `|`, `~` (not `and`, `or`, `not`). Wrap each comparison in parens.

12. **Broadcasting** auto-aligns shapes from the right. Dimensions match if equal or one is 1. Use `keepdims=True` when broadcasting a reduction back against the original.

13. **Aggregations:** `axis=0` collapses rows → per-column result. `axis=1` collapses columns → per-row result. Memory aid: "axis that disappears."

14. **Reshape, flatten, ravel, transpose** — same data, different shape/view. `-1` in reshape infers the missing dim. `flatten` always copies; `ravel` views when it can; `.T` is always a view.

15. **Stacking & splitting:** `concatenate` joins along existing axes; `stack` creates new ones. `array_split` tolerates uneven divisions.

16. **Sorting:** `sort` gives values, `argsort` gives indices. To sort a 2D array by a column: `data[data[:, col].argsort()]`.

17. **NaN handling:** `np.isnan` (not `==`), and use `nanmean/nansum/nanstd` to skip them.

18. **Conditional ops:** `np.where(cond, a, b)`, `np.clip`, `np.select` for multi-branch, `np.percentile` for quantiles.

19. **Linear algebra:** `A * B` is element-wise; `A @ B` is matmul. Prefer `np.linalg.solve` over `np.linalg.inv`.

20. **Strides** are the secret behind why slicing/transposing are free — only `shape`/`strides` change, the buffer stays put.

### Mental anchors (the things that "click")

- **`* 2`** — the one example that explains lists vs arrays
- **Tea room** — the analogy for why NumPy is fast
- **Race line ends at the finish line** — slicing end is exclusive
- **Axis that disappears** — for axis= reductions
- **Right-align shapes** — for broadcasting

If you can re-explain those five anchors in 60 seconds each, you've internalized the module.

[🔝 Back to top](#top)

---

<a id="27-revision"></a>
## 27. 🔁 Revision drill — 10-minute speed-run

Designed as a **timed revision tool** — read the question, answer in your head (or out loud), then peek. Aim for under 20 seconds per question.

### Round 1 — Foundations (1 minute)

1. NumPy stands for? → **Numerical Python**
2. Core data structure? → **`ndarray`**
3. Why faster than lists? → **Contiguous memory + homogeneous dtype + vectorized C**
4. `[1,2,3] * 2` Python list? → **`[1,2,3,1,2,3]`**
5. `np.array([1,2,3]) * 2`? → **`[2,4,6]`**

### Round 2 — Shape & dtype (1 minute)

6. Shape of a 1D array with 10 elements? → **`(10,)`**
7. Difference between `shape` and `size`? → **Structure vs total count**
8. `dtype='<U5'` means? → **Unicode string max 5 chars**
9. Type priority order? → **String > Float > Int > Bool**
10. Bytes per element of int64? → **8**

### Round 3 — Indexing & slicing (1 minute)

11. Last element? → **`arr[-1]`**
12. Is slicing end inclusive? → **No, exclusive**
13. Reverse an array? → **`arr[::-1]`**
14. 2D indexing pattern? → **`arr[row, col]`**
15. Shape of `arr[:, 0]` for `(m, n)`? → **`(m,)`**
16. Shape of `arr[:, 0:1]`? → **`(m, 1)`**

### Round 4 — Views & copies (1 minute)

17. Basic slicing returns? → **View**
18. Fancy indexing returns? → **Copy**
19. Boolean indexing returns? → **Copy**
20. How to force a copy? → **`.copy()`**
21. Check if `b` is view of `a`? → **`b.base is a`**

### Round 5 — Boolean & broadcasting (1 minute)

22. Filter > 500? → **`arr[arr > 500]`**
23. Why `&` not `and`? → **`and` isn't element-wise**
24. Why parentheses in mask? → **`&` precedence is higher than `>`**
25. Broadcasting rule? → **Right-align, dims must be equal or one is 1**
26. `keepdims=True` purpose? → **Keep size-1 axis after reduction for broadcasting**

### Round 6 — Aggregations & reshape (1 minute)

27. `axis=0` collapses? → **Rows (column-wise)**
28. `axis=1` collapses? → **Columns (row-wise)**
29. `-1` in reshape? → **Infer dim**
30. Flatten vs ravel? → **Flatten copy; ravel view-if-possible**
31. Transpose syntax? → **`arr.T`**

### Round 7 — Advanced (1 minute)

32. `concatenate` vs `stack`? → **Existing axis vs new axis**
33. Sort 2D by column 0? → **`data[data[:, 0].argsort()]`**
34. Count NaNs? → **`np.isnan(arr).sum()`**
35. Mean ignoring NaNs? → **`np.nanmean(arr)`**
36. Cap into [0, 100]? → **`np.clip(arr, 0, 100)`**

### Round 8 — Linalg & memory (1 minute)

37. `A * B` 2D? → **Element-wise**
38. `A @ B` 2D? → **Matrix multiplication**
39. L2 norm? → **`np.linalg.norm(v)`**
40. Strides? → **Bytes to step along each axis**
41. Transpose copies? → **No, just swaps strides**

### Round 9 — Application traps (1 minute)

42. `astype(int)` rounds or truncates? → **Truncates**
43. `arr == np.nan` returns? → **All False**
44. Convert `"1,200"` to float? → **strip commas, `astype(float)`**
45. `int(arr)` valid? → **No, scalar only**
46. `astype` modifies original? → **No, returns new**

### Round 10 — Big picture (1 minute)

47. Why EDA before modelling? → **Remove irrelevant patterns + clean corrupt data**
48. Three ML types? → **Supervised, unsupervised, reinforcement**
49. AI vs ML vs DL? → **AI ⊃ ML ⊃ DL**
50. The four core libraries? → **NumPy, Pandas, Matplotlib, Seaborn**

**Score yourself:** 45+ = strong, 35–44 = mostly there, <35 = re-read the sections.

[🔝 Back to top](#top)

---

<a id="28-bestpractices"></a>
## 28. ✅ Best practices cheat sheet

The 25 do's and don'ts that separate junior NumPy code from production-grade. Skim before any interview or coding round.

### Performance

1. **Vectorize everything.** Replace `for` loops with array operations. `arr * 2` not `[x*2 for x in arr]`.
2. **Pre-allocate, don't append.** `np.empty(N)` then fill, not repeated `np.append` (which copies every time).
3. **Use the right dtype.** `int32` instead of `int64` halves memory if your range fits. `float32` is plenty for most ML.
4. **Use `np.linalg.solve(A, b)`** instead of `np.linalg.inv(A) @ b` — faster and more numerically stable.
5. **Iterate the inner stride.** For C-order arrays, the column index varies fastest. Don't transpose just to iterate the "natural" way.
6. **Use `np.empty` over `np.zeros`** when you'll overwrite everything anyway — skips initialization.
7. **Release Python with `out=` parameters.** `np.add(a, b, out=c)` writes into `c` with no extra allocation.

### Correctness

8. **Beware view vs copy.** When in doubt, `.copy()`. Test with `np.shares_memory(a, b)`.
9. **Use `np.isnan(x)`**, never `x == np.nan` (always False).
10. **Use `nan*` aggregations** (`nanmean`, `nansum`) when NaNs may be present.
11. **Wrap boolean conditions in parens.** `(a > 0) & (a < 5)`, not `a > 0 & a < 5`.
12. **`&`/`|`/`~` for arrays, `and`/`or`/`not` for scalars.**
13. **Watch for integer overflow** with small dtypes. `int8` wraps at 127.
14. **Use `keepdims=True`** when you'll broadcast a reduction back. Avoids the `(5,)` vs `(5, 1)` trap.
15. **Match `*` for elementwise, `@` for matmul.** Bugs from confusing these are silent and severe.

### Code style

16. **Prefer `arr.method()` over `np.func(arr)`** when both exist — slightly faster, cleaner.
17. **Use `arr[i, j]`, not `arr[i][j]`.** Single-bracket form is faster (one indexing op vs two).
18. **Avoid `dtype=object`.** It's a Python list with NumPy syntax — none of the speed.
19. **Don't use `np.vectorize` for performance.** It's a clarity wrapper; it loops in Python.
20. **Seed your randomness.** `np.random.seed(42)` (or modern `np.random.default_rng(42)`) before any random call you want to reproduce.

### EDA workflow

21. **Always check `.shape` and `.dtype` first** when you receive a new array. Half of bugs come from one being wrong.
22. **Clean strings → cast type → compute.** Never try math on `"'800.0'"`.
23. **Look at `.describe()` (in Pandas) or summary stats** before modelling. Outliers, negatives, NaNs all hide in plain numbers.
24. **Save intermediate cleaned arrays** with `np.save` — re-running cleaning every time is wasteful.
25. **Document the dtype contract.** "This column is `float64` after cleaning" — write it down in your notebook.

### Interview-day reminders

- Walk through dimensions out loud: "shape `(5, 2)` — that's 5 rows, 2 columns."
- For broadcasting questions, **right-align the shapes** on scratch paper.
- For axis questions, say "**axis 0 disappears, axis 1 disappears**" — let the dimension that vanishes guide the answer.
- For view/copy questions: **basic slice = view, fancy/boolean = copy.** Memorize this trio.
- When unsure, **show you'd test it**: `np.shares_memory(a, b)` or `arr.base`.

[🔝 Back to top](#top)

---

<a id="29-checklist"></a>
## 29. Final checklist before any data-science interview

- [ ] Can you explain the **list vs array** difference with the `* 2` example?
- [ ] Can you give **three reasons** NumPy is faster, with the tea-room analogy?
- [ ] Can you derive the shape returned by `arr[:, 0]` and `arr[:, 0:1]` cold?
- [ ] Do you know the **type priority** rule and can give a coercion example?
- [ ] Can you explain why slicing returns a view but boolean indexing returns a copy?
- [ ] Can you write a one-liner to filter rows by condition?
- [ ] Can you broadcast `(5, 2) - (2,)` and explain why it works?
- [ ] Can you compute per-row mean of a 2D array and subtract it back (hint: `keepdims=True`)?
- [ ] Can you describe an EDA pipeline for the Zomato dataset in 60 seconds?
- [ ] Can you `astype(float)` a string column with commas like `"1,200"` end-to-end?
- [ ] Can you explain the difference between `*` and `@` on 2D arrays?
- [ ] Can you handle NaNs end-to-end: detect, count, replace, aggregate?
- [ ] Can you sort a 2D array by one column keeping rows aligned?
- [ ] Can you explain strides and why transpose is free?
- [ ] Can you choose between `concatenate` and `stack` instantly?

If you can do all fifteen without notes, you're solid on the NumPy + EDA-intro module.

[🔝 Back to top](#top)

---

<a id="30-mapping"></a>
## 30. Mapping back to your notebook

| Notebook section | This guide |
|---|---|
| What is Machine Learning | §1 |
| EDA / DAV | §2 |
| Python Lists vs NumPy Arrays | §3 |
| Why is NumPy Faster? | §4 |
| Homogeneity & Priority | §7 |
| Array Dimensions and Shape | §6 |
| Type conversion (`astype`) | §8 |
| Indexing | §9 |
| Slicing (1D and 2D) | §10 |
| MCQ: `arr.astype(int)` | §8 Q22, §24 Q1 |
| MCQ: shape of `arr[:, 0]` | §10 Q30, §24 Q2 |
| *Not yet in notebook — added here* | §11 (views/copies), §12 (boolean), §13 (broadcasting), §14 (aggregations), §15 (reshape), §16 (stacking), §17 (sorting), §18 (NaN), §19 (where/clip), §20 (linalg), §21 (strides), §22 (I/O), §23 (datetime) |

The sections marked "not yet in notebook" come up *constantly* in interviews even when they're skipped in the introductory module. Don't skip them.

[🔝 Back to top](#top)
