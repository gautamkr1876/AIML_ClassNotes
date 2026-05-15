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
11. [⚠️ Views vs copies — the #1 NumPy gotcha](#11-views-copies)
12. [Boolean indexing](#12-boolean)
13. [Broadcasting](#13-broadcasting)
14. [Aggregations and axis](#14-aggregations)
15. [Reshape, flatten, ravel, transpose](#15-reshape)
16. [Common trick / multi-correct questions](#16-tricks)
17. [Rapid-fire flashcards](#17-flashcards)

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
np.eye(3)                     # 3×3 identity matrix
np.random.rand(3, 4)          # uniform random [0, 1)
np.random.randn(3, 4)         # standard normal (mean 0, std 1)
np.random.randint(0, 10, 5)   # 5 random ints in [0, 10)
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
  np.array(["1,200", "800", "550"])           # all strings
  cleaned = np.char.replace(arr, ",", "")     # strip commas
  cleaned.astype(float)                       # → [1200., 800., 550.]
  ```
  Or with Pandas: `df['cost'].str.replace(',', '').astype(float)`.

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

Common aggregations: `sum`, `mean`, `min`, `max`, `std`, `var`, `median`, `argmin`, `argmax`, `cumsum`, `cumprod`.

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

---

<a id="16-tricks"></a>
## 16. Common trick / multi-correct questions

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

---

<a id="17-flashcards"></a>
## 17. Rapid-fire flashcards

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

---

## Final checklist before any data-science interview

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

If you can do all ten without notes, you're solid on the NumPy + EDA-intro module.

---

## Mapping back to your notebook

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
| MCQ: `arr.astype(int)` | §8 Q22, §16 Q1 |
| MCQ: shape of `arr[:, 0]` | §10 Q30, §16 Q2 |
| *Not yet in notebook — added here* | §11 (views/copies), §12 (boolean), §13 (broadcasting), §14 (aggregations), §15 (reshape) |

The sections marked "not yet in notebook" come up *constantly* in interviews even when they're skipped in the introductory module. Don't skip them.
