<a id="top"></a>
# Pandas — Master Revision Guide

> **Standalone pandas revision sheet.** Extracted and expanded from the original Data Foundation guide so you can drill Series / DataFrame basics independently from NumPy. Pair it with [`Data_Foundation_Revision_Guide.md`](./Data_Foundation_Revision_Guide.md) for NumPy, and with [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) for advanced pandas (joins, groupby, apply, reshape, datetime, plotting).

**How to use:**
- **First-time learning a concept:** open the module's **📖 Guided concept walkthrough** ([M1](#1g-guided) · [M2](#2g-guided) · [M3](#3g-guided)). Each concept is introduced *what → why → how → where → related → code → gotcha* — no follow-up search needed.
- **Pre-interview revision:** topic finder → cheat sheets → Q&A and trap list.
- **Just before a coding round:** run the [§9 Revision Drill](#9-drill).
- **Quick term lookup:** [§4 Pandas terms glossary](#4-terms) — every term has a 2–4 sentence beginner-friendly definition that links back to its module walkthrough.
- **For applied / project-level pandas:** see [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md).

**External practice (use after you've drilled this guide):**
- 🎯 [`guipsamora/pandas_exercises`](https://github.com/guipsamora/pandas_exercises) — graded pandas problems, beginner → advanced.
- 🎯 [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles) — 100 short pandas puzzles.
- 🎯 **StrataScratch** — real pandas/SQL interview questions from Amazon, Meta, Google, Airbnb.
- 🎯 **LeetCode** → *Pandas* track — paid but tightly graded.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — theory + coding.

---

## 🚀 Topic finder

Jump straight to the topic you need.

| Need to revise… | Go to |
|---|---|
| 📖 First-time intro to a concept (what / why / how / where / related) | [M1 walkthrough](#1g-guided), [M2 walkthrough](#2g-guided), [M3 walkthrough](#3g-guided) |
| Series, DataFrame, indices, file I/O | [Module 1](#1-module1) → [walkthrough](#1g-guided) / [cheat sheet](#1c-cheat) |
| `pd.read_csv`, `to_csv`, `head`, `info`, `describe` | [Module 1](#1-module1) → [walkthrough](#1g-guided) |
| `.iloc` vs `.loc`, `set_index`/`reset_index` | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| Column ops, boolean filtering, `rename`, `drop`, dot-vs-bracket | [Module 2](#2-module2) → [walkthrough](#2g-guided) |
| `astype`, `pd.to_numeric(errors='coerce')`, `.str` accessor | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
| `replace`, NaN, `unique`/`nunique`/`value_counts`, `SettingWithCopyWarning` | [Module 3](#3-module3) → [walkthrough](#3g-guided) |
| All key pandas terms | [§4 Terms](#4-terms) |
| Every pandas API at once | [§5 API cheat sheet](#5-apis) |
| Common pandas gotchas | [§6 Gotchas](#6-gotchas) |
| Pandas advanced Q&A | [§7 Advanced](#7-advanced) |
| Common "business question → API" map | [§8 Business map](#8-businessmap) |
| 🌐 Sourced interview questions (drill bank) | [Sourced bank](#sourced-bank) |
| Speed-run revision before interview | [§9 Drill](#9-drill) |
| Best practices | [§10 Best practices](#10-bestpractices) |

---

## 📑 Table of contents

1. [Module 1 — Series, DataFrame, I/O](#1-module1) · [📖 Guided walkthrough](#1g-guided) · [🧠 Cheat sheet](#1c-cheat)
2. [Module 2 — Selection: `.iloc` / `.loc` / column ops](#2-module2) · [📖 Guided walkthrough](#2g-guided)
3. [Module 3 — Cleanup: dtype, strings, NaN, exploration](#3-module3) · [📖 Guided walkthrough](#3g-guided)
4. [📚 Pandas terms glossary](#4-terms)
5. [⚙️ API cheat sheet](#5-apis)
6. [⚠️ Gotchas & traps](#6-gotchas)
7. [🎯 Advanced interview Q&A](#7-advanced)
8. [Business questions → which API](#8-businessmap)
8B. [🌐 Sourced interview questions](#sourced-bank)
9. [🔁 Revision drill (50 questions)](#9-drill)
10. [✅ Best practices](#10-bestpractices)

---

<a id="1-module1"></a>
## 1. Module 1 — Series, DataFrame, I/O

> Foundation — Pandas's two core data structures and how to load and preview data.

### 🪜 Mental model

**Series is a named column; DataFrame is a spreadsheet with row labels.** A Series carries one *index axis* (row labels) and a `.name` attribute. A DataFrame is just many Series glued side-by-side, sharing one row index. *Every operation in Pandas is either "do this to one Series" or "do this to many Series in parallel."* When you're confused, slice down to a single Series and reason about that.

<a id="1g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 1 concept. Read this top-to-bottom on a first pass; the cheat sheet below is the recap surface.

#### Series — a 1D labelled array

> **🪜 Mental model:** *A column with a name and row labels.* Think of one column ripped out of a spreadsheet — it has values, an index (the row labels), and a column name.

**What it is.** A **Series** is Pandas's one-dimensional data structure: a flat array of values plus an **index** (the row labels) plus an optional `.name`. The values are stored in a NumPy ndarray under the hood, which is why Series operations are fast. You can build one from a Python list, a dict (keys become the index), or a NumPy array.

**Why it matters.** Series is the atomic unit of Pandas — *every column of every DataFrame is a Series*. Most of the surprises you'll hit ("why did this `.mean()` return NaN?", "why is this slow?") come down to a property of the underlying Series (its `dtype`, its index, its NaN handling). If you understand Series, you understand Pandas.

**How it works.**
1. You create a Series: `pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='votes')`.
2. Pandas wraps the values in a NumPy array (one `dtype` for all elements), stores the index, and remembers the name.
3. Any operation (`s + 5`, `s.mean()`, `s.str.upper()`) dispatches to NumPy or to Pandas-specific code that streams over the array.
4. Index labels travel with the values — `s.loc['a']` and `s.iloc[0]` return the same value here, but they ask different questions (by label vs by position).

**Where it's used.**
- Every column you pull off a DataFrame: `df['rating']` is a Series.
- The output of most aggregations: `df['rating'].value_counts()` returns a Series (categories → counts).
- Standalone time series before you wrap them into a DataFrame.
- scikit-learn target vectors: `y = df['target']` is a Series.

**Related terms.**
- **DataFrame** — sibling concept; a 2D table = many Series glued by a shared index. ([walkthrough](#1g-guided))
- **Index** — the row labels carried by every Series and DataFrame. ([walkthrough](#1g-guided))
- **`dtype`** — element type; one per Series (like NumPy). See [Data_Foundation_Revision_Guide.md](./Data_Foundation_Revision_Guide.md#1g-guided).
- **NumPy ndarray** — the under-the-hood storage; that's why Series ops are vectorised.
- **`.values`** / **`.to_numpy()`** — drop the Pandas wrapper and get the raw ndarray.

```python
import pandas as pd
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='votes')
s['b']           # 20
s.mean()         # 20.0
```

**Gotcha.** A Series with a non-unique index (`['a', 'a', 'b']`) breaks `loc`-based lookups in surprising ways — `s.loc['a']` returns *all* matching rows, not one value. Keep indices unique unless you know why you're not.

#### DataFrame — a 2D labelled table

> **🪜 Mental model:** *A spreadsheet with row labels and a name on every column.* Each column is a Series; all columns share the same row index.

**What it is.** A **DataFrame** is Pandas's two-dimensional table: rows × columns, where columns can have *different* `dtypes` and rows share a single index. You can build one from a dict of lists (keys become column names), a list of dicts, a list of lists with `columns=`, a NumPy 2D array, or by loading a file. Internally it's a collection of aligned Series.

**Why it matters.** DataFrame is the workhorse data structure for all of data science in Python — EDA, cleaning, feature engineering, model inputs. scikit-learn, statsmodels, plotnine, seaborn — almost everything accepts a DataFrame. If "spreadsheet in code" is the mental model, DataFrame is the implementation.

**How it works.**
1. You create one: `pd.DataFrame({'name': ['A', 'B'], 'votes': [120, 240]})`.
2. Pandas builds one Series per column (each with its own dtype) and shares one row index across them.
3. Column access (`df['votes']`) returns the underlying Series.
4. Row access via `.iloc` / `.loc` returns either a Series (one row) or a DataFrame (multiple rows).
5. Operations like `df.mean()` walk every column in parallel and return a Series of per-column results.

**Where it's used.**
- The output of `pd.read_csv` / `pd.read_excel` / `pd.read_sql`.
- The input to every scikit-learn `fit(X, y)` (`X` is typically a DataFrame).
- EDA: `df.head()`, `df.info()`, `df.describe()` are the first three commands on any new dataset.
- Visualization: `df.plot()` and seaborn's `data=df, x=, y=` API.

**Related terms.**
- **Series** — sibling; one column of a DataFrame is a Series. ([walkthrough](#1g-guided))
- **Index** — the row labels every DataFrame carries. ([walkthrough](#1g-guided))
- **`.columns`** — the column labels (a separate Index object).
- **`.dtypes`** — Series mapping column name → dtype; the cheapest schema check after `.info()`.
- **NumPy 2D array** — like a DataFrame but without column names, row labels, or per-column dtypes.

```python
df = pd.DataFrame({'name': ['A', 'B'], 'votes': [120, 240]})
df['votes']           # → Series
df.iloc[0]            # → Series (row 0)
df[['name', 'votes']] # → DataFrame
```

**Gotcha.** `df['col']` returns a **Series**; `df[['col']]` returns a **DataFrame** with one column. The single vs double brackets is a habit you must build — many APIs (plotting, sklearn) want a DataFrame, not a Series.

#### Index — the row labels

> **🪜 Mental model:** *Built-in row IDs.* Every Series and DataFrame ships with an index — by default integers `0, 1, 2, …`; once you replace it, lookups by `.loc` change meaning.

**What it is.** An **Index** is the array of row labels attached to a Series or DataFrame. By default it's a `RangeIndex` (integers `0` to `n-1`). You can replace it with strings (`'restaurant_name'`), dates (a `DatetimeIndex`), or any hashable values. The index is what makes Pandas *Pandas* — every alignment, every join, every `.loc` lookup uses it.

**Why it matters.** Misunderstanding the index causes the majority of beginner Pandas bugs: `loc[0]` vs `iloc[0]` diverging after a sort, `pd.concat` producing NaNs because indexes don't align, `groupby` results having unexpected row labels. Once you internalise *"the index is a separate thing from the data, and operations align by it,"* most of these vanish.

**How it works.**
1. Every DataFrame has an index (`df.index`) and a column index (`df.columns`).
2. When you compute `df_a + df_b`, Pandas first **aligns** them by index — matching labels are added; non-matching labels appear in the output with NaN values.
3. Operations like `set_index('name')` *replace* the row index with values from a column. `reset_index()` puts a default integer index back.
4. Indexes can be unique (the safe default) or non-unique (legal but error-prone for `.loc`).

**Where it's used.**
- Time series: `df.set_index('date')` enables `df.loc['2024-01':'2024-03']` slicing.
- Joins: `pd.merge` and `df.join` use the index by default to align rows.
- Grouped output: `df.groupby('city')['rating'].mean()` returns a Series whose index is the cities.
- Pivot tables and unstack/stack reshaping rely on indexes.

**Related terms.**
- **`RangeIndex`** — default integer index `0, 1, 2, …` — the *implicit* index.
- **Explicit index** — any index you set yourself (strings, dates, custom IDs).
- **`MultiIndex`** — a hierarchical index with multiple levels; produced by `groupby` on multiple keys.
- **`.iloc` vs `.loc`** — position vs label; the distinction only matters once the index isn't the default. ([walkthrough](#2g-guided))
- **Alignment** — Pandas's habit of matching rows by index before any operation.

```python
df = pd.DataFrame({'name': ['A', 'B'], 'votes': [120, 240]})
df.index                    # RangeIndex(start=0, stop=2, step=1)
df2 = df.set_index('name')  # index is now ['A', 'B']
df2.loc['A']                # row labelled 'A'
```

**Gotcha.** After `set_index('name')`, `df.iloc[0]` is still "the first row," but `df.loc[0]` will raise `KeyError` — there's no row labelled `0` any more.

#### `pd.read_csv` and file I/O

> **🪜 Mental model:** *The front door of every project.* `read_csv` is the most-used Pandas function; mastering its parameters saves hours of post-load cleanup.

**What it is.** `pd.read_csv(path, ...)` parses a CSV file (or URL) into a DataFrame. It auto-detects the delimiter, infers each column's dtype from a sample of rows, and treats `NA`, `NaN`, `null`, etc. as missing values by default. Sibling readers exist for Excel (`pd.read_excel`), JSON (`pd.read_json`), Parquet (`pd.read_parquet`), and SQL (`pd.read_sql`). The reverse operation is `df.to_csv(path, index=False)`.

**Why it matters.** 90% of "the data is messy" headaches can be fixed *at load time* by passing the right `read_csv` arguments — `na_values`, `thousands`, `dtype`, `parse_dates`, `dtype={'id': str}`. Skipping these means doing the same cleanup downstream with `.str.replace`, `.astype`, and `pd.to_numeric`. Interviewers love asking "what flags would you pass to load this CSV correctly?" because it tests whether you've actually shipped data work.

**How it works.**
1. Opens the file and reads a chunk of rows.
2. Sniffs the delimiter (unless you pass `sep=`) and uses the first row as headers (unless you pass `header=None`).
3. Scans each column to infer a `dtype`: all-int → `int64`; any float → `float64`; any unparseable string → `object`.
4. Converts cells matching `na_values` (default: `''`, `'NA'`, `'NaN'`, etc.) into `NaN`.
5. Applies any user-supplied `dtype=`, `parse_dates=`, `thousands=`, `decimal=` overrides.
6. Returns the assembled DataFrame.

**Where it's used.**
- The very first cell of almost every Jupyter notebook in this repo.
- Loading FAANG-style take-home datasets (sales logs, click streams, sensor data).
- ETL pipelines reading from S3 / GCS / local disk.
- Reading SQL exports that need light cleanup before joining.

**Related terms.**
- **`df.to_csv(path, index=False)`** — the reverse direction; `index=False` skips writing the row labels as a column.
- **`pd.read_excel`** / **`pd.read_parquet`** / **`pd.read_sql`** — sibling readers for other formats.
- **`na_values=[...]`** — list of strings that should be treated as missing.
- **`dtype={col: type}`** — pin a column's dtype at load time (the #1 way to preserve leading-zero IDs).
- **`parse_dates=['col']`** — convert a string column to `datetime64` at load time.

```python
df = pd.read_csv(
    'data.csv',
    na_values=['N', 'NEW', '-'],
    thousands=',',
    parse_dates=['order_ts'],
    dtype={'id': str},
)
```

**Gotcha.** `pd.read_csv` infers dtypes from a *sample* of the column; one stray `'NEW'` near the top and an otherwise-numeric column becomes `object`. Either pre-clean, pass `na_values=['NEW']`, or override with `dtype=`.

#### `head` / `tail` / `sample` — peeking at data

> **🪜 Mental model:** *Three windows on the dataset.* `head` shows the top, `tail` shows the bottom, `sample` shows a random slice.

**What it is.** Three sister methods for previewing a DataFrame or Series:
- **`df.head(n=5)`** — first `n` rows.
- **`df.tail(n=5)`** — last `n` rows.
- **`df.sample(n=5)`** — `n` randomly chosen rows (or `frac=0.01` for a percentage). Pass `random_state=42` for reproducibility.

**Why it matters.** Every EDA starts here. `head` shows you the schema and a few real values; `tail` catches "did the file end cleanly or with a trailing junk row?"; `sample` is the antidote to assuming the first 5 rows are representative — a sorted dataset can mislead you. In an interview, "what's the first thing you do with a new dataset?" should always include `head` + `info`.

**How it works.** Each is a cheap slice operation. `head(n)` is equivalent to `df.iloc[:n]`; `tail(n)` is `df.iloc[-n:]`. `sample` uses NumPy's RNG to pick row positions and then materialises those rows.

**Where it's used.**
- The very first commands in a new Jupyter notebook after `read_csv`.
- Sanity-checking after a transformation: `df.head()` to see "did the new column appear?"
- Building reproducible mini-datasets for unit tests: `df.sample(100, random_state=0)`.

**Related terms.**
- **`df.iloc[:n]`** — exactly what `head(n)` does under the hood.
- **`df.shape`** — first thing to check alongside `head`; tells you size before you peek.
- **`df.describe()`** — numeric summary; pairs with `head` to know "what's in this column."

```python
df.head()              # first 5 rows
df.tail(3)             # last 3 rows
df.sample(5, random_state=42)  # 5 reproducible random rows
```

**Gotcha.** `head()` on a sorted DataFrame shows only the small-value end — easy to misread the data's true distribution. Reach for `sample` to break sorting bias.

#### `.info()` — schema + null overview

> **🪜 Mental model:** *X-ray of a DataFrame.* Shows every column's dtype, how many non-null values it has, and the total memory footprint, in one printout.

**What it is.** `df.info()` prints a textual summary of the DataFrame: number of rows, list of columns with their dtype and non-null counts, and total memory usage. It does *not* return a value — it prints. Pair with `df.shape` (just the size) and `df.describe()` (numeric summary) for a complete first-pass picture.

**Why it matters.** This is the single highest-information command in EDA. In one call you see (a) which columns have missing data and how much, (b) which columns have the wrong dtype (`object` when you expected `float`), and (c) whether the dataset fits in memory. Beginners run `.head()` and stop; pros always run `.info()` next.

**How it works.** Pandas iterates over each column, counts non-null values (`Series.count()`), reads off the dtype, and sums up `arr.nbytes` per column for the memory line. The output goes to stdout.

**Where it's used.**
- Every EDA cell, right after `read_csv` and `head`.
- Debugging "why did `.mean()` return NaN?" — `.info()` shows the column is still `object` (string).
- Memory triage on big datasets — the memory line tells you when to downcast (`int64` → `int32`) or use chunking.

**Related terms.**
- **`df.shape`** — just `(rows, cols)`; the cheaper size check.
- **`df.dtypes`** — Series of column → dtype, programmatic equivalent.
- **`df.describe()`** — numeric summary (count, mean, std, quartiles).
- **`df.isnull().sum()`** — Series of per-column NaN counts; complementary view.

```python
df.info()
# RangeIndex: 51717 entries, 0 to 51716
# Data columns (total 17 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   name            51717 non-null  object
#  1   rating          43942 non-null  object  ← object, not float; investigate
# ...
```

**Gotcha.** `.info()` returns `None` — never assign it. `s = df.info()` makes `s` `None`. Just call it as a statement.

#### `.describe()` — numeric summary

> **🪜 Mental model:** *Five-number summary times every numeric column.* Count, mean, std, min, 25/50/75% percentiles, max — for every numeric column at once.

**What it is.** `df.describe()` returns a DataFrame of summary statistics — by default only for **numeric** columns. The default rows are `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`. Pass `include='all'` to include object columns (you get `unique`, `top`, `freq` for them instead) or `include=['object']` for *only* the categorical view.

**Why it matters.** It tells you the **range and skew** of every numeric column in one call. Min < 0 on a "price" column? Bug or refund. Max wildly higher than the 75th percentile? Outlier. Std huge relative to mean? Spread is wild — consider log-scaling. Interviewers love asking "what would you check first" on a numeric dataset; the answer always includes `.describe()`.

**How it works.** For each numeric column, Pandas calls `count`, `mean`, `std`, `min`, `quantile([.25, .5, .75])`, `max` — each implemented as a NumPy reduction on the underlying ndarray. Object columns produce different stats (`unique`, `top`, `freq`) when included.

**Where it's used.**
- The third command of any EDA, after `shape` and `info`.
- Spotting outliers (look at min/max vs 75th percentile).
- Comparing train vs test distributions: `train.describe()` next to `test.describe()`.
- Quick sanity checks after cleanup: did my replacement of `'NEW'` with NaN drop the count from 51,717 to 43,942?

**Related terms.**
- **`include='all'`** — show both numeric and object summaries.
- **`include=['object']`** — show *only* the categorical / string summary.
- **`df.quantile(q)`** — individual percentile (e.g., `df['x'].quantile(0.99)` for 99th).
- **`df['col'].value_counts()`** — what `describe` does poorly for categoricals; reach for `value_counts` instead.

```python
df.describe()                    # numeric only
df.describe(include='all')       # numeric + object stats
df.describe(include=['object'])  # only string/categorical
```

**Gotcha.** `.describe()` *silently skips* object columns by default. If you expected to see your `'rating'` column and it's missing, it's still `object` dtype — fix dtypes first, *then* re-describe.

#### `.shape`, `.columns`, `.dtypes` — the schema trio

> **🪜 Mental model:** *Three free attributes every DataFrame exposes.* They're O(1) lookups — call them as often as you want.

**What it is.** Three attributes (no parentheses):
- **`df.shape`** — tuple `(n_rows, n_cols)`.
- **`df.columns`** — an Index of column names.
- **`df.dtypes`** — a Series mapping each column name to its `dtype`.

**Why it matters.** These are the "I'm doing EDA correctly" reflexes. Before every transformation, check `shape` to know how big you're working with. After every transformation, check `shape` again to confirm "I didn't accidentally delete 90% of rows." `dtypes` is the cheapest dtype audit; `columns` is what you compare against your expected schema.

**How it works.** All three read pre-stored metadata — no scan of the data is needed. They're as cheap as a Python attribute access.

**Where it's used.**
- `assert df.shape[0] > 0` — guard against empty results.
- `assert 'rating' in df.columns` — guard against typo-renames.
- `df.dtypes.value_counts()` — quick summary of "how many object vs numeric vs datetime columns do I have?"
- Logging in production pipelines: log `df.shape` at every step.

**Related terms.**
- **`df.index`** — the row labels; the fourth attribute in this family.
- **`df.size`** — total cell count = `n_rows × n_cols`.
- **`df.info()`** — the verbose printout that includes all of these plus null counts.

```python
df.shape       # (51717, 17)
df.columns     # Index(['name', 'rating', ...])
df.dtypes      # Series: 'name': object, 'rating': object, ...
```

**Gotcha.** These are **attributes**, not methods — no parentheses. `df.shape()` raises `TypeError: 'tuple' object is not callable`.

<a id="1c-cheat"></a>
### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#1g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Series** | Pandas's 1D labelled array — values + index + name. Built on a NumPy ndarray, so it's fast. | Every time you pull one column off a DataFrame (`df['col']`) or build a standalone vector. |
| **DataFrame** | 2D labelled table — many Series glued by a shared row index. Columns can have different dtypes. | The default container for tabular data; output of `read_csv`, input to sklearn. |
| **Index** | The row labels every Series/DataFrame carries — default `0,1,2,…` or any custom values. | Time series slicing, joins, `groupby` results, label-based `.loc` lookups. |
| **`RangeIndex` (implicit)** | The default `0, 1, 2, …` index Pandas attaches when you don't set one. | Fresh CSV loads, unsorted data — the safe baseline. |
| **Explicit index** | An index you set yourself with `set_index('col')` or pass in at construction. | When labels (names, dates, IDs) are more useful than positions for lookup. |
| **`pd.read_csv`** | The front door — load a CSV (or URL) → DataFrame. Auto-infers dtypes from a sample. | Every project. Pass `na_values`, `dtype`, `parse_dates`, `thousands` to clean at load time. |
| **`df.to_csv(path, index=False)`** | Save back to CSV. `index=False` skips writing the row index as a column. | Persisting cleaned data between notebook runs. |
| **`pd.concat([...], axis=0/1)`** | Glue DataFrames / Series along rows (`0`) or columns (`1`). Aligns by index — mismatches become NaN. | Stacking multiple files into one table; gluing new columns onto a DataFrame. |
| **`.head(n)` / `.tail(n)`** | First / last `n` rows. Default `n=5`. Equivalent to `iloc[:n]` / `iloc[-n:]`. | Every EDA cell — the first peek at the data. |
| **`.sample(n)`** | `n` *random* rows; pass `random_state=` for reproducibility. | When `head` would mislead because the data is sorted. |
| **`.info()`** | Schema + dtype + non-null count + memory footprint, printed (returns `None`). | After every `read_csv` — catches missing data and wrong-dtype columns. |
| **`.describe()`** | Per-numeric-column summary (count, mean, std, min, quartiles, max). | Range/skew/outlier check; pair with `.info()` for full first-pass. |
| **`.shape`** | Tuple `(rows, cols)`. O(1) — no parens. | Before/after every transform, to confirm row count didn't blow up. |
| **`.columns`** | Index of column names. | Schema checks; `'col' in df.columns` guards. |
| **`.dtypes`** | Series of column → dtype. The cheapest dtype audit. | Verifying cleanup worked; spotting `object` columns that should be numeric. |

### 🪞 Basic → Intermediate → Advanced — `pd.read_csv`

**Basic** — load a CSV with default settings.
```python
df = pd.read_csv('data.csv')
```

**Intermediate** — clean at load time instead of after.
```python
df = pd.read_csv(
    'data.csv',
    na_values=['N', 'NEW', '-'],     # custom sentinels → NaN
    thousands=',',                   # parse "1,200" as 1200
    parse_dates=['order_ts'],        # parse to datetime on load
)
```

**Advanced** — preserve identifier columns (e.g. `"00123"`) by forcing a dtype, otherwise leading zeros are silently dropped.
```python
df = pd.read_csv('data.csv', dtype={'id': str, 'phone': str})
df.dtypes['id']                      # → object (string) — zeros preserved
```

### ⚙️ Top APIs

```python
pd.Series(arr, index=[], name='')
pd.DataFrame({...}) / pd.DataFrame([[...]])
pd.concat([s1, s2], axis=1)
pd.read_csv('path', sep=',', header=0, na_values=[...], dtype={...}, thousands=',')

df.head(n), df.tail(n), df.sample(n, random_state=42)
df.info(), df.describe(include='all')
df.shape, df.dtypes, df.columns, df.index
df.to_csv('out.csv', index=False)
```

### 🧩 Code patterns

```python
# 1. Load → preview → schema
df = pd.read_csv('data.csv')
df.head(); df.info(); df.describe()

# 2. Build a small DataFrame from a dict (columns as keys)
df = pd.DataFrame({
    'restaurant': ['R1', 'R2'],
    'votes':      [120, 240],
    'rating':     [4.1, 4.5],
})

# 3. Glue two Series side-by-side
pd.concat([s1, s2], axis=1)

# 4. Append rows from a second table
combined = pd.concat([df_a, df_b], ignore_index=True)

# 5. Save a cleaned table
df.to_csv('clean.csv', index=False)
```

### 🎯 Q&A — Module 1

1. **What's the difference between a Series and a DataFrame?**
   A Series is 1D with one index axis. A DataFrame is 2D with a row index *and* column labels. Each column of a DataFrame is a Series; together they share the same row index.

2. **What's `pd.concat([s1, s2], axis=1)` with mismatched indexes?**
   Union join — aligned by index, missing positions filled with `NaN`. Use `join='inner'` for intersection.

3. **How does `pd.read_csv` decide dtypes?**
   It scans the column; if all parseable as int, uses int; if any float, uses float; otherwise `object`. Override with `dtype={'cost': 'float64'}`. Use `na_values=['N', 'NEW', '-']` to convert sentinels to NaN at load time.

4. **Why does `.describe()` sometimes show fewer columns than expected?**
   By default it only profiles **numeric** columns. Strings/objects are excluded unless you pass `include='all'`.

5. **What's the cheapest first-pass on any new dataset?**
   `df.shape`, `df.head()`, `df.info()` — size, sample, schema. Tells you 80% of what you need before diving in.

6. **`df['col']` vs `df[['col']]` — what's the difference?** *(adapted from `guipsamora/pandas_exercises`)*
   Single brackets return a **Series** (1D, has `.name`). Double brackets return a **DataFrame** with one column (2D, has `.columns`). Plotting and sklearn often need a DataFrame, not a Series.

7. **`df.sample(5)` vs `df.head(5)` — when to use each?** *(common FAANG EDA question)*
   `head` is the first 5 rows in physical order — biased if the data is sorted. `sample` is random — pass `random_state` for reproducibility. Use `sample` whenever you suspect sorting or block-ordering.

[🔝 Back to top](#top)

---

<a id="2-module2"></a>
## 2. Module 2 — Selection: `.iloc` / `.loc` / column ops

> How to grab the rows and columns you actually want.

### 🪜 Mental model

**`iloc` is the row *number*; `loc` is the row *name*.** After a fresh `read_csv`, both refer to the same row because the default index is `0, 1, 2, …`. The instant you `set_index`, sort, or filter, the **label** and the **position** diverge — and so do `iloc[0]` and `loc[0]`. *Position is "show me the 3rd row"; label is "show me the row called X."* Pick whichever question you're actually asking.

<a id="2g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 2 concept. The cheat sheet below is the recap surface.

#### `.iloc` — position-based selection

> **🪜 Mental model:** *"Give me the 3rd row."* `iloc` always asks by **integer position**, regardless of what the index labels look like.

**What it is.** `df.iloc[i]` selects row `i` by position. `df.iloc[i, j]` selects the cell at row `i`, column `j`. `df.iloc[i:j]` is positional slicing — same rules as NumPy: **end exclusive**, supports `:`, supports negative indices. The `i` is *always* an integer (or a slice/list of integers), never a label.

**Why it matters.** When the row labels are meaningless (default `0, 1, 2, …` index) or when you want "the first N rows" regardless of label, `iloc` is the unambiguous tool. It's also the only safe way to write code that survives an index reset — `df.iloc[0]` works no matter what the index is.

**How it works.**
1. You pass an integer (or slice / list of integers).
2. Pandas looks at the *position* in the row order, completely ignoring the index labels.
3. Single int → returns a Series (one row). Slice / list of ints → returns a DataFrame (multiple rows).
4. Two-axis form: `df.iloc[1, 2]` returns a single cell; `df.iloc[:, 0]` returns the first column as a Series.

**Where it's used.**
- "Top N rows" idioms: `df.iloc[:5]` (same as `df.head(5)`).
- Train/test split by position: `df.iloc[:80]`, `df.iloc[80:]`.
- Iterating over a DataFrame by row position (rare — prefer `apply` or vectorisation).
- Inside library code that needs to be index-agnostic.

**Related terms.**
- **`.loc`** — sibling; label-based. Diverges from `iloc` once you `set_index`/sort/filter. ([walkthrough](#2g-guided))
- **`df.iloc[mask]`** — combining boolean masks with positional access (rare but legal).
- **`df.iat[i, j]`** — faster scalar accessor; like `iloc` but for one cell only.
- **Basic slicing** — `df.iloc[1:5]` follows the same end-exclusive rule as NumPy slicing.

```python
df.iloc[0]            # first row (Series)
df.iloc[-1]           # last row
df.iloc[0:5, 0:3]     # first 5 rows × first 3 columns (DataFrame)
df.iloc[[0, 2, 4]]    # specific rows by position
```

**Gotcha.** `df.iloc[5]` and `df.loc[5]` look identical until you've sorted or set the index. The instant the index isn't `0, 1, 2, …` in order, they refer to *different* rows.

#### `.loc` — label-based selection

> **🪜 Mental model:** *"Give me the row called Truffles."* `loc` asks by **index label**, not position.

**What it is.** `df.loc[label]` selects row(s) by index label. `df.loc[label, 'col']` selects a single cell by row label and column name. `df.loc[r1:r2]` is **inclusive on both ends** — unlike `iloc` and unlike NumPy, `loc` slices include the stop label. Also accepts boolean masks (`df.loc[mask]`) and the killer one-step assignment form: `df.loc[mask, 'col'] = value`.

**Why it matters.** Once you have meaningful row labels (a date index, a restaurant name index, a user ID index), `loc` lets you query the way humans think: "row for `'Truffles'`" instead of "row number 217." `loc` is also the *safe* form for assignment — it's the antidote to `SettingWithCopyWarning`.

**How it works.**
1. You pass a label (or a list/slice/mask of labels).
2. Pandas looks up the label(s) in the index — `KeyError` if not found.
3. For slices, both endpoints are **included** in the result.
4. Combined with a column argument, `loc` does a label-based 2D pick.
5. On assignment, `df.loc[mask, 'col'] = val` writes back to the original DataFrame in one atomic step — no copy ambiguity.

**Where it's used.**
- Time-series slicing: `df.loc['2024-01':'2024-03']` (after `set_index('date')`).
- Lookups by ID: `df.loc[user_id]`.
- Safe conditional assignment: `df.loc[df['rating'] < 3, 'flag'] = -1`.
- Multi-column reads with row filter: `df.loc[mask, ['name', 'rating']]`.

**Related terms.**
- **`.iloc`** — sibling; position-based. The famous lookalike. ([walkthrough](#2g-guided))
- **`.at[label, 'col']`** — faster scalar version; like `loc` but for one cell only.
- **Boolean mask** — `loc` accepts a `Series[bool]` the same length as the DataFrame.
- **`SettingWithCopyWarning`** — what you avoid by using single-step `.loc` assignment. ([walkthrough](#3g-guided))

```python
df = df.set_index('name')
df.loc['Truffles']               # row labelled 'Truffles' (Series)
df.loc['Truffles', 'rating']     # single cell
df.loc[df['rating'] < 3, 'flag'] = -1   # safe conditional write
df.loc['A':'C']                  # inclusive — A, B, C all included
```

**Gotcha.** `df.loc[1:5]` includes row labelled `5`; `df.iloc[1:5]` stops at position 5 (does not include it). Inclusivity is the most-missed difference between the two.

#### `.iloc` vs `.loc` — THE biggest pandas trap

> **🪜 Mental model:** *Position is the row's seat number; label is the row's name.* They agree on a default `0,1,2,…` index — they diverge the moment that index changes.

**What it is.** Two indexers that look identical for the first 30 seconds and behave very differently once the index isn't the default. `iloc` always selects by **integer position** in the row order (`iloc[0]` is the first row, period). `loc` selects by **index label** (`loc[0]` is the row whose label is `0` — which may not exist). They also differ in slice semantics: `iloc` slices are **end-exclusive** like NumPy; `loc` slices are **inclusive on both ends**.

**Why it matters.** This is the single most common pandas bug. After a `set_index('user_id')`, junior devs reach for `df.loc[0]` expecting "the first row" and get a `KeyError`. After a `sort_values`, `df.loc[0]` returns "the row that was originally first" — which is buried somewhere in the new order. Confusing the two is the #1 cause of "the test passed locally but failed on the real data" stories in interviews.

**How it works.** Think of three states:
1. **Fresh DataFrame, default index.** `iloc[0]` == `loc[0]`. Both pick the first row.
2. **After `sort_values` or filter.** Position changes, label stays. `iloc[0]` is the new top row; `loc[0]` is the same row it was before (just possibly not on top any more).
3. **After `set_index('col')`.** The index is now strings/dates/IDs. `loc[0]` raises `KeyError` (no row labelled `0`). `iloc[0]` still works — first row.

A two-cell mental table:

| | After `read_csv` (default index) | After `set_index('name')` |
|---|---|---|
| `df.iloc[0]` | First row | First row |
| `df.loc[0]` | First row | `KeyError` (no label `0`) |
| `df.loc['Truffles']` | `KeyError` (no label `'Truffles'`) | The row called Truffles |

**Where it's used.**
- Choosing positions (`iloc`): "first N rows," "every other row," index-agnostic code.
- Choosing labels (`loc`): time-series slices, ID lookups, conditional assignment, joins.

**Related terms.**
- **`.iat` / `.at`** — fast scalar accessors; same position/label distinction but for single cells.
- **`reset_index(drop=True)`** — the way to make `iloc` and `loc` agree again. ([walkthrough](#2g-guided))
- **`SettingWithCopyWarning`** — `.loc[mask, 'col'] = v` is the safe-assignment form. ([walkthrough](#3g-guided))
- **Slice semantics** — `iloc` end-exclusive (like NumPy/Python); `loc` end-inclusive (the only Pandas exception).

```python
df.iloc[0]                  # always the first row, regardless of index
df.loc[0]                   # row labelled 0 (KeyError if no such label)
df.iloc[0:5]                # 5 rows (end exclusive)
df.loc['A':'C']             # rows A, B, C (end inclusive)
df.loc[df['x'] > 0, 'y'] = 1  # safe conditional assignment
```

**Gotcha.** Once you've done *anything* that re-labels the index (`set_index`, drop duplicates, filter, sort, groupby), `iloc[0]` and `loc[0]` no longer refer to the same row. Pick the one that matches the question you're actually asking.

#### Column selection — `df['col']` vs `df[['col']]`

> **🪜 Mental model:** *Single brackets unwrap; double brackets keep the box.* `df['col']` gives you the bare column (Series); `df[['col']]` gives you a one-column DataFrame.

**What it is.** Two related column-selection forms:
- **`df['col']`** — single brackets, single string → returns a **Series** (1D, has `.name`, no `.columns`).
- **`df[['c1', 'c2']]`** — brackets around a **list** of names → returns a **DataFrame** (2D, has `.columns`).
- **`df[['c1']]`** — list with one item → still a DataFrame (with one column).
- **`df.col`** — attribute access; same as `df['col']` *only when* the name is a clean Python identifier (no spaces, no dots, no method collisions).

**Why it matters.** Many libraries demand a DataFrame, not a Series — plotting libraries (seaborn's `data=` argument), sklearn's `fit(X, y)` (`X` is 2D), and any code that calls `.merge()`. Mixing up Series vs DataFrame produces silent shape bugs that propagate downstream. And `df.col` is a footgun: it silently fails on names like `'cost for two (₹)'` or shadows methods like `df.shape`.

**How it works.**
1. `df['col']` indexes into the column index, returns the underlying Series.
2. `df[[...]]` triggers list-based fancy indexing — Pandas builds a new DataFrame containing copies of those Series in the requested order.
3. `df.col` is Python attribute access — works only if `'col'` is a valid identifier and not a built-in DataFrame attribute.

**Where it's used.**
- Pulling a target vector: `y = df['target']` (Series, perfect for sklearn).
- Pulling a feature matrix: `X = df[['age', 'income']]` (DataFrame, perfect for sklearn).
- Reordering or subsetting columns: `df = df[['name', 'rating', 'votes']]`.
- Quick column reads in clean column names: `df.rating` (works for `'rating'`, breaks for `'cost for two'`).

**Related terms.**
- **`.loc[:, ['c1', 'c2']]`** — verbose equivalent of `df[['c1', 'c2']]`; preferred inside `.loc` chains.
- **`df.iloc[:, [0, 2]]`** — column-by-position version.
- **`df.filter(items=[...])`** — explicit selection by exact name (also supports regex with `regex=`).
- **`df.drop(columns=[...])`** — the inverse operation. ([walkthrough](#2g-guided))

```python
df['rating']             # Series
df[['rating']]           # DataFrame with 1 column
df[['name', 'rating']]   # DataFrame with 2 columns
df.rating                # Same as df['rating'] — only if name is clean
```

**Gotcha.** `df.shape` is the DataFrame's shape attribute, not a column named `shape`. If your column is named `'shape'`, `df.shape` returns the tuple, not the column. **Bracket form is the safe default.**

#### Boolean filtering — `df[mask]`

> **🪜 Mental model:** *A True/False stencil over the rows.* Build a boolean Series the same length as the DataFrame; pass it in brackets; only `True` rows survive.

**What it is.** Pandas boolean filtering uses the same idea as NumPy boolean masking: a `Series[bool]` of length `len(df)`, passed in brackets, returns only the rows where the mask is `True`. Multi-condition masks combine with bitwise operators (`&`, `|`, `~`) — **parentheses are mandatory** around each comparison because of operator precedence.

**Why it matters.** This is the single most-used Pandas operation. "Show me restaurants above 4 stars in Bangalore" is `df[(df['rating'] > 4) & (df['city'] == 'Bangalore')]`. It's vectorised (no Python loop), readable, and chainable. Interviewers expect this on sight — anyone using `df.iterrows()` for filtering has flunked.

**How it works.**
1. The comparison (`df['rating'] > 4`) produces a Series of booleans, one per row.
2. `df[mask]` walks the mask and keeps only the rows where the mask is `True`. Output is a new DataFrame (a copy).
3. Multi-condition masks combine boolean Series with bitwise `&` / `|` / `~`. Each comparison must be wrapped in parens — `&` binds tighter than `>`, so `df['a'] > 0 & df['b'] < 5` parses wrong.

**Where it's used.**
- Subsetting any DataFrame by a condition.
- Conditional assignment via `.loc`: `df.loc[df['x'] < 0, 'flag'] = -1`.
- Building a "valid rows" filter pipeline: `df[df['rating'].notna() & (df['votes'] > 50)]`.
- Train-test masks: `train = df[df['date'] < cutoff]`, `test = df[df['date'] >= cutoff]`.

**Related terms.**
- **`.query('col > 5')`** — string-form filter; readable for complex multi-condition expressions.
- **`.isin([...])`** — mask of "is the value in this list?": `df[df['city'].isin(['Bangalore', 'Chennai'])]`.
- **`.between(lo, hi)`** — inclusive-range mask: `df[df['rating'].between(3, 5)]`.
- **`.loc[mask, 'col'] = v`** — the safe-assignment form. ([walkthrough](#2g-guided))
- **NumPy boolean masking** — same idea, same parens-and-bitwise rule. See [Data_Foundation_Revision_Guide.md](./Data_Foundation_Revision_Guide.md#2g-guided).

```python
df[df['rating'] > 4]
df[(df['rating'] > 4) & (df['city'] == 'Bangalore')]   # parens required
df[df['city'].isin(['Bangalore', 'Chennai'])]
df.query('rating > 4 and city == "Bangalore"')         # equivalent, string form
```

**Gotcha.** `df[df['a'] > 0 and df['b'] < 5]` raises `ValueError: The truth value of an array is ambiguous`. Use bitwise `&` *and* wrap each comparison in parens: `df[(df['a'] > 0) & (df['b'] < 5)]`.

#### `set_index` / `reset_index`

> **🪜 Mental model:** *Swap the row-label column.* `set_index('name')` promotes a column to be the new index; `reset_index()` demotes the index back to a column.

**What it is.** Two methods that swap data between the index and a column:
- **`df.set_index('col')`** — moves the values of `'col'` into the index. The column disappears (unless you pass `drop=False`). Returns a new DataFrame.
- **`df.reset_index()`** — puts the current index back as a regular column and replaces it with a default integer index. Pass `drop=True` to discard the old index entirely.

**Why it matters.** Pandas's most powerful features — `.loc` label lookup, time-series slicing, joining by key — all depend on having a meaningful index. `set_index` is how you switch from "table with a name column" to "table indexed by name." `reset_index` is what you call after every `groupby` to flatten the result back into a regular DataFrame.

**How it works.**
1. `set_index('col')` builds a new index from the column's values, removes that column from the data, and returns the new DataFrame.
2. `reset_index()` builds a default `RangeIndex(0, len(df))`, optionally inserts the old index as the leftmost column (`drop=False`, the default), and returns the new DataFrame.
3. `inplace=True` modifies the original DataFrame and returns `None` — modern Pandas prefers reassignment.
4. After `groupby`, the group keys become the new index — call `reset_index()` to get them back as columns.

**Where it's used.**
- Setting a date column as the index for time-series ops: `df.set_index('date')`.
- Joining tables on a key column: set the key as the index on both before `df_a.join(df_b)`.
- Cleaning up post-`groupby` output: `df.groupby('city')['rating'].mean().reset_index()`.
- Reverting after experimentation: `df.reset_index(drop=True)` to start fresh.

**Related terms.**
- **`.loc`** — depends on the index; setting a meaningful index unlocks meaningful `.loc` lookups. ([walkthrough](#2g-guided))
- **`MultiIndex`** — `set_index(['city', 'cuisine'])` creates a hierarchical index.
- **`inplace=True`** — modifies in place, returns `None`. Avoid in modern Pandas; prefer reassignment.
- **`drop=True`** — on `reset_index`, discards the old index instead of saving it as a column.

```python
by_name = df.set_index('name')           # 'name' is now the row label
by_name.loc['Truffles']                  # label lookup works

df_flat = by_name.reset_index()          # 'name' is a column again
df_flat = by_name.reset_index(drop=True) # discard the old index entirely
```

**Gotcha.** `reset_index()` without `drop=True` keeps the old index as a new column called `'index'` (or its previous name). If you didn't want that column, you've now polluted your DataFrame — always think about `drop=True`.

#### `rename` — change column or index names

> **🪜 Mental model:** *Relabel without rebuilding.* `rename` swaps out names in the column list or index; the data doesn't move.

**What it is.** `df.rename(columns={'old': 'new'})` renames columns; `df.rename(index={'old': 'new'})` renames index labels. You pass a dict (or a function) mapping old → new names. Keys that don't match an existing name are **silently ignored** — no error. Returns a new DataFrame; pass `inplace=True` to modify in place (and return `None`).

**Why it matters.** Real-world column names are messy: `'approx_cost(for two people)'`, `'listed_in(type)'`, leading whitespace, mixed case. Standardising them early (`cost_for_two`, `listing_type`) saves repeated quoting and prevents `df.col` from breaking. It's also a common interview ask: "the raw column names are bad; rename them."

**How it works.**
1. Pandas walks the column (or index) labels.
2. For each label, looks it up in your mapping dict. If found → use the new name; if not → keep the old name.
3. Builds a new column index (or row index) with the updated names; returns a new DataFrame sharing the underlying data buffers.
4. Missing keys in your dict are *not* flagged — silent no-op.

**Where it's used.**
- First step after `pd.read_csv` to fix bad column names: `df = df.rename(columns={...})`.
- Aligning two DataFrames before a join (rename so the join keys have matching names).
- After `groupby` to give a sensible name to the aggregated column: `g = df.groupby('city').size().reset_index(name='count')`.
- Lowercasing all columns at once: `df.rename(columns=str.lower)`.

**Related terms.**
- **`df.columns = [...]`** — bulk replace; faster but requires the full new list in order.
- **`.add_prefix('p_')`** / **`.add_suffix('_v2')`** — bulk add to every column name.
- **`drop`** — sibling cleanup operation. ([walkthrough](#2g-guided))
- **`set_axis(new_names, axis=1)`** — alternate bulk-rename API.

```python
df = df.rename(columns={
    'approx_cost(for two people)': 'cost_for_two',
    'listed_in(type)': 'listing_type',
})
df.rename(columns=str.lower, inplace=True)   # lowercase all columns
df.rename(columns={'nonexistent': 'x'})      # silently no-ops
```

**Gotcha.** A typo in the old name (`'aprox_cost'` instead of `'approx_cost'`) makes `rename` silently do nothing. Always verify with `df.columns` after.

#### `drop` — remove rows or columns

> **🪜 Mental model:** *Cut a row or column out by label.* `drop('col', axis=1)` removes a column; `drop(0, axis=0)` removes a row.

**What it is.** `df.drop(labels, axis=0/1)` removes the rows (`axis=0`) or columns (`axis=1`) at the given labels and returns a new DataFrame. Pass a list to drop multiple at once. The clearer modern form is `df.drop(columns=[...])` or `df.drop(index=[...])` — explicit, no `axis` confusion.

**Why it matters.** Cleanup is half of Pandas work — you constantly need to drop columns you won't model on (`'id'`, `'name'`), drop rows that failed a sanity check, or drop duplicates. Forgetting that `drop` returns a new DataFrame (default) is a classic beginner trap — `df.drop('col', axis=1)` alone does nothing; you must reassign or pass `inplace=True`.

**How it works.**
1. Pandas looks up each label in the requested axis.
2. Builds a new column index (or row index) with those labels removed.
3. Returns a new DataFrame referencing the surviving columns/rows.
4. Missing labels raise `KeyError` by default; pass `errors='ignore'` to silence.

**Where it's used.**
- Removing identifier columns before modelling: `X = df.drop(columns=['id', 'name'])`.
- Dropping rows with bad data: `df = df.drop(index=bad_idx)`.
- After feature engineering: drop the source columns once the derived ones are in.
- Cleaning up after `merge`: drop the duplicated key columns Pandas added.

**Related terms.**
- **`df.dropna()`** — drops rows (or columns, with `axis=1`) that contain NaN. ([walkthrough](#3g-guided))
- **`df.drop_duplicates()`** — drops repeated rows.
- **`df.pop('col')`** — removes the column from the DataFrame *and* returns it as a Series (in one step).
- **`del df['col']`** — Python-style in-place delete; less Pandas-idiomatic but works.

```python
df = df.drop(columns=['id', 'phone'])
df = df.drop(index=[0, 1, 2])
df.drop(columns='id', inplace=True)
df.drop(columns='missing', errors='ignore')   # don't crash if not present
```

**Gotcha.** `df.drop('col')` alone — without `columns=`, `axis=1`, or reassignment — does nothing visible because (a) without `axis=1` it tries to drop a *row* labelled `'col'`, and (b) without reassignment, the dropped DataFrame is thrown away.

#### Adding / modifying columns

> **🪜 Mental model:** *Just assign to a column name.* Existing name → modify it. New name → create it.

**What it is.** Pandas lets you create or update a column by simple assignment: `df['new'] = expr` adds (or overwrites) a column. The right-hand side can be a scalar (broadcasts), a Series (aligned by index), or a NumPy array (matched by position). For a cleaner method-chained form, use `df.assign(new=expr)` which returns a new DataFrame.

**Why it matters.** Every feature engineering step lives here. "Cost per person" = `df['cost_for_two'] / 2`. "Is weekend?" = `df['date'].dt.dayofweek >= 5`. "Score" = `df['rating'] / np.log(df['votes'])`. These are the building blocks of any ML feature set, and being fluent here is the difference between writing Pandas-feeling code and Python-loop code.

**How it works.**
1. `df['new'] = scalar` — broadcasts the scalar to every row.
2. `df['new'] = Series` — aligned by index; missing index entries become NaN.
3. `df['new'] = np.array` — aligned by *position*; lengths must match.
4. `df.assign(new=expr)` — same effect but returns a new DataFrame, so you can chain: `df.assign(x=...).assign(y=...).query('x > 0')`.

**Where it's used.**
- Feature engineering: derived columns from existing ones.
- Adding flags: `df['is_high_rated'] = df['rating'] > 4`.
- Bringing in a Series result: `df['rolling_mean'] = df['x'].rolling(7).mean()`.
- Conditional columns: `df['bucket'] = np.where(df['age'] >= 18, 'adult', 'minor')`.

**Related terms.**
- **`df.assign(**kwargs)`** — chainable equivalent; doesn't mutate.
- **`df.loc[mask, 'col'] = v`** — conditional write to *existing* column. ([walkthrough](#2g-guided))
- **`np.where(cond, a, b)`** — vectorised if/else for column creation. See [NumPy guide](./Data_Foundation_Revision_Guide.md#2g-guided).
- **`df.eval('new = a + b')`** — string-form column assignment; useful for huge tables (uses `numexpr`).

```python
df['cost_for_one'] = df['cost_for_two'] / 2
df['flag'] = df['rating'] > 4
df = df.assign(score=lambda d: d['rating'] / np.log(d['votes']))
df.loc[df['rating'].isna(), 'rating'] = 0    # conditional fill
```

**Gotcha.** Assigning a *Series* with a different index than `df` produces NaNs where the indexes don't match. If you mean position-based assignment, convert to a NumPy array first: `df['new'] = other_series.to_numpy()`.

### 🪞 Basic → Intermediate → Advanced — `iloc` vs `loc`

**Basic** — both work on the default integer index.
```python
df.iloc[0]                           # first row (by position)
df.loc[0]                            # row labeled 0
```

**Intermediate** — `loc` is the right tool once you have meaningful labels.
```python
df = df.set_index('name')
df.loc['Truffles']                   # by label
df.iloc[0]                           # still just "first row," whatever it is
```

**Advanced** — `.loc` is also the safe one-shot for conditional assignment. Two-step assignment via slicing triggers `SettingWithCopyWarning` because Pandas can't tell whether you mean the original or a view.
```python
df.loc[df['rating'] < 3, 'flag'] = -1            # right — one step
df[df['rating'] < 3]['flag'] = -1                # SettingWithCopyWarning — may or may not stick
```

### 🪞 Basic → Intermediate → Advanced — column access

**Basic** — bracket form returns a Series; double-bracket returns a DataFrame.
```python
df['rating']                         # Series
df[['rating']]                       # DataFrame with one column
```

**Intermediate** — derived columns are just assignment.
```python
df['cost_per_person'] = df['cost_for_two'] / 2
```

**Advanced** — `df.col` (attribute access) silently fails when the name has spaces, special chars, or collides with a method (`df.shape`, `df.index`, `df.size`). Bracket form is the only safe default.
```python
df.cost_for_two                      # works only if name is a clean identifier
df['cost for two (₹)']               # bracket form: handles any string
```

### 🧠 Concept cheat sheet (recap)

> Recap table — full definitions are in [the guided walkthrough above](#2g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **`.iloc[i]`** | Position-based row access. `iloc[0]` is **always** the first row, no matter what the index labels look like. | "Top N rows," index-agnostic code, train/test split by position. |
| **`.loc[label]`** | Label-based row access. `loc[0]` is the row whose **index label** is `0` — may not exist. | ID lookups, date slicing, safe conditional assignment. |
| **`.iloc` slice** | End-**exclusive** like NumPy: `iloc[1:5]` is 4 rows. | Positional ranges. |
| **`.loc` slice** | End-**inclusive** (Pandas's only exception): `loc['A':'C']` includes `C`. | Label ranges, time-series slices. |
| **`df['col']`** | Single brackets → **Series** (1D). | Single-column reads, target vectors `y`. |
| **`df[['c1','c2']]`** | List in brackets → **DataFrame** (2D). | Feature matrix `X`, multi-column reads. |
| **`df.col`** | Attribute access — same as `df['col']` **only if** name is a clean identifier and not a method collision. | Quick interactive reads on clean schemas; *never* in production code. |
| **Boolean filter `df[mask]`** | Pass a `Series[bool]` of length `len(df)`; only `True` rows survive. Combine with `&` / `\|` / `~` and parentheses. | Every row filter — by condition, multi-condition, or `.isin([...])`. |
| **`.isin([...])`** | Mask of "value is in this list?". | Categorical filtering: `df[df['city'].isin(['BLR','MAA'])]`. |
| **`.query('expr')`** | String-form filter — readable for multi-condition expressions. | Long compound filters; `df.query('a > 0 and b < 5')`. |
| **`set_index('col')`** | Promote a column to be the new row index; the column disappears. | Time-series prep, ID-based lookups, before `join`. |
| **`reset_index(drop=True)`** | Demote the index back to a column (or drop it). Default integer index returns. | After `groupby`; cleaning up after experimentation. |
| **`rename(columns={...})`** | Relabel columns by mapping old → new. Silently ignores missing keys. | Standardising messy column names from `read_csv`. |
| **`drop(columns=[...])`** | Remove columns by name. Use `index=[...]` for rows. Returns a new DataFrame. | Removing identifier columns; cleaning up after `merge`. |
| **Derived column** | `df['new'] = df['existing'] / 2` — scalar broadcasts, Series aligns by index. | Feature engineering — the entire `cost_for_one = cost_for_two / 2` family. |
| **`df.assign(new=expr)`** | Chainable equivalent of column assignment. Returns a new DataFrame. | Method chains: `df.assign(x=...).query(...).head()`. |
| **`inplace=True`** | Modify the DataFrame in place; returns `None`. Modern Pandas de-emphasises it. | Memory-tight loops only; prefer reassignment. |

### ⚙️ Top APIs

```python
df.iloc[i], df.iloc[i:j:k], df.iloc[:, j], df.iloc[i, j]
df.loc[label], df.loc[label, 'col'], df.loc[mask, 'col'] = val
df.set_index('col'), df.reset_index(drop=True, inplace=True)
df['col'], df[['c1','c2']], df.col
df.rename(columns={'old':'new'}, inplace=True)
df.drop(columns=['c1','c2']), df.drop(index=[0, 1])
df['new'] = df['existing'] * 2
df.assign(new=lambda d: d['x'] + d['y'])
df[df['x'] > 0], df.query('x > 0 and y < 5'), df[df['city'].isin([...])]
```

### 🧩 Code patterns

```python
# 1. Standardize column names early
df = df.rename(columns={
    'approx_cost(for two people)': 'cost_for_two',
    'listed_in(type)': 'listing_type',
    'rate': 'rating',
})

# 2. Index by a label
by_name = df.set_index('name')
by_name.loc['Truffles']

# 3. Filter
high_rated = df[df['rating'] > 4]

# 4. Filter + select column in one shot (safe assignment)
df.loc[df['rating'] < 3, 'flag'] = -1

# 5. Multi-column slice with iloc
df.iloc[0:5, 0:3]

# 6. Drop modelling-irrelevant columns
X = df.drop(columns=['id', 'name', 'url'])
```

### 🎯 Q&A — Module 2

1. **`.iloc` vs `.loc` — when do they differ even if both work?**
   `iloc[0]` is **always** the first row, regardless of how the index is labeled. `loc[0]` returns the row whose **label** is `0` — which may not be the first row after a sort, a filter, or a `set_index`. Use `iloc` for positional logic and `loc` for label logic. Also note: `iloc` slices are end-exclusive; `loc` slices are end-inclusive.

2. **`df['col']` vs `df.col` — any real difference?**
   Same Series most of the time. **But** `df.col` fails if the name has spaces, special chars, or collides with a method (e.g., `df.shape` is the property, not a column named `shape`). Bracket form is the safe default.

3. **What does `inplace=True` actually do, and should I use it?**
   Modifies the DataFrame in-place; returns `None`. Saves memory by skipping the copy. Modern Pandas (≥2.0) is de-emphasizing it because it complicates method-chaining and copy-on-write semantics. Prefer `df = df.rename(...)` unless memory is critical.

4. **What's `SettingWithCopyWarning` and how do you fix it?**
   When you assign to a slice that may be a view: `df[df['x']>0]['y'] = 1`. Pandas can't tell if you meant to modify `df` or a copy. Fix: use `.loc` in one shot — `df.loc[df['x']>0, 'y'] = 1`.

5. **What does `rename(columns={'nonexistent': 'x'})` do?**
   Nothing — it silently no-ops on missing keys. No error. Useful but a footgun if you typo a name and don't realize the rename didn't happen.

6. **What's the difference between `df.drop('col')` and `df.drop(columns='col')`?** *(adapted from `guipsamora/pandas_exercises`)*
   `df.drop('col')` defaults to `axis=0` — tries to drop a *row* labelled `'col'`, usually a `KeyError`. `df.drop(columns='col')` is the explicit modern form. Always use `columns=` or `index=` to avoid the axis-confusion bug.

7. **`df.loc['A':'C']` returns rows A, B, *and* C — why is that different from `iloc`?** *(common FAANG trap)*
   `loc` is **end-inclusive** — the only Pandas indexer that includes both endpoints. `iloc` is end-exclusive like NumPy / Python slicing. Mixing them up causes off-by-one bugs.

8. **You want rows where `city` is one of `['Bangalore', 'Chennai', 'Mumbai']`. What's the idiom?** *(common StrataScratch pattern)*
   `df[df['city'].isin(['Bangalore', 'Chennai', 'Mumbai'])]`. Clean, vectorised. Don't write a multi-condition `|` chain.

[🔝 Back to top](#top)

---

<a id="3-module3"></a>
## 3. Module 3 — Cleanup: dtype, strings, NaN, exploration

> Real-world pandas data is messy. This module is the cleanup toolkit.

### 🪜 Mental model

**Coerce, don't trust.** Every column from a CSV is a string until proven otherwise. Pandas guesses dtype from the first few rows; one stray `"NEW"` is enough to demote `rating` from `float64` to `object`, and now `.mean()` returns NaN. The cleanup pipeline is always: **strip → replace sentinels with NaN → `pd.to_numeric(errors='coerce')` → verify dtype**. *If a numeric op returns NaN unexpectedly, your column is still `object` — every time.*

<a id="3g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 3 concept. The cheat sheet below is the recap surface.

#### Dtype inference at read time

> **🪜 Mental model:** *Pandas guesses; you verify.* `read_csv` scans the column and picks a `dtype` — one bad value and the whole column becomes `object`.

**What it is.** When `pd.read_csv` loads a column, it scans a sample of rows and picks a `dtype` (data type) that can hold every value seen: all parseable as int → `int64`; any float → `float64`; any unparseable string → `object` (i.e., Python strings). This inference is *per column* — one column can be `int64`, another `float64`, another `object`. The decision is irreversible until you explicitly recast.

**Why it matters.** Inference is the single biggest source of "why is this column behaving weirdly?" bugs. A single `"NEW"` in an otherwise numeric column promotes it to `object`, which silently breaks `.mean()`, `.sum()`, `.describe()`, and downstream sklearn `fit()`. The fix is always one of: pre-clean the source, pass `na_values=[...]`, pass `dtype={...}`, or recast after load with `pd.to_numeric(errors='coerce')`.

**How it works.**
1. `read_csv` reads a sample (default: the whole file).
2. For each column, it tries `int64`; falls back to `float64` if any float-looking value; falls back to `object` if anything won't parse.
3. Values matching `na_values` (default includes `''`, `'NA'`, `'NaN'`, `'null'`, …) are flagged as missing first, *then* the dtype inference happens — so missing markers don't force `object`.
4. Once the dtype is chosen, every cell is stored in that representation. You can override with `dtype={'col': str}` at load time.

**Where it's used.**
- Every `read_csv` call you'll write.
- Anywhere you wonder "why is my column an `object` even though it looks like numbers?"
- Defending against silently-coerced ID columns (`"00123"` → `123`).

**Related terms.**
- **`object` dtype** — Pandas's catch-all for non-numeric (usually strings or mixed types); slow, no numeric ops. ([walkthrough](#3g-guided))
- **`na_values=[...]`** — list of strings to treat as missing *before* dtype inference; the cleanest fix.
- **`dtype={'col': str}`** — pin a column's dtype at load time; bypass inference entirely.
- **`pd.to_numeric(errors='coerce')`** — recast after load; bad values → NaN. ([walkthrough](#3g-guided))
- **Type priority** — same as NumPy's: string > float > int > bool. See [NumPy guide](./Data_Foundation_Revision_Guide.md#1g-guided).

```python
df = pd.read_csv('data.csv')
df.dtypes['rating']                  # 'object' — investigate!

# Three ways to fix:
pd.read_csv('data.csv', na_values=['N','NEW','-'])      # at load (best)
pd.read_csv('data.csv', dtype={'rating': float})        # forced — may raise
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # after load
```

**Gotcha.** Inference happens on a *sample* in some `read_csv` modes; one stray value early in the file is enough to demote the whole column to `object` even if 99% of cells are numeric.

#### `astype` — strict dtype conversion

> **🪜 Mental model:** *"Cast this column to this type — and fail loudly if anything won't parse."* Strict version of dtype coercion.

**What it is.** `df['col'].astype(dtype)` returns a new Series with every element converted to the requested `dtype` — `float`, `int`, `str`, `'category'`, etc. For numeric → numeric it does straight casting; for string → numeric it raises `ValueError` if *any* element won't parse. For float → int it **truncates** (chops the decimal — does *not* round). Inherits NumPy's casting rules.

**Why it matters.** `astype` is the right tool when you *trust* the data — clean numeric, clean strings, no surprises. It's strict by design so silent corruption isn't possible. The cleanup pipeline alternates between `astype` (when you've already cleaned) and `pd.to_numeric(errors='coerce')` (when you haven't).

**How it works.**
1. Pandas calls NumPy's `.astype()` on the underlying array.
2. NumPy walks the array, converting each element to the target dtype.
3. If any element can't be converted, `ValueError` — no partial result.
4. Returns a *new* Series; the original is unchanged.

**Where it's used.**
- After `pd.to_numeric(errors='coerce')` to lock in a precise dtype: `s = pd.to_numeric(s, errors='coerce').astype('float32')`.
- Shrinking memory: `df['amount'] = df['amount'].astype('float32')` (halves memory if precision allows).
- Categorical encoding: `df['city'] = df['city'].astype('category')` (fast filtering, low memory).
- Bool → int: `df['flag'].astype(int)` (True→1, False→0).

**Related terms.**
- **`pd.to_numeric(errors='coerce')`** — soft version; bad values become NaN instead of raising. ([walkthrough](#3g-guided))
- **`pd.to_datetime`** / **`pd.to_timedelta`** — type-specific soft converters.
- **`.astype('category')`** — special Pandas dtype for low-cardinality strings; saves memory.
- **Type priority** — same string > float > int > bool order as NumPy.

```python
df['votes'].astype(float)           # int → float, safe
df['flag'].astype(int)              # True → 1, False → 0
df['rating'].astype(float)          # ValueError if any string is unparseable!
df['price'].astype('float32')       # shrink memory
```

**Gotcha.** `astype(int)` on `1.9` returns `1` (truncates, doesn't round). To round properly: `df['x'].round().astype(int)` — or `np.round(arr).astype(int)`.

#### `pd.to_numeric` — safe numeric cast

> **🪜 Mental model:** *Astype that doesn't crash.* `errors='coerce'` turns unparseable values into NaN instead of raising.

**What it is.** `pd.to_numeric(series, errors='coerce')` is the standard "convert this Series to numbers" function. With `errors='raise'` (default), it behaves like `astype(float)` — crashes on bad data. With `errors='coerce'`, **any value it can't parse becomes `NaN`** and the cast succeeds. With `errors='ignore'`, it silently returns the original if parsing fails.

**Why it matters.** This is the single most useful cleanup function in Pandas. Almost every real-world dataset has columns that *look* numeric but have a few stray strings (`'NEW'`, `'-'`, `'?'`, empty strings). `to_numeric(errors='coerce')` cleans them in one step — bad values become NaN (which you can then `fillna` or `dropna`), the column becomes `float64`, and `.mean()` / `.sum()` / sklearn all start working.

**How it works.**
1. Pandas walks each element of the Series.
2. Tries to parse as int (or float); on success, stores it.
3. On parse failure, behaviour depends on `errors=`:
   - `'raise'` → throws `ValueError` (default).
   - `'coerce'` → stores `NaN`.
   - `'ignore'` → returns the original Series unchanged.
4. Returns a new Series with the appropriate numeric dtype.

**Where it's used.**
- Cleaning any object-dtype column that should be numeric: `df['rating'] = pd.to_numeric(df['rating'], errors='coerce')`.
- Right after stripping unit suffixes: `s.str[:-2].pipe(pd.to_numeric, errors='coerce')`.
- Building cleanup pipelines: strip → replace → `to_numeric(coerce)`.
- Validating: `pd.to_numeric(s, errors='coerce').isna().sum()` tells you how many rows are unrecoverable.

**Related terms.**
- **`astype(float)`** — strict version; raises on bad data. ([walkthrough](#3g-guided))
- **`pd.to_datetime(errors='coerce')`** — datetime equivalent; same `errors=` API.
- **`pd.to_timedelta`** — duration equivalent.
- **`np.nan`** — what bad values become with `coerce`. ([walkthrough](#3g-guided))
- **`downcast='integer'`** — shrink the resulting dtype if values fit (saves memory).

```python
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating'].isna().sum()           # how many rows lost
df['rating'].dtype                  # 'float64' — confirmed
```

**Gotcha.** `errors='ignore'` (the dangerous middle option) silently returns the *original* Series if any value won't parse — so you think you cast it, but the dtype is still `object`. Use `'coerce'` instead, always.

#### The `.str` accessor

> **🪜 Mental model:** *Vectorised string methods on a whole column.* `s.str.lower()`, `s.str.replace(',', '')`, `s.str.contains('foo')` — applied to every element in one call.

**What it is.** Every Series of `object` (or `string`) dtype exposes a `.str` accessor — a namespace of string methods that work on every element in parallel. It mirrors Python's string methods (`.lower`, `.upper`, `.strip`, `.split`, `.replace`, `.startswith`, `.contains`, …) but vectorised under the hood. It's **NaN-safe by default**: NaN values stay NaN, no error.

**Why it matters.** Every real-world dataset has text cleanup work — trim whitespace, lowercase categories, strip unit suffixes, extract substrings. The `.str` accessor is the idiomatic, fast way to do all of that without a `for` loop or `.apply(lambda x: x.lower())`. It's also a frequent interview ask: "clean this messy text column."

**How it works.**
1. `s.str` returns a `StringMethods` accessor object bound to the Series.
2. Each method (`.lower()`, `.replace(...)`) walks the underlying array, applying the string operation element-by-element.
3. NaN elements are preserved as NaN — no exception thrown.
4. Returns a new Series of the same length, usually still `object` dtype (or `bool` for predicates like `.contains`).

**Where it's used.**
- Lowercasing for case-insensitive grouping: `df['city'] = df['city'].str.lower()`.
- Stripping unit suffixes: `df['rating'] = df['rating'].str[:-2]` (drops last 2 chars).
- Stripping whitespace: `df['name'] = df['name'].str.strip()`.
- Substring search: `df[df['cuisine'].str.contains('North Indian', na=False)]`.
- Splitting and exploding: `df['tags'].str.split(',').explode()`.
- Replacing commas in numeric strings: `df['cost'].str.replace(',', '').astype(float)`.

**Related terms.**
- **`Series.replace`** — different! Operates on *whole values*, not substrings. ([walkthrough](#3g-guided))
- **`.str.contains(pat, na=False)`** — substring/regex predicate; pass `na=False` to treat NaN as False instead of NaN.
- **`.str.split(sep)`** — returns a Series of lists; chain with `.explode()` to expand into rows.
- **`.str.extract(regex)`** — extract regex groups into columns.
- **`.dt` accessor** — sibling accessor for `datetime64` columns.

```python
df['city'] = df['city'].str.lower().str.strip()
df['rating'] = df['rating'].str[:-2]                       # drop '/5'
df['cost'] = df['cost'].str.replace(',', '').astype(float) # commas out
df[df['cuisine'].str.contains('Indian', na=False)]          # substring filter
df['tags'].str.split(',', expand=True)                      # split into columns
```

**Gotcha.** `.str` methods only work on object/string dtypes. Calling `.str.lower()` on an int column raises. And `.str.contains(pat)` on a column with NaN returns NaN (not False) — pass `na=False` to treat NaN as not-matching.

#### `replace` — value replacement

> **🪜 Mental model:** *Swap whole values out.* `s.replace('NEW', np.nan)` replaces every *cell equal to* `'NEW'` with NaN — it does **not** do substring replacement.

**What it is.** `Series.replace(old, new)` (or `DataFrame.replace`) substitutes one value for another, **matching by whole-value equality**. Pass lists for multi-value swaps: `s.replace(['NEW', '-', ''], np.nan)`. Pass a dict for multiple mappings: `s.replace({'M': 'Male', 'F': 'Female'})`. Unlike `.str.replace`, it does not do substring matching — `'NEW york'` is not affected by `replace('NEW', ...)`.

**Why it matters.** Many real-world datasets use sentinel strings for missing or unknown data — `'NEW'`, `'-'`, `'?'`, `'N/A'`. Before you can `pd.to_numeric` the column, these must become NaN. `replace` is the dedicated tool for that swap. Confusing `replace` (whole-value) with `.str.replace` (substring) is one of the most common Pandas misunderstandings.

**How it works.**
1. Pandas walks each element of the Series.
2. Checks if the element matches any key in the mapping (value, list, or dict).
3. On match, swaps in the new value; on no match, keeps the original.
4. Returns a new Series of the same length.

**Where it's used.**
- Sentinel cleanup before numeric cast: `s.replace(['NEW', '-'], np.nan)` then `pd.to_numeric(errors='coerce')`.
- Recoding categories: `df['gender'].replace({'M': 'Male', 'F': 'Female'})`.
- Bulk fixes: `df.replace({np.inf: np.nan, -np.inf: np.nan})`.
- Cleanup after a merge: replace `'Unknown'` with NaN to keep downstream `isna()` consistent.

**Related terms.**
- **`.str.replace(pattern, new)`** — substring replacement; *different operation*. ([walkthrough](#3g-guided))
- **`.fillna(value)`** — special case for NaN-only replacement. ([walkthrough](#3g-guided))
- **`.map({old: new})`** — like `replace` for a Series but unmapped values become NaN (use carefully).
- **`np.where(cond, a, b)`** — conditional replacement based on a condition, not value-equality.

```python
df['rating'] = df['rating'].replace('NEW', np.nan)
df['rating'] = df['rating'].replace(['NEW', '-', ''], np.nan)
df['gender'] = df['gender'].replace({'M': 'Male', 'F': 'Female'})
df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
```

**Gotcha.** `replace` matches *whole values*. `s.replace('NEW', np.nan)` will NOT touch `'NEW york'`. Use `.str.replace('NEW', '')` for substring removal.

#### NaN — Pandas's missing-value sentinel

> **🪜 Mental model:** *The empty cell.* NaN is a special floating-point value that means "missing"; it sits in `float64` columns and never equals itself.

**What it is.** **NaN** ("Not a Number") is the IEEE-754 floating-point value Pandas uses to represent missing data. It only lives in `float` dtype columns — an `int` column can't hold NaN (Pandas auto-promotes to `float64` if you introduce NaN into ints). Key surprising property: **`NaN == NaN` is `False`** by definition. So you can't test for missingness with `== np.nan`; you must use `.isna()` / `.isnull()` / `.notna()` / `.notnull()`.

**Why it matters.** Missing data is everywhere — failed sensor reads, optional form fields, joined tables with no match. Pandas's whole missing-data API revolves around NaN. Understanding *(a)* how to detect it (`isna`), *(b)* how to count it (`isna().sum()`), *(c)* how to drop it (`dropna`), *(d)* how to fill it (`fillna`), and *(e)* the silent dtype promotion to `float64` is mandatory for any data role.

**How it works.**
- **Detection:** `s.isna()` / `s.isnull()` (aliases) return a boolean Series; True where the value is NaN. `s.notna()` is the negation. Never use `s == np.nan` — always False.
- **Counting:** `df.isna().sum()` gives per-column NaN counts.
- **Dropping:** `df.dropna()` drops any row with at least one NaN; `df.dropna(subset=['col'])` only considers `'col'`; `df.dropna(axis=1)` drops columns instead of rows.
- **Filling:** `s.fillna(0)` replaces NaN with `0`; `s.fillna(method='ffill')` forward-fills; `s.fillna(s.mean())` fills with the column mean.
- **Aggregations:** Pandas methods (`mean`, `sum`, etc.) **skip NaN by default**; NumPy's `np.mean` does not — use `np.nanmean` if you reach for NumPy.

**Where it's used.**
- Every EDA: `df.isna().sum().sort_values(ascending=False)` to find which columns are dirtiest.
- Pre-modelling imputation: `df['age'].fillna(df['age'].median())`.
- Cleanup after `pd.to_numeric(errors='coerce')` (which produces NaN for unparseable values).
- Guard clauses: `df.dropna(subset=['target'])` to ensure every training row has a label.

**Related terms.**
- **`isna` / `isnull`** — identical aliases; pick one and stick with it.
- **`notna` / `notnull`** — negation.
- **`fillna(value)` / `fillna(method='ffill')`** — replace missing with a value or via forward/backward fill.
- **`dropna(subset=, how='any'|'all')`** — drop rows with any (default) or all NaN.
- **`pd.NA`** — newer "missing" sentinel for nullable dtypes (`Int64`, `string`); fixes some NaN-in-int gotchas but is still spreading.
- **`np.nan` vs `None`** — Pandas treats both as NaN for missingness purposes; the underlying ndarray prefers `np.nan`.

```python
s.isna().sum()                      # how many missing
df.dropna()                         # drop any row with any NaN
df.dropna(subset=['rating'])        # only consider 'rating'
df['rating'].fillna(df['rating'].median())
df['x'] == np.nan                   # always False — bug!
df['x'].isna()                      # correct
```

**Gotcha.** `df['x'] == np.nan` is **always False** for every row. To detect NaN: `.isna()` / `.isnull()`. Forgetting this is the #1 NaN bug in pandas.

#### `unique` / `nunique` / `value_counts` — categorical exploration

> **🪜 Mental model:** *Three views of "what's in this column?".* `unique` gives distinct values; `nunique` counts them; `value_counts` gives the frequency table.

**What it is.** Three categorical-exploration methods on a Series:
- **`s.unique()`** — array of distinct values, in **order of first appearance**. Includes NaN if present.
- **`s.nunique()`** — single integer; count of distinct values. Excludes NaN by default; pass `dropna=False` to include.
- **`s.value_counts()`** — Series of `{value: frequency}`, sorted **descending by frequency**. Excludes NaN by default; pass `dropna=False` to include. Pass `normalize=True` for proportions instead of raw counts.

**Why it matters.** These three are the bread and butter of categorical EDA. "What cuisines exist?" (`unique`). "How many distinct cities?" (`nunique`). "What's the most common cuisine?" (`value_counts().head()`). They also surface data-quality issues — duplicates that should be one category (`'Bangalore'` vs `'bangalore'`), unexpected codes, rare classes you'll need to handle.

**How it works.**
1. `unique`: scans the Series, records each value the first time it appears, returns them in that order as a NumPy array.
2. `nunique`: just `len(s.unique())` with NaN handling.
3. `value_counts`: builds a frequency hash map (`value → count`), sorts by count descending, returns as a Series whose index is the values and whose values are the counts.

**Where it's used.**
- "How clean is this categorical column?": `df['cuisine'].value_counts()` to spot dirty / rare values.
- Top-N analysis: `df['location'].value_counts().head(10)` → top 10 locations.
- Cardinality check before encoding: `df['user_id'].nunique()` (if too high, don't one-hot encode).
- Missing data audit: `df['city'].value_counts(dropna=False, normalize=True)` — NaN might be the biggest bucket.

**Related terms.**
- **`describe(include=['object'])`** — coarser categorical summary (count, unique, top, freq).
- **`.value_counts(normalize=True)`** — proportions instead of raw counts; sums to 1.0.
- **`.value_counts(bins=10)`** — bucket continuous values into bins and count.
- **`pd.crosstab(a, b)`** — two-categorical frequency table (2D version).
- **`groupby('col').size()`** — almost-equivalent, but returns a different sort order.

```python
df['city'].unique()                           # distinct values in order of appearance
df['city'].nunique()                          # count of distinct
df['city'].value_counts().head(10)            # top 10 by frequency
df['city'].value_counts(dropna=False, normalize=True)  # include NaN, as proportions
```

**Gotcha.** `value_counts()` excludes NaN by default — but NaN might be the single most important fact about a column. Always run with `dropna=False` at least once during EDA to see whether NaN dominates.

#### `SettingWithCopyWarning` — the chained-assignment trap

> **🪜 Mental model:** *Pandas doesn't know if you meant the view or the original.* When you assign through a chained slice, the warning fires because *either* outcome is plausible — and Pandas refuses to silently pick.

**What it is.** `SettingWithCopyWarning` is the warning Pandas raises when you assign to a "chained" indexing expression — `df[df['x'] > 0]['y'] = 1` instead of the single-step `df.loc[df['x'] > 0, 'y'] = 1`. The chained form first builds a temporary filtered DataFrame (which may be a view or a copy — Pandas doesn't know in advance) and then assigns to its `'y'` column. The assignment may or may not propagate back to the original; the warning exists to flag that ambiguity.

**Why it matters.** This is the single most-asked Pandas warning in interviews ("what is SettingWithCopyWarning and how do you fix it?"). It's also the cause of the most insidious bugs in real code: the assignment silently doesn't stick on the original, the model trains on un-modified data, and the bug is invisible until results look wrong. The fix is always the same: replace chained assignment with single-step `.loc` assignment.

**How it works.**
1. `df[mask]` returns a temporary DataFrame — sometimes a view of the original, sometimes a copy. Pandas can't always tell.
2. Assigning to a column of that temporary (`temp['y'] = 1`) modifies the temporary. Whether it propagates to `df` depends on whether `temp` was a view or a copy.
3. Pandas detects this two-step pattern and raises `SettingWithCopyWarning` rather than silently producing one of two possible behaviours.
4. Single-step `.loc` assignment (`df.loc[mask, 'y'] = 1`) bypasses the ambiguity — Pandas knows you mean the original, period.

**Where it's used.**
- Conditional writes: replace `df[df['x'] < 0]['flag'] = -1` with `df.loc[df['x'] < 0, 'flag'] = -1`.
- After a filter: instead of `sub = df[df['city']=='X']; sub['z'] = 1`, do `sub = df[df['city']=='X'].copy(); sub['z'] = 1` (explicit copy).
- Pandas 2.x with "Copy-on-Write" mode (`pd.set_option('mode.copy_on_write', True)`) silences these by always copying.

**Related terms.**
- **`.loc[mask, 'col'] = val`** — the canonical single-step fix. ([walkthrough](#2g-guided))
- **`.copy()`** — explicit "I want a separate DataFrame": `sub = df[mask].copy()`.
- **View vs copy** — same underlying NumPy concept; see [NumPy guide](./Data_Foundation_Revision_Guide.md#1g-guided).
- **Chained indexing** — the general name for `df[...][...]` patterns; the source of the warning.
- **Copy-on-Write (CoW)** — Pandas 2.x option that fixes this by making views explicit.

```python
# BAD — triggers SettingWithCopyWarning, may or may not stick
df[df['rating'] < 3]['flag'] = -1

# GOOD — explicit, atomic, always works
df.loc[df['rating'] < 3, 'flag'] = -1

# Alternative — explicit copy, isolated mutation
sub = df[df['city'] == 'X'].copy()
sub['flag'] = -1
```

**Gotcha.** The warning is *not* an error — your code keeps running. So a careless developer might ignore it. Treat `SettingWithCopyWarning` as a bug, not a notice — every occurrence is a potential silent data-corruption issue.

### 🪞 Basic → Intermediate → Advanced — dtype coercion

**Basic** — strict cast works on clean data, raises on dirty.
```python
df['price'].astype(float)            # ValueError if any element isn't parseable
```

**Intermediate** — `pd.to_numeric` with `errors='coerce'` turns bad values into NaN instead of crashing.
```python
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating'].isna().sum()            # how many rows became NaN
```

**Advanced** — real cleanup chains strip → replace → coerce. Always verify `.dtype` after.
```python
df['rating'] = (
    df['rating'].astype(str).str[:-2]            # "4.1/5" → "4.1"
                .replace(['NEW', '-', ''], np.nan)
                .pipe(pd.to_numeric, errors='coerce')
)
assert df['rating'].dtype == 'float64'
```

### 🪞 Basic → Intermediate → Advanced — exploration

**Basic** — single-column profiling.
```python
df['city'].unique()                  # distinct values in order of appearance
df['city'].nunique()                 # count of distinct values
```

**Intermediate** — frequency table sorted by count.
```python
df['city'].value_counts().head(10)   # top 10 cities
```

**Advanced** — `value_counts` drops NaN by default; `normalize=True` returns proportions; combining the two answers "what % of records are missing this field?"
```python
df['city'].value_counts(dropna=False, normalize=True).head()
# NaN may now be the largest bucket — useful audit signal
```

### 🧠 Concept cheat sheet (recap)

> Recap table — full definitions are in [the guided walkthrough above](#3g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Dtype inference (`read_csv`)** | Pandas guesses each column's `dtype` from a sample. One bad value → whole column becomes `object`. | Every load. Pass `na_values=`, `dtype=`, or `thousands=` at read time to dodge the gotcha. |
| **`object` dtype** | Catch-all for non-numeric columns — strings, mixed types, or numerics polluted by stray strings. Slow; no numeric ops. | Spotting columns that need cleanup (`df.dtypes` after every load). |
| **`astype(t)`** | Strict cast — raises `ValueError` on any unparseable value. Float→int **truncates**, doesn't round. | When you trust the data is clean; or to lock in a dtype after `pd.to_numeric`. |
| **`pd.to_numeric(s, errors='coerce')`** | Safe numeric cast — bad values become NaN, column becomes `float64`. | The default cleanup step for any object column that should be numeric. |
| **`pd.to_datetime(errors='coerce')`** | Same idea, for date strings → `datetime64`. | Parsing messy date columns from CSV. |
| **`.str` accessor** | Vectorised string methods on a Series (`.str.strip()`, `.str.lower()`, `.str.replace()`, `.str.contains()`, `.str.split()`). NaN-safe. | Cleaning any text column — case, whitespace, substring removal, regex. |
| **`s.replace(old, new)`** | Whole-value substitution (not substring). Accepts lists and dicts. | Recoding categories, cleaning sentinel strings before `pd.to_numeric`. |
| **`.str.replace(pat, new)`** | Substring (or regex) substitution within strings. Distinct from `s.replace`. | Cleaning commas/units inside string-typed numerics. |
| **NaN** | Floating-point "missing" sentinel; lives in float columns. Never equals itself. | Detect with `isna()`, drop with `dropna()`, fill with `fillna()`. |
| **`isna()` / `notna()`** | Boolean Series — True where the value is / isn't NaN. | The *only* correct way to test for NaN (never `== np.nan`). |
| **`fillna(v)`** | Replace NaN with a value (scalar, Series, or `method='ffill'`/`'bfill'`). | Imputation; cleanup before modelling. |
| **`dropna()`** | Drop rows (default) or columns (`axis=1`) that contain NaN. Use `subset=` to limit. | Removing unrecoverable rows before model training. |
| **`.unique()`** | Distinct values, **in order of appearance**. Includes NaN. | Quick "what's in this column?" peek. |
| **`.nunique()`** | Count of distinct values. Excludes NaN by default. | Cardinality check before one-hot encoding. |
| **`.value_counts()`** | Frequency table, **sorted descending by frequency**. Excludes NaN by default. | Top-N analysis; spotting dirty / rare categories. |
| **`value_counts(dropna=False)`** | Include NaN in the frequency table. | EDA — see whether NaN is the dominant bucket. |
| **`value_counts(normalize=True)`** | Return proportions instead of counts. Sums to 1.0. | "What % of users are X?" |
| **`df.sort_values('col', ascending=False)`** | Sort rows by a column (or list of columns with mixed `ascending=[True, False]`). | Top-N reports; preparing data for rolling windows. |
| **`SettingWithCopyWarning`** | Pandas can't tell if you're modifying the original or a copy — fires on chained assignment. | Always treat as a bug. Fix: single-step `.loc[mask, 'col'] = v`. |

### ⚙️ Top APIs

```python
df['col'] = df['col'].astype(float)
df['col'] = pd.to_numeric(df['col'], errors='coerce')
df['col'].str.strip("'\""), df['col'].str[:-2], df['col'].str.lower()
df['col'].str.replace(',', ''), df['col'].str.contains('foo', case=False, na=False)
df['col'].replace('NEW', np.nan), df['col'].replace(['NEW', '-'], np.nan)

df['col'].unique(), df['col'].nunique(), df['col'].value_counts(dropna=False)
df.sort_values('col', ascending=False)
df.dropna(), df.fillna(0), df.isnull().sum()
df['col'].isna(), df['col'].notna()
```

### 🧩 Code patterns

```python
# 1. Clean a numeric column polluted with strings
df['rating'] = df['rating'].astype(str).str[:-2]              # "4.1/5" → "4.1"
df['rating'] = df['rating'].replace(['NEW', '-'], np.nan)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# 2. Cost-with-commas cleanup
df['cost_for_two'] = df['cost_for_two'].str.replace(',', '').astype(float)

# 3. Per-person cost
df['cost_for_one'] = df['cost_for_two'] / 2

# 4. Top categories
df['location'].value_counts().head(10)

# 5. NaN audit
df.isnull().sum().sort_values(ascending=False)

# 6. Safe conditional fill
df.loc[df['rating'].isna(), 'rating'] = df['rating'].median()
```

### 🎯 Q&A — Module 3

1. **`astype(float)` vs `pd.to_numeric(errors='coerce')`?**
   `astype(float)` raises on any unparseable value. `pd.to_numeric(errors='coerce')` converts bad values to `NaN`. The coerce path is the safer cleanup pattern; the strict path is for data you trust.

2. **Why does `df['rating'].mean()` return `NaN` even after dropping NaNs?**
   Because the column dtype is `object` (strings). Pandas treats numeric methods differently for non-numeric. Cast first: `pd.to_numeric(errors='coerce')`.

3. **`.unique()` vs `.value_counts()` — when each?**
   `.unique()` returns distinct values in order of appearance — quick check of "what categories exist." `.value_counts()` returns frequencies sorted descending — what you want for "what's most common."

4. **`value_counts()` excludes NaN by default — what's the fix?**
   Pass `dropna=False` to include `NaN` in the frequency table.

5. **You see `df['cost_for_two']` as `object`. The first 5 values look like `'1,200'`, `'1,500'`, `'800'`. What dtype was inferred and what's the fix?**
   Inferred as `object` because of the commas — Pandas couldn't parse `'1,200'` as a number.
   Fix: `df['cost_for_two'] = df['cost_for_two'].str.replace(',', '').astype(float)`. Or load with `pd.read_csv(thousands=',')` from the start.

6. **A column has values like `"4.1/5"`, `"NEW"`, `"-"`. Walk me through the cleanup.**
   ```python
   df['rating'] = df['rating'].astype(str).str[:-2]       # strip '/5'
   df['rating'] = df['rating'].replace(['NEW', '-', ''], np.nan)
   df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
   ```
   Then verify: `df['rating'].dtype` is `float64`, `df['rating'].isna().sum()` shows count of unrecoverable rows.

7. **Mean of a pandas Series skips NaN by default; NumPy's `mean` doesn't.** Know which you're using — same bug, different libraries.

8. **What's the difference between `s.replace('NEW', np.nan)` and `s.str.replace('NEW', '')`?** *(common interview trap)*
   `s.replace` matches **whole values** — only cells equal to `'NEW'` are touched. `s.str.replace` does **substring** replacement — `'NEW york'` becomes `' york'`. Mixing them up causes subtle bugs.

9. **You filter `df[df['x'] > 0]` and assign to a column — get `SettingWithCopyWarning`. What's happening?** *(common StrataScratch question)*
   Chained assignment: Pandas can't tell if your target is the original or a copy. Fix: single-step `df.loc[df['x'] > 0, 'col'] = val`. Or explicit copy: `sub = df[df['x'] > 0].copy()`.

10. **How would you detect if `'col'` has any missing values?** *(adapted from `ajcr/100-pandas-puzzles`)*
    `df['col'].isna().any()` returns one boolean. For per-row check across all columns: `df.isna().any(axis=1)`. Never use `== np.nan`.

[🔝 Back to top](#top)

---

<a id="4-terms"></a>
## 4. 📚 Pandas terms glossary

All key Pandas terms — alphabetical for quick lookup. Each entry is a **2–4 sentence beginner-friendly definition** with a back-link to its full walkthrough.

| Term | Definition |
|---|---|
| **Alignment** | Pandas's habit of matching rows by **index label** (not position) before any operation. `df_a + df_b` aligns indexes first; mismatched labels in the output get NaN. This is what makes `pd.concat`, `merge`, and arithmetic between Series "do the right thing" without you specifying alignment manually. ([walkthrough](#1g-guided)) |
| **`astype(t)`** | Strict dtype conversion — returns a new Series cast to type `t`, or raises `ValueError` if any value can't be parsed. Float → int **truncates** (chops the decimal — never rounds). The right tool when you trust the data; use `pd.to_numeric(errors='coerce')` instead when it's messy. ([walkthrough](#3g-guided)) |
| **Boolean filter** | `df[mask]` — pass a boolean Series the same length as `df`; only `True` rows survive. Combine conditions with `&` / `\|` / `~`, **always wrap each comparison in parens**. The single most-used Pandas operation. ([walkthrough](#2g-guided)) |
| **Coercion** | Forcing a value into a different type. Pandas does it automatically during `read_csv` (string > float > int priority) and explicitly via `astype` or `pd.to_numeric(errors='coerce')` (where bad values become NaN). The silent place where data goes wrong — always verify with `.dtypes` after. ([walkthrough](#3g-guided)) |
| **Column access (`df['col']` vs `df[['col']]`)** | Single brackets return a **Series** (1D); double brackets return a **DataFrame** (2D, one column). Many libraries (sklearn, plotting) require a DataFrame — getting this wrong causes silent shape bugs downstream. ([walkthrough](#2g-guided)) |
| **DataFrame** | Pandas's 2D labelled table — many Series glued by a shared row index. Columns can have different dtypes. The default container for tabular data in Python; output of `read_csv`, input to sklearn. ([walkthrough](#1g-guided)) |
| **Derived column** | A column you create from existing ones — `df['cost_for_one'] = df['cost_for_two'] / 2`. Pandas accepts scalars (broadcast), Series (aligned by index), or arrays (matched by position) on the right-hand side. The unit of feature engineering. ([walkthrough](#2g-guided)) |
| **`drop`** | Remove rows (`index=`) or columns (`columns=`) by label. Returns a new DataFrame (use `inplace=True` or reassign). Errors on missing labels by default; pass `errors='ignore'` to silence. ([walkthrough](#2g-guided)) |
| **`dropna`** | Drop rows that contain any NaN (default), or columns with `axis=1`. Use `subset=['col']` to limit the check to specific columns, and `how='all'` to require all values to be NaN. ([walkthrough](#3g-guided)) |
| **`dtype`** | The element type of a Series (one per column). Common ones: `int64`, `float64`, `object` (strings), `datetime64`, `bool`, `category`. Determines what operations are legal and how much memory the column uses. ([walkthrough](#3g-guided)) |
| **Dtype inference** | `pd.read_csv`'s automatic guess for each column's `dtype` based on the sample it reads. Robust on clean data; one stray string demotes the whole column to `object`. Defend with `na_values=`, `dtype=`, or `pd.to_numeric` after. ([walkthrough](#3g-guided)) |
| **Explicit index** | Any non-default index — strings, dates, IDs — set with `set_index('col')` or passed at construction. Once explicit, `loc` lookups become meaningful ("row for Truffles") and `iloc` ≠ `loc[0]`. ([walkthrough](#1g-guided)) |
| **`fillna`** | Replace NaN with a value, a Series, or via `method='ffill'` (forward fill) / `'bfill'` (backward fill). Common imputation: `df['x'].fillna(df['x'].median())`. Returns a new Series unless `inplace=True`. ([walkthrough](#3g-guided)) |
| **`head` / `tail` / `sample`** | The three first-peek methods. `head(n)` shows the first `n` rows, `tail(n)` the last, `sample(n)` `n` random rows (`random_state=` for reproducibility). Reach for `sample` when the data might be sorted and `head` would mislead. ([walkthrough](#1g-guided)) |
| **`iloc`** | Position-based row/column access. `iloc[0]` is **always** the first row regardless of index labels; slices are **end-exclusive** like NumPy. The right tool when index labels are meaningless or you want index-agnostic code. ([walkthrough](#2g-guided)) |
| **Implicit index** | The default integer index Pandas attaches when you don't set one (`RangeIndex(0, n)`). Position and label agree on this index — they diverge the moment you sort, filter, or `set_index`. ([walkthrough](#1g-guided)) |
| **Index** | The row labels every Series and DataFrame carries. By default an integer `RangeIndex`; can be replaced with strings, dates, IDs, or a `MultiIndex`. Used by `.loc`, by `groupby`, by joins, by alignment. ([walkthrough](#1g-guided)) |
| **`info()`** | The X-ray method — prints every column's dtype, non-null count, and total memory in one call. Returns `None` (never assign it). The standard second command after `read_csv`. ([walkthrough](#1g-guided)) |
| **`inplace=True`** | A method-argument flag that modifies the DataFrame in place and returns `None`. Modern Pandas (≥2.0) de-emphasises it because it complicates method-chaining and copy-on-write semantics. Prefer reassignment (`df = df.rename(...)`). ([walkthrough](#2g-guided)) |
| **`isna` / `isnull`** | Identical aliases — return a boolean Series, True where the value is NaN. The **only** correct way to test for missingness in Pandas; `== np.nan` is always False. ([walkthrough](#3g-guided)) |
| **`loc`** | Label-based row/column access. `loc[label]` returns the row with that index label; slices are **end-inclusive** (the only Pandas indexer where this is true). Also the safe form for conditional assignment: `df.loc[mask, 'col'] = v`. ([walkthrough](#2g-guided)) |
| **NaN (Not a Number)** | The floating-point sentinel Pandas uses for missing values. Only lives in `float` (and `object`, `datetime`) dtypes; an `int` column auto-promotes to `float64` if you introduce NaN. Key gotcha: `NaN == NaN` is False — always use `isna()`. ([walkthrough](#3g-guided)) |
| **`nunique`** | Count of distinct values in a Series. Excludes NaN by default; pass `dropna=False` to include. Cardinality check — useful before one-hot encoding (high cardinality → avoid one-hot). ([walkthrough](#3g-guided)) |
| **`object` dtype** | Pandas's catch-all dtype for non-numeric — usually strings, sometimes mixed types. Slow because each cell is a separate Python object (no NumPy vectorisation). One stray string in a numeric column silently promotes the whole column to `object`. ([walkthrough](#3g-guided)) |
| **`pd.read_csv`** | The front door — load a CSV (or URL) into a DataFrame. Auto-infers dtypes from a sample. Key arguments: `na_values=`, `dtype=`, `parse_dates=`, `thousands=`, `sep=`, `header=`. 90% of "messy data" pain can be fixed by passing the right ones. ([walkthrough](#1g-guided)) |
| **`pd.to_numeric(errors='coerce')`** | Safe numeric cast — bad values become NaN instead of raising. The cleanup default; pairs naturally with downstream `fillna` / `dropna`. The strict alternative is `astype(float)`. ([walkthrough](#3g-guided)) |
| **`RangeIndex`** | The default integer index Pandas attaches when none is specified — `0, 1, …, n-1`. Lightweight (stored as start/stop/step, not actual values). Same as "implicit index." ([walkthrough](#1g-guided)) |
| **`rename`** | Relabel columns or index entries by passing a dict mapping old → new. Silently ignores missing keys, which is convenient but a footgun for typos. Standard first cleanup step after `read_csv`. ([walkthrough](#2g-guided)) |
| **`replace`** | Whole-value substitution — `s.replace('NEW', np.nan)` swaps every cell equal to `'NEW'`. Different from `.str.replace`, which does substring replacement. Accepts lists and dicts for bulk swaps. ([walkthrough](#3g-guided)) |
| **`reset_index`** | Demote the current index back to a regular column (or drop it with `drop=True`). Standard cleanup after `groupby`. Without `drop=True`, the old index becomes a new column. ([walkthrough](#2g-guided)) |
| **`sample`** | Random `n` (or fractional `frac=`) rows. Pass `random_state=42` for reproducibility. The antidote to `head`'s sorting bias. ([walkthrough](#1g-guided)) |
| **Series** | Pandas's 1D labelled array — values + index + optional name. Every column of a DataFrame is a Series. Built on a NumPy ndarray, so vectorised operations are fast. ([walkthrough](#1g-guided)) |
| **`set_index`** | Promote a column to be the new row index — the column disappears from the data and becomes labels. Unlocks meaningful `.loc` lookups (time-series slicing, ID-based reads). Pair with `reset_index` to revert. ([walkthrough](#2g-guided)) |
| **`SettingWithCopyWarning`** | The warning Pandas raises on chained assignment (`df[mask]['col'] = v`) because it can't tell whether your target is a view or a copy. Always treat as a bug — fix with single-step `.loc[mask, 'col'] = v` or explicit `.copy()`. ([walkthrough](#3g-guided)) |
| **`.shape` / `.columns` / `.dtypes`** | Three free attributes (no parens) — `(rows, cols)` tuple, column-names Index, column → dtype Series. The cheapest sanity checks; call them before and after every transformation. ([walkthrough](#1g-guided)) |
| **`sort_values`** | Sort rows by one or more columns. Pass a list of columns and `ascending=[True, False]` for mixed directions. Common preparation step for top-N reports and rolling windows. ([walkthrough](#3g-guided)) |
| **`.str` accessor** | Namespace of vectorised string methods on a Series — `.str.lower()`, `.str.strip()`, `.str.replace()`, `.str.contains()`, `.str.split()`. NaN-safe by default. Only works on `object` / `string` dtype. ([walkthrough](#3g-guided)) |
| **Type priority** | The order Pandas (and NumPy under it) use when promoting mixed types to one dtype: **string > float > int > bool**. One stray string demotes a whole column to `object`. Same rule as NumPy. ([walkthrough](#3g-guided)) |
| **`unique`** | Array of distinct values in a Series, **in order of first appearance**. Includes NaN if present. Quick "what's in this column?" peek. ([walkthrough](#3g-guided)) |
| **`value_counts`** | Frequency table for a Series — value → count, sorted descending by count. Excludes NaN by default (`dropna=False` to include); pass `normalize=True` for proportions. ([walkthrough](#3g-guided)) |

[🔝 Back to top](#top)

---

<a id="5-apis"></a>
## 5. ⚙️ API cheat sheet — Pandas only

### Creation & I/O
| Call | Purpose |
|---|---|
| `pd.Series(arr, index=...)` | 1D labeled array |
| `pd.DataFrame({...})` / `pd.DataFrame([[...]])` | 2D table |
| `pd.concat([...], axis=0/1)` | Glue along axis |
| `pd.read_csv(path)` | Load CSV |
| `df.to_csv(path, index=False)` | Save CSV |

### Selection
| Call | Purpose |
|---|---|
| `df.head(n)` / `df.tail(n)` / `df.sample(n)` | First / last / random n rows |
| `df.iloc[i, j]` | Position-based |
| `df.loc[label, 'col']` | Label-based |
| `df['col']` / `df[['c1','c2']]` | Single / multi column |
| `df.set_index('col')` / `df.reset_index(drop=True)` | Index manip |
| `df[df['x'] > 0]`, `df.query('x > 0')`, `df[df['x'].isin([...])]` | Boolean filter |

### Info & cleanup
| Call | Purpose |
|---|---|
| `df.info()` | Schema + nulls |
| `df.describe(include='all')` | Stats |
| `df.rename(columns={...})` | Rename |
| `df.drop(columns=[...])` / `df.drop(index=[...])` | Remove |
| `df['c'].astype(t)` | Cast (strict) |
| `pd.to_numeric(s, errors='coerce')` | Cast (NaN on fail) |
| `s.str.strip / .replace / .lower / .[i:j]` | Vectorized string ops |
| `s.replace(old, new)` | Replace values |
| `s.unique() / .nunique() / .value_counts()` | Distinct / count / frequency |
| `df.sort_values('col', ascending=False)` | Sort rows |
| `df.dropna() / df.fillna(v)` | NaN handling |
| `df.isnull().sum()` | Per-column NaN counts |

[🔝 Back to top](#top)

---

<a id="6-gotchas"></a>
## 6. ⚠️ Gotchas & traps

1. **`df.col` fails on names with spaces or special characters.** Use `df['col']`.
2. **`.describe()` skips object columns by default.** Use `include='all'` or fix dtypes.
3. **`astype(float)` raises on a single bad string.** Use `pd.to_numeric(errors='coerce')` for messy data.
4. **`SettingWithCopyWarning`:** assign via `.loc` in one step — `df.loc[mask, 'col'] = val`.
5. **`rename(columns={'nonexistent': 'x'})` doesn't error** — it silently does nothing.
6. **`.iloc[0]` ≠ `.loc[0]` when the index is non-default.**
7. **`pd.concat([s1, s2], axis=1)` with mismatched indexes** → union with NaN fills.
8. **`inplace=True` returns `None`** — never write `df = df.rename(..., inplace=True)`.
9. **Mean of a Pandas Series skips NaN by default; NumPy's `mean` doesn't.** Know which you're using.
10. **`value_counts()` excludes NaN by default.** Pass `dropna=False` to include.
11. **`pd.read_csv` infers dtypes from the data** — one stray `'N'` and the whole column becomes `object`.
12. **`reset_index()` without `drop=True` keeps the old index as a new column.**
13. **`df['col'].str` requires the column to be string-typed** — fails on numeric/object-with-mixed.
14. **`df['col'] == np.nan` is always False.** Use `df['col'].isna()`.
15. **Reading IDs like `"00123"`?** Pandas drops leading zeros if it parses as int. Use `dtype={'id': str}` on `read_csv`.
16. **`s.replace('NEW', np.nan)` ≠ `s.str.replace('NEW', '')`.** Former matches whole values; latter matches substrings.
17. **`.loc[mask]` slice is end-inclusive; `.iloc[i:j]` is end-exclusive.** Off-by-one bugs live here.
18. **`df.drop('col')` without `columns=` tries to drop a *row* labelled `'col'`.** Always use `columns=` or `index=`.

[🔝 Back to top](#top)

---

<a id="7-advanced"></a>
## 7. 🎯 Advanced interview Q&A

> Mix of original drills and questions adapted from `guipsamora/pandas_exercises`, `ajcr/100-pandas-puzzles`, `alexeygrigorev/data-science-interviews`, and common StrataScratch / LeetCode patterns.

**Q1. Difference between `df.iloc[0]` and `df.loc[0]`?** *(very common opener)*
`iloc` is **position**-based — first row. `loc` is **label**-based — the row whose index equals `0`. They diverge after `set_index`, sort, or filter. Also note `iloc` slices are end-exclusive; `loc` slices are end-inclusive.

**Q2. Why might `df['rating'].mean()` return `NaN` even after dropping NaNs?**
Because the column dtype is `object` (strings) — Pandas treats numeric methods differently for non-numeric. Cast first: `pd.to_numeric(errors='coerce')`.

**Q3. What's `SettingWithCopyWarning` and how do you fix it?**
Pandas can't tell if you're modifying the original or a copy. Fix: single-step `.loc` assignment — `df.loc[mask, 'col'] = val`. Treat the warning as a bug, not a notice.

**Q4. `pd.concat([s1, s2], axis=1)` with mismatched indexes — what's the result?**
Union join with NaN fills. Use `join='inner'` to keep only intersecting labels.

**Q5. `astype(float)` vs `pd.to_numeric(errors='coerce')` — which and when?**
`astype` is strict — raises on bad data. `to_numeric(coerce)` returns NaN for bad data. Cleanup pipelines almost always use `coerce`.

**Q6. A column has values like `"4.1/5"`, `"NEW"`, `"-"`. Walk me through the cleanup.**
```python
df['rating'] = df['rating'].astype(str).str[:-2]       # strip '/5'
df['rating'] = df['rating'].replace(['NEW', '-', ''], np.nan)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
```
Then verify: `df['rating'].dtype` is `float64`, `df['rating'].isna().sum()` shows count of unrecoverable rows.

**Q7. You see `df['cost_for_two']` as `object`. First 5 values: `'1,200'`, `'1,500'`, `'800'`. What's the fix?**
Inferred as `object` because of the commas. Fix: `df['cost_for_two'] = df['cost_for_two'].str.replace(',', '').astype(float)`. Or load with `pd.read_csv(thousands=',')`.

**Q8. The dataset has 51K rows; you want top 10 locations. How do you do it in three lines?**
```python
top = df['location'].value_counts().head(10)
top.plot(kind='bar')
```

**Q9. Compute "value score" = rating ÷ cost_for_two, then top 5.**
```python
df['value_score'] = df['rating'] / df['cost_for_two']
df.nlargest(5, 'value_score')[['name', 'rating', 'cost_for_two', 'value_score']]
```

**Q10. You load a CSV but ID column "00123" gets stored as `123` — leading zeros lost. Best strategy?**
Pass `dtype={'id': str}` to `pd.read_csv` so the column is read as string from the start. IDs are identifiers, not numbers.

**Q11. `s.str.upper()` on a Series containing `np.nan` — what happens?**
NaN is preserved; no error. The `.str` accessor is NaN-safe by design.

**Q12. `s.replace('NEW', np.nan)` vs `s.str.replace('NEW', '')` — different operations?**
Yes. `replace` matches whole values (a cell containing exactly `'NEW'`). `.str.replace` matches substrings (a cell containing `'NEW york'` becomes `' york'`). Mixing them up is a popular interview trap.

**Q13. How do you filter for rows where a categorical column is in a specific list?** *(adapted from `guipsamora/pandas_exercises`)*
`df[df['city'].isin(['Bangalore', 'Chennai', 'Mumbai'])]`. Vectorised, single-line, no `|` chain.

[🔝 Back to top](#top)

---

<a id="8-businessmap"></a>
## 8. Business questions → which Pandas API

| Business question | API |
|---|---|
| "Which restaurants are above 4★?" | `df[df['rating'] > 4]` |
| "Top 10 locations by restaurant count?" | `df['location'].value_counts().head(10)` |
| "Cost per person?" | `df['cost_for_two'] / 2` |
| "Filter high-vote and low-cost?" | `df[(df['votes']>500) & (df['cost']<500)]` |
| "Replace 'NEW' ratings with NaN?" | `df['rating'].replace('NEW', np.nan)` |
| "How many distinct locations?" | `df['location'].nunique()` |
| "Sort by votes descending?" | `df.sort_values('votes', ascending=False)` |
| "How many NaNs per column?" | `df.isnull().sum()` |
| "Save cleaned data?" | `df.to_csv('clean.csv', index=False)` |

[🔝 Back to top](#top)

---

<a id="sourced-bank"></a>
## 🌐 Sourced interview questions

> **Real questions paraphrased from canonical pandas interview-prep sources.** Use this as your standalone practice bank — no internet required. Each batch keeps the source's original numbering.

### Batch 1 — from [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles)

| # | Question | Answer |
|---|---|---|
| 1 (#4) | Create a DataFrame from a dict with custom index labels. | `pd.DataFrame({'x':[1,2]}, index=['a','b'])` |
| 2 (#5) | Show schema + null counts + memory for a DataFrame. | `df.info()` |
| 3 (#6) | Return the first N rows. | `df.head(n)` |
| 4 (#7) | Select specific columns by name. | `df[['c1','c2']]` |
| 5 (#8) | Select rows + columns using label-based indexing. | `df.loc[labels, ['c1','c2']]` |
| 6 (#9) | Filter rows where a numeric column exceeds a threshold. | `df[df['x'] > k]` |
| 7 (#10) | Select rows with any NaN values. | `df[df.isna().any(axis=1)]` |
| 8 (#11) | Apply multiple conditions to filter rows. | `df[(df['x'] > 0) & (df['y'] < 5)]` (parens required) |
| 9 (#12) | Select rows where col is between two values (inclusive). | `df[df['x'].between(lo, hi)]` |
| 10 (#13) | Modify a value at a specific row/column location. | `df.loc[idx, 'col'] = v` |
| 11 (#14) | Sum of an entire column. | `df['col'].sum()` |
| 12 (#15) | Mean per group using groupby. | `df.groupby('k')['x'].mean()` |
| 13 (#17) | Count occurrences of each category. | `df['col'].value_counts()` |
| 14 (#18) | Sort by multiple columns with mixed asc/desc. | `df.sort_values(['a','b'], ascending=[True, False])` |
| 15 (#19) | Replace categorical values with booleans. | `df['flag'] = df['cat'].map({'Y': True, 'N': False})` |
| 16 (#20) | Replace string values in a column. | `df['col'].str.replace('old', 'new', regex=False)` |
| 17 (#21) | Create a pivot table with aggregation. | `pd.pivot_table(df, index=, columns=, values=, aggfunc='mean')` |
| 18 (#22) | Remove duplicate rows. | `df.drop_duplicates()` |
| 19 (#23) | Subtract row mean from each element. | `df.sub(df.mean(axis=1), axis=0)` |
| 20 (#24) | Find the column with the minimum sum. | `df.sum().idxmin()` |
| 21 (#25) | Count unique rows (ignore duplicates). | `len(df.drop_duplicates())` |
| 22 (#26) | Position of the Nth NaN in each row. | `df.isna().cumsum(axis=1).eq(n).idxmax(axis=1)` |
| 23 (#27) | Sum the top-3 values per group. | `df.groupby('k')['v'].apply(lambda g: g.nlargest(3).sum())` |
| 24 (#28) | Bin continuous values and sum by interval. | `df.groupby(pd.cut(df['x'], bins=...))['v'].sum()` |
| 25 (#33) | Create a DatetimeIndex of business days. | `pd.bdate_range(start, end)` |
| 26 (#35) | Resample a time series by month and aggregate. | `df.set_index('ts').resample('M').sum()` |
| 27 (#43) | Expand a list-valued column into separate rows. | `df.explode('col')` |

### Batch 2 — common FAANG/StrataScratch-style patterns

| # | Question (paraphrased) | One-liner answer |
|---|---|---|
| 28 | (Amazon, Medium) Filter products with sales above their category's median. | `df[df.groupby('cat')['sales'].transform('median') < df['sales']]` |
| 29 | (Meta, Medium) For each user, return their top-N events per day. | `df.groupby(['user','day'])['event'].rank(ascending=False).le(N)` |
| 30 | (Google, Easy) Compute a 7-day rolling average. | `df.set_index('date').rolling('7D').mean()` |
| 31 | (Common) Why does `df['rating'].mean()` return NaN after cleanup? | Column dtype is still `object` — cast with `pd.to_numeric(errors='coerce')` first. |
| 32 | (Common) Difference between `.iloc[0]` and `.loc[0]`? | Position vs label. Identical until `set_index`/sort/filter shifts the labels. |

### Citations & where to drill more
- 🎯 [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles) — full 100 puzzles with solutions.
- 🎯 [`guipsamora/pandas_exercises`](https://github.com/guipsamora/pandas_exercises) — 10 graded chapters.
- 🎯 **StrataScratch** — real pandas/SQL from Amazon, Meta, Google.
- 🎯 **LeetCode → Pandas track** — graded tests on canonical patterns.

[🔝 Back to top](#top)

---

<a id="9-drill"></a>
## 9. 🔁 Revision drill (50 questions)

Designed as a **timed pre-interview tool**. Read each question, answer in your head, peek. Aim for under 15 seconds per question.

### Block A — Series, DataFrame, I/O (Q1–15)

1. Series? → **1D labeled array**
2. DataFrame? → **2D labeled table**
3. Default index? → **Integers 0, 1, 2…**
4. Build df from dict — keys become? → **Column names**
5. Preview first 5 rows? → **`df.head()`**
6. Schema + nulls + dtypes? → **`df.info()`**
7. Summary stats? → **`df.describe()`**
8. `.describe()` default columns? → **Numeric only**
9. Include all in describe? → **`include='all'`**
10. Shape? → **`df.shape`**
11. Glue two tables side-by-side? → **`pd.concat([a, b], axis=1)`**
12. Concat with mismatched indexes? → **Union with NaN fills**
13. Save CSV no index? → **`to_csv('x.csv', index=False)`**
14. Read CSV preserving leading-zero IDs? → **`dtype={'id': str}`**
15. Read CSV with `'1,200'` numerics? → **`thousands=','`**

### Block B — Selection (Q16–28)

16. `iloc` is? → **Position-based**
17. `loc` is? → **Label-based**
18. `iloc[0]` after `set_index`? → **Still first row**
19. `loc[0]` after `set_index`? → **Row labeled `0` (may not exist)**
20. `set_index('col')` does? → **Make column the index**
21. `reset_index(drop=True)`? → **Default int index; old discarded**
22. `inplace=True`? → **Modify in place; returns None**
23. `df['col']` returns? → **Series**
24. `df[['c1','c2']]` returns? → **DataFrame**
25. `df.col` fails when? → **Name has space/special char/collides with method**
26. Rename a column? → **`df.rename(columns={'old':'new'})`**
27. Rename silent on missing? → **Yes**
28. Safe single-step assignment? → **`df.loc[mask, 'col'] = val`**

### Block C — Cleanup & exploration (Q29–45)

29. Strict cast? → **`.astype(float)`**
30. Safe cast? → **`pd.to_numeric(errors='coerce')`**
31. String accessor? → **`.str`** (e.g., `.str.strip()`)
32. `.str.upper()` on NaN? → **NaN preserved**
33. Replace value? → **`s.replace('NEW', np.nan)`**
34. Replace multiple values? → **`s.replace(['NEW','-'], np.nan)`**
35. Distinct values? → **`.unique()`**
36. Count of distinct? → **`.nunique()`**
37. Frequency table? → **`.value_counts()`**
38. value_counts excludes NaN? → **Yes by default — `dropna=False` to include**
39. Sort rows by col? → **`df.sort_values('col', ascending=False)`**
40. Sort multi-col? → **`df.sort_values(['c1','c2'], ascending=[True,False])`**
41. Per-column NaN count? → **`df.isnull().sum()`**
42. Drop NaN rows? → **`df.dropna()`**
43. Fill NaN with value? → **`df.fillna(0)`**
44. `df['x'] == np.nan` returns? → **All False — use `.isna()`**
45. Strip `/5` from `"4.1/5"`? → **`s.astype(str).str[:-2]`**

### Block D — Application & traps (Q46–50)

46. `SettingWithCopyWarning` fix? → **Use `.loc` in one step**
47. Why `df['rating'].mean()` returns NaN after cleanup? → **Column still `object` dtype**
48. Filter compound? → **`df[(df.x>0) & (df.y<5)]`**
49. NaN-aware mean (Pandas)? → **`.mean()` skips NaN by default**
50. `inplace=True` return value? → **`None`** — never re-assign

**Score yourself:** 45+ = strong, 38–44 = solid, 30–37 = revise, <30 = re-read modules.

[🔝 Back to top](#top)

---

<a id="10-bestpractices"></a>
## 10. ✅ Best practices

1. **Always start with `df.shape`, `df.info()`, `df.head()`** when handed a new dataset.
2. **Standardize column names** early (rename) — saves typos and shame.
3. **Cast strings to numeric early** with `pd.to_numeric(errors='coerce')`.
4. **Use `pd.read_csv(thousands=',', na_values=['N','NEW','-'])`** to clean at load time.
5. **Save intermediate cleaned data** with `.to_csv` — cleaning is wasteful to repeat.
6. **One-step `.loc` assignment** to avoid `SettingWithCopyWarning`.
7. **Method-chain cleanly:** `df.rename(...).assign(...).query(...)` reads better than mutating `inplace`.
8. **Don't use `df.col` if the name has spaces / special chars** — use `df['col']`.
9. **Verify dtypes after cleanup.** `df.dtypes` is the cheapest sanity check.
10. **Use `dtype={...}`** on `read_csv` to lock in expected types — esp. for ID columns.

[🔝 Back to top](#top)
