<a id="top"></a>
# Pandas — Master Revision Guide

> **Standalone pandas revision sheet.** Extracted and expanded from the original Data Foundation guide so you can drill Series / DataFrame basics independently from NumPy. Pair it with [`Data_Foundation_Revision_Guide.md`](./Data_Foundation_Revision_Guide.md) for NumPy, and with [`Amazon_Sachin_EDA_Revision_Guide.md`](./Amazon_Sachin_EDA_Revision_Guide.md) for advanced pandas (joins, groupby, apply, reshape, datetime, plotting).

**How to use:**
- **Pre-interview:** read the "🚀 Topic finder" → skim the cheat sheets → drill the Q&A and trap list.
- **Just before a coding round:** run the [§9 Revision Drill](#9-drill).
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
| Series, DataFrame, indices | [Module 1](#1-module1) → [Cheat sheet](#1c-cheat) |
| `pd.read_csv`, `to_csv`, `head`, `info`, `describe` | [Module 1](#1-module1) |
| `.iloc` vs `.loc`, `set_index`/`reset_index` | [Module 2](#2-module2) |
| Column ops, derived columns, `rename`, dot-vs-bracket | [Module 2](#2-module2) |
| `astype`, `pd.to_numeric(errors='coerce')`, string accessor | [Module 3](#3-module3) |
| `replace`, NaN, `unique`/`nunique`/`value_counts` | [Module 3](#3-module3) |
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

1. [Module 1 — Series, DataFrame, I/O](#1-module1)
2. [Module 2 — Selection: `.iloc` / `.loc` / column ops](#2-module2)
3. [Module 3 — Cleanup: dtype, strings, NaN, exploration](#3-module3)
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

<a id="1c-cheat"></a>
### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Series | 1D labeled array (one column of a table) |
| DataFrame | 2D labeled table (rows × columns) |
| Index | Row labels — implicit (`0,1,2`) or explicit (custom) |
| Implicit index | Default integer 0,1,2,… index |
| Explicit index | Custom-labeled DataFrame index |
| `pd.read_csv` | Load CSV → DataFrame; auto-infers dtypes |
| `pd.concat(axis=0/1)` | Glue Series/DataFrames along rows (`0`) or columns (`1`) |
| `.head(n)` / `.tail(n)` | First / last n rows |
| `.info()` | Schema + null counts + dtype + memory footprint |
| `.describe()` | Summary stats — **numeric columns only by default** |
| `.shape` | `(rows, cols)` tuple |
| `.to_csv('out.csv', index=False)` | Export CSV; drop the row-index column |

### ⚙️ Top APIs

```python
pd.Series(arr, index=[], name='')
pd.DataFrame({...}) / pd.DataFrame([[...]])
pd.concat([s1, s2], axis=1)
pd.read_csv('path', sep=',', header=0, na_values=[...], dtype={...}, thousands=',')

df.head(n), df.tail(n)
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

[🔝 Back to top](#top)

---

<a id="2-module2"></a>
## 2. Module 2 — Selection: `.iloc` / `.loc` / column ops

> How to grab the rows and columns you actually want.

### 🪜 Mental model

**`iloc` is the row *number*; `loc` is the row *name*.** After a fresh `read_csv`, both refer to the same row because the default index is `0, 1, 2, …`. The instant you `set_index`, sort, or filter, the **label** and the **position** diverge — and so do `iloc[0]` and `loc[0]`. *Position is "show me the 3rd row"; label is "show me the row called X."* Pick whichever question you're actually asking.

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

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| `.iloc[i]` | Position-based row access (always first row for `iloc[0]`) |
| `.loc[label]` | Label-based row access |
| `df['col']` | Single column → Series |
| `df[['c1','c2']]` | Multiple columns → DataFrame |
| `df.col` | Same as `df['col']` **only if** name is a clean identifier |
| `set_index('col')` | Make a column the new index |
| `reset_index(drop=True)` | Revert to default int index; `drop` discards old |
| `inplace=True` | Modify the DataFrame in place; returns `None` |
| Derived column | `df['new'] = df['existing'] / 2` |
| `df.rename(columns={...})` | New names; silent on missing keys |

### ⚙️ Top APIs

```python
df.iloc[i], df.iloc[i:j:k], df.iloc[:, j], df.iloc[i, j]
df.loc[label], df.loc[label, 'col'], df.loc[mask, 'col'] = val
df.set_index('col'), df.reset_index(drop=True, inplace=True)
df['col'], df[['c1','c2']], df.col
df.rename(columns={'old':'new'}, inplace=True)
df['new'] = df['existing'] * 2
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
```

### 🎯 Q&A — Module 2

1. **`.iloc` vs `.loc` — when do they differ even if both work?**
   `iloc[0]` is **always** the first row, regardless of how the index is labeled. `loc[0]` returns the row whose **label** is `0` — which may not be the first row after a sort, a filter, or a `set_index`. Use `iloc` for positional logic and `loc` for label logic.

2. **`df['col']` vs `df.col` — any real difference?**
   Same Series most of the time. **But** `df.col` fails if the name has spaces, special chars, or collides with a method (e.g., `df.shape` is the property, not a column named `shape`). Bracket form is the safe default.

3. **What does `inplace=True` actually do, and should I use it?**
   Modifies the DataFrame in-place; returns `None`. Saves memory by skipping the copy. Modern Pandas (≥2.0) is de-emphasizing it because it complicates method-chaining and copy-on-write semantics. Prefer `df = df.rename(...)` unless memory is critical.

4. **What's `SettingWithCopyWarning` and how do you fix it?**
   When you assign to a slice that may be a view: `df[df['x']>0]['y'] = 1`. Pandas can't tell if you meant to modify `df` or a copy. Fix: use `.loc` in one shot — `df.loc[df['x']>0, 'y'] = 1`.

5. **What does `rename(columns={'nonexistent': 'x'})` do?**
   Nothing — it silently no-ops on missing keys. No error. Useful but a footgun if you typo a name and don't realize the rename didn't happen.

[🔝 Back to top](#top)

---

<a id="3-module3"></a>
## 3. Module 3 — Cleanup: dtype, strings, NaN, exploration

> Real-world pandas data is messy. This module is the cleanup toolkit.

### 🪜 Mental model

**Coerce, don't trust.** Every column from a CSV is a string until proven otherwise. Pandas guesses dtype from the first few rows; one stray `"NEW"` is enough to demote `rating` from `float64` to `object`, and now `.mean()` returns NaN. The cleanup pipeline is always: **strip → replace sentinels with NaN → `pd.to_numeric(errors='coerce')` → verify dtype**. *If a numeric op returns NaN unexpectedly, your column is still `object` — every time.*

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

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| `astype(t)` | Strict cast — raises on any unparseable value |
| `pd.to_numeric(s, errors='coerce')` | Safe cast — bad values become NaN |
| `.str` accessor | Vectorized string ops on a Series (`.str.strip()`, `.str[:-2]`, `.str.lower()`) |
| `s.replace(old, new)` | Swap values; useful before `astype` |
| NaN | Floating-point "not a number" sentinel; never equals itself |
| `.unique()` | Distinct values, in order of appearance |
| `.nunique()` | Count of distinct values |
| `.value_counts()` | Frequencies, sorted descending |
| `dropna=False` | Include NaN in `value_counts` |
| `df.sort_values('col', ascending=False)` | Sort rows by a column |
| `df.dropna() / df.fillna(v)` | NaN handling |

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

[🔝 Back to top](#top)

---

<a id="4-terms"></a>
## 4. 📚 Pandas terms glossary

| Term | Definition |
|---|---|
| Coercion | Forcing one type into another (e.g., `errors='coerce'` makes bad values NaN) |
| DataFrame | Pandas 2D labeled table |
| Explicit index | Custom-labeled DataFrame index |
| `iloc` | Position-based row access |
| Implicit index | Default integer 0,1,2,… index |
| `loc` | Label-based row access |
| NaN | Floating-point "not a number" sentinel |
| Object dtype | Any non-numeric column (often strings or mixed) |
| Series | Pandas 1D labeled array |
| `SettingWithCopyWarning` | Ambiguity warning when assigning to a slice that may be a view |
| `value_counts` | Frequency table for a Series |
| `.str` accessor | Namespace for vectorized string ops on a Series |
| Derived column | A new column computed from existing ones |
| `read_csv` | CSV → DataFrame loader with extensive parsing options |
| `to_numeric(coerce)` | Safe numeric cast that turns bad values into NaN |

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
| `df.head(n)` / `df.tail(n)` | First / last n rows |
| `df.iloc[i, j]` | Position-based |
| `df.loc[label, 'col']` | Label-based |
| `df['col']` / `df[['c1','c2']]` | Single / multi column |
| `df.set_index('col')` / `df.reset_index(drop=True)` | Index manip |

### Info & cleanup
| Call | Purpose |
|---|---|
| `df.info()` | Schema + nulls |
| `df.describe(include='all')` | Stats |
| `df.rename(columns={...})` | Rename |
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

[🔝 Back to top](#top)

---

<a id="7-advanced"></a>
## 7. 🎯 Advanced interview Q&A

> Mix of original drills and questions adapted from `guipsamora/pandas_exercises`, `ajcr/100-pandas-puzzles`, `alexeygrigorev/data-science-interviews`, and common StrataScratch / LeetCode patterns.

**Q1. Difference between `df.iloc[0]` and `df.loc[0]`?** *(very common opener)*
`iloc` is **position**-based — first row. `loc` is **label**-based — the row whose index equals `0`. They diverge after `set_index`, sort, or filter.

**Q2. Why might `df['rating'].mean()` return `NaN` even after dropping NaNs?**
Because the column dtype is `object` (strings) — Pandas treats numeric methods differently for non-numeric. Cast first: `pd.to_numeric(errors='coerce')`.

**Q3. What's `SettingWithCopyWarning` and how do you fix it?**
Pandas can't tell if you're modifying the original or a copy. Fix: single-step `.loc` assignment — `df.loc[mask, 'col'] = val`.

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
