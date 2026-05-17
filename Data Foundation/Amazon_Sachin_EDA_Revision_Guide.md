<a id="top"></a>
# Amazon Sales + Sachin Tendulkar EDA — Master Revision Guide

> **Interview-ready revision sheet for the four new notebooks** (Amazon Sales Data Analysis 1/2/3 + Analyzing Sachin Tendulkar's ODI Career). Covers applied pandas, data cleaning, reshaping, datetime, visualization, correlation, and probability — with cheat sheets, advanced Q&A, gotchas, and a 100-question drill.

**Companion guides:**
- 🐍 [`Data_Foundation_Revision_Guide.md`](./Data_Foundation_Revision_Guide.md) — NumPy foundations
- 🐼 [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) — Pandas basics (Series, DataFrame, `iloc`/`loc`, cleanup)

**How to use:**
- **Pre-interview:** scan the [🚀 Topic finder](#topic-finder) → open the relevant module → drill the Q&A.
- **Just before a coding round:** run the [§14 Drill](#14-drill) end-to-end.
- **For per-notebook depth:** see the notebook in each `G -Amazon sales data analysis N/` or `G-Analyzing Sachin Tendulkar's ODI Career/` folder.

**External practice (after you've drilled this guide):**
- 🎯 **StrataScratch** — real pandas/SQL questions from Amazon, Meta, Google, Airbnb (groupby, joins, window functions).
- 🎯 [`guipsamora/pandas_exercises`](https://github.com/guipsamora/pandas_exercises) — graded pandas (merge, groupby, apply, datetime).
- 🎯 [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles) — 100 short pandas puzzles.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — probability, ML, coding.
- 🎯 [`kojino/120-Data-Science-Interview-Questions`](https://github.com/kojino/120-Data-Science-Interview-Questions) — probability + Bayes section is gold.
- 🎯 [`chiphuyen/ml-interviews-book`](https://huyenchip.com/ml-interviews-book/) — probability, ML system design.
- 🎯 **LeetCode → Database / Pandas tracks** — paid but tightly graded.

---

<a id="topic-finder"></a>
## 🚀 Topic finder

| Need to revise… | Go to |
|---|---|
| `iloc`/`loc` for rows + columns, conditional updates | [Module 1](#1-module1) |
| Duplicates (`duplicated`/`drop_duplicates`, `keep`, `subset`) | [Module 1](#1-module1) |
| Aggregate functions (`sum`/`mean`/`agg`), sorting (multi-col) | [Module 1](#1-module1) |
| `pd.concat` vs `pd.merge`, SQL joins (inner/outer/left/right) | [Module 1](#1-module1) |
| `.apply()` column/row/lambda, `axis=1` semantics | [Module 1](#1-module1) |
| GroupBy: split-apply-combine, `.filter()`, `.apply()` | [Module 1](#1-module1) |
| Missing data (`isna`/`dropna`/`fillna`), imputation strategies | [Module 2](#2-module2) |
| Reshape: `pd.melt` (wide→long), `pd.pivot_table` (long→wide) | [Module 2](#2-module2) |
| Binning with `pd.cut`, categorical buckets | [Module 2](#2-module2) |
| String ops: `.str.contains`, `.str.extract`, custom `.apply` | [Module 2](#2-module2) |
| DateTime: `pd.to_datetime`, `.dt` accessor (`week`, `day_name`, `strftime`) | [Module 2](#2-module2) |
| Univariate plots (histplot, countplot) | [Module 2](#2-module2) |
| Bivariate plots (scatter/line/box/violin/grouped bar) | [Module 3](#3-module3) |
| Multivariate plots (pairplot/heatmap/stacked) | [Module 3](#3-module3) |
| Correlation (`.corr`) — and what it misses | [Module 3](#3-module3) |
| `hue`, `subplots`, `figsize`, `alpha`, `palette` | [Module 3](#3-module3) |
| Probability fundamentals, sample space, events | [Module 4](#4-module4) |
| Addition / multiplication / complement rules | [Module 4](#4-module4) |
| Conditional probability and Bayes' Theorem | [Module 4](#4-module4) |
| All terms at once | [§5 Glossary](#5-terms) |
| Every applied API at once | [§6 API cheat sheet](#6-apis) |
| Common gotchas | [§7 Gotchas](#7-gotchas) |
| Advanced interview Q&A | [§8 Advanced](#8-advanced) |
| Visualization decision tree | [§9 Viz decisions](#9-vizdecisions) |
| Dataset cheat sheets | [§10 Datasets](#10-datasets) |
| Business questions → which API | [§11 Business map](#11-business) |
| Probability formulas cheat sheet | [§12 Probability](#12-probability) |
| Cross-module concept map | [§13 Concept map](#13-conceptmap) |
| 🌐 Sourced interview questions (drill bank) | [Sourced bank](#sourced-bank) |
| Speed-run revision before interview | [§14 Drill](#14-drill) |
| Best practices | [§15 Best practices](#15-bestpractices) |
| Notebook mapping | [§16 Mapping](#16-mapping) |

---

## 📑 Table of contents

1. [Module 1 — Joins, GroupBy, Apply (Amazon 1)](#1-module1)
2. [Module 2 — Missing Data, Reshape, Strings, DateTime, Univariate Viz (Amazon 2)](#2-module2)
3. [Module 3 — Bivariate & Multivariate Visualization (Amazon 3)](#3-module3)
4. [Module 4 — Probability via Pandas (Sachin Tendulkar ODI)](#4-module4)
5. [📚 Terms glossary](#5-terms)
6. [⚙️ API cheat sheet](#6-apis)
7. [⚠️ Gotchas & traps](#7-gotchas)
8. [🎯 Advanced interview Q&A](#8-advanced)
9. [📊 Visualization decision tree](#9-vizdecisions)
10. [📦 Dataset cheat sheets](#10-datasets)
11. [Business questions → which API](#11-business)
12. [🎲 Probability formula cheat sheet](#12-probability)
13. [Cross-module concept map](#13-conceptmap)
13B. [🌐 Sourced interview questions](#sourced-bank)
14. [🔁 100-question revision drill](#14-drill)
15. [✅ Best practices](#15-bestpractices)
16. [Notebook mapping](#16-mapping)

---

<a id="1-module1"></a>
## 1. Module 1 — Joins, GroupBy, Apply (Amazon Sales Data Analysis 1)

> **What the notebook covers:** Combined row+column access with `iloc`/`loc`; conditional updates; handling duplicates; aggregate functions; sorting (single + multi-column); concatenation vs merging; SQL-style joins; `.apply()` (column/row/lambda); group-based aggregation, filtering, and apply (the split-apply-combine paradigm).

### 🪜 Mental model

**Split-apply-combine.** GroupBy is three steps disguised as one method: **split** the table into bins by some key, **apply** a function inside each bin independently, **combine** the per-bin results into a Series or DataFrame. Hold that 3-step picture and every confusion about `.agg` vs `.transform` vs `.filter` vs `.apply` resolves to "what does each step return?" — `.agg` collapses each group to one row; `.transform` returns the same shape as input; `.filter` keeps/drops whole groups; `.apply` is the escape hatch when nothing else fits.

For **joins**, the mental model is the SQL Venn diagram: inner = ∩, outer = ∪, left = all of left + matches from right, right = mirror.

### 🪞 Basic → Intermediate → Advanced — `groupby`

**Basic** — split by one key, aggregate one column.
```python
df.groupby('category')['rating'].mean()       # Series indexed by category
```

**Intermediate** — multiple keys + multiple aggregations.
```python
df.groupby(['category', 'product'])['rating_count'].agg(['mean', 'max', 'count'])
# DataFrame with hierarchical row index
```

**Advanced** — `.filter()` keeps rows belonging to groups that pass a predicate; `.apply()` returns whatever your function returns. Both are slower than `.agg()` — reach for them only when `.agg` can't express it.
```python
# Keep only users with > 15 reviews — returns rows, not groups
heavy = df.groupby('user_id').filter(lambda g: len(g) > 15)

# Custom per-group transform returning a Series
df.groupby('category').apply(
    lambda g: pd.Series({'spread': g['price'].max() - g['price'].min()}),
    include_groups=False,
)
```

### 🪞 Basic → Intermediate → Advanced — joins

**Basic** — inner join keeps only matching keys.
```python
pd.merge(orders, products, on='product_id', how='inner')
```

**Intermediate** — left join preserves every row from `orders`, NaN-fills missing product info.
```python
pd.merge(orders, products, on='product_id', how='left')
# len() == len(orders) — your sanity check after every join
```

**Advanced** — `pd.merge` with no `on=` joins on **all** common column names (often wrong). Always pass `on=` explicitly. Duplicate keys produce row explosions: 3 matches × 4 matches → 12 rows.
```python
pd.merge(orders, products)                    # silently joins on every shared column
pd.merge(orders, products, on='product_id', validate='m:1')   # raises if rhs has duplicate keys
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| `df.iloc[i:j, m:n]` | Position-based rectangular slice |
| `df.loc[mask, 'col'] = v` | Single-step conditional update (avoids `SettingWithCopyWarning`) |
| `df.duplicated(keep='first'/'last'/False)` | Boolean mask of duplicate rows |
| `df.drop_duplicates(subset=[...], keep='first')` | Remove duplicates by some/all columns |
| `df['col'].agg([...])` | Multiple aggregations in one call |
| `df.sort_values(by, ascending)` | Sort by 1+ columns; `ascending` can be a list |
| `pd.concat([a, b], axis=0/1)` | Stack rows or columns — does NOT align on a key |
| `pd.merge(a, b, on='id', how='inner')` | Align on a key — SQL-style join |
| Join types | inner (∩), outer (∪), left (all left), right (all right) |
| `df['col'].apply(fn)` | Element-wise function on a Series |
| `df.apply(fn, axis=1)` | Function applied to each ROW (row passed as Series) |
| `df.apply(fn, axis=0)` | Function applied to each COLUMN (default) |
| `df.isnull().sum()` | Per-column NaN audit |
| `df.groupby('k')` | Lazy: defines splits; aggregate to collapse |
| `df.groupby('k')['c'].agg(['max','min'])` | Multiple aggs per group |
| `df.groupby('k').filter(fn)` | Keep ROWS belonging to groups that pass `fn` |
| `df.groupby('k').apply(fn)` | Custom per-group transform / aggregation |

### ⚙️ Top APIs

```python
# Selection / updates
df.iloc[i:j, m:n] = [[...], [...]]
df.loc[mask, 'col'] = val
df.loc[df['rating'] < 3, 'flag'] = -1

# Duplicates
df.duplicated(keep='first' | 'last' | False)
df.drop_duplicates(subset=['name','votes'], keep='first', inplace=False)

# Aggregations
df['col'].sum() / mean() / count() / min() / max()
df[['c1','c2']].agg(['sum','mean','min','max'])

# Sort
df.sort_values(by='votes', ascending=False)
df.sort_values(by=['votes','rating'], ascending=[False, True])

# Combine — no key
pd.concat([df1, df2], axis=0, ignore_index=True)   # row-wise
pd.concat([df1, df2], axis=1)                       # column-wise

# Combine — by key
pd.merge(orders, products, on='product_id', how='inner')   # only matching
pd.merge(orders, products, on='product_id', how='outer')   # all rows, NaN fills
pd.merge(orders, products, on='product_id', how='left')    # keep all orders
pd.merge(orders, products, on='product_id', how='right')   # keep all products

# Apply
df['x'] = df['col'].apply(lambda v: v.upper())          # element-wise
df['disc'] = df.apply(lambda row: (row.actual - row.disc) / row.actual, axis=1)  # row-wise
products[['rating','rating_count']].apply([max, min])   # column-wise aggs

# Missing
df.isnull().sum()
df.isna().sum().sort_values(ascending=False)

# GroupBy
df.groupby('category')['rating'].mean()
df.groupby(['category','product_name'])['rating_count'].agg(['max','min'])
df.groupby('user_id').filter(lambda g: len(g) > 15)
df.groupby('category').apply(lambda g: g['actual_price'].mean() - g['disc_price'].mean())
```

### 🧩 Code patterns

```python
# 1. Conditional flag — one-shot .loc assignment
df['flag'] = 1
df.loc[df['rating'] < 3, 'flag'] = -1

# 2. Append rows + reset index
new_row = pd.DataFrame([["R7", 100, 3.7]], columns=df.columns)
df = pd.concat([df, new_row], ignore_index=True)

# 3. Drop duplicates by a subset
df = df.drop_duplicates(subset=['name', 'votes'], keep='first')

# 4. Multi-column sort with mixed ascending order
df.sort_values(by=['votes', 'rating'], ascending=[False, True])

# 5. Left join — keep ALL orders, add product info where it matches
df = pd.merge(orders, products, on='product_id', how='left')

# 6. Clean price strings via apply
def extract_price(s):
    try:    return float(str(s).replace('₹', '').replace(',', ''))
    except: return None

df['actual_price'] = df['actual_price'].apply(extract_price)

# 7. Row-wise apply (needs axis=1)
df['discount_pct'] = df.apply(
    lambda r: (r['actual_price'] - r['discounted_price']) / r['actual_price'] * 100,
    axis=1,
)

# 8. Group-based filter — KEEP rows whose group satisfies a condition
heavy_reviewers = df.groupby('user_id').filter(lambda g: g['review_id'].count() > 15)

# 9. Group-based custom apply returning a Series
def stats(g):
    return pd.Series({
        'mean_rating':         g['rating'].mean(),
        'min_discounted_price': g['discounted_price'].min(),
        'max_discounted_price': g['discounted_price'].max(),
    })

df.groupby('category').apply(stats, include_groups=False)
```

### 🎯 Q&A — Module 1

> Mix of original drills and questions adapted from `guipsamora/pandas_exercises`, `alexeygrigorev/data-science-interviews`, and common StrataScratch patterns.

1. **`pd.concat` vs `pd.merge` — when which?** *(common StrataScratch opener)*
   `concat` stacks DataFrames along an axis with **no key alignment** — same shape implied. `merge` joins by **shared key(s)** with SQL semantics. Rule of thumb: appending more rows of the same schema → `concat`. Joining two tables that share an `id` → `merge`.

2. **Inner / outer / left / right join — predict result counts.**
   - Inner: rows where the key matches in **both** → smallest result.
   - Outer: union of keys, missing fills with NaN → largest result.
   - Left: all rows from left + matches from right → size of left.
   - Right: all rows from right + matches from left → size of right.

3. **Why is `.apply()` "less efficient than vectorized ops"?**
   `.apply()` typically runs a Python-level loop, breaking the C-vectorized path. Where possible, use array math, `np.where`, `.str.*`, or built-in agg/groupby. Reserve `.apply()` for genuinely custom logic.

4. **What's the most common `.apply()` confusion?**
   `axis=1` means "function applied **per row**, with the entire row passed as a Series." Students often expect `axis=1` to be vectorized — it isn't. It's one Python call per row.

5. **GroupBy split-apply-combine — explain in one breath.**
   **Split** rows into groups by key(s), **apply** an operation (aggregate / transform / filter) to each group, **combine** the results back into a Series or DataFrame. The result type depends on what you applied.

6. **`df.groupby('k')['x'].mean()` returns Series or DataFrame?**
   **Series**, indexed by `k`. If you select multiple columns (`df.groupby('k')[['x','y']].mean()`), you get a DataFrame.

7. **`groupby().filter(fn)` returns what?**
   A **DataFrame**, not a GroupBy object. It keeps the **rows** belonging to groups whose function returns `True`. So you're filtering **groups**, not rows — but you receive rows back.

8. **`drop_duplicates(keep=False)` — what does it do?**
   Drops **all** rows that have any duplicate (no occurrences kept). Useful to find "uniquely unique" rows. Compare with `keep='first'` (one survivor per duplicate group).

9. **`df.drop_duplicates(subset=['x','y'])` — does it modify in-place?**
   No — it returns a new DataFrame. Pass `inplace=True` (or re-assign) to actually mutate `df`.

10. **`ascending=[True, False]` on a single-column sort — what happens?**
    `ValueError` — length mismatch. `ascending` must be a scalar bool when `by` is a single column, or a same-length list when `by` is a list.

[🔝 Back to top](#top)

---

<a id="2-module2"></a>
## 2. Module 2 — Missing Data, Reshape, Strings, DateTime, Univariate Viz (Amazon Sales Data Analysis 2)

> **What the notebook covers:** Identifying and handling missing values; wide↔long reshaping with `melt`/`pivot_table`; binning continuous data; vectorized string ops; datetime parsing and extraction; univariate plotting with Matplotlib / Seaborn.

### 🪜 Mental model

**Wide is human-readable; long is computer-readable.** Wide tables (one column per metric) are great for spreadsheets and human eyes. Long tables (one row per observation, with a "kind" column naming the metric) are great for plotting, grouping, and aggregating. `pd.melt` goes wide → long; `pd.pivot_table` goes long → wide. *Almost every plotting/grouping bug is "I have wide data but my function expects long" or vice versa.*

For **missing data**, the mental model is **audit before imputing**: `df.isna().sum()` is always the first call. Then decide *per column* — drop if rare and meaningless; impute with mean/median/mode if recoverable; impute with a sentinel (0 for "discount", "Unknown" for category) when missing has business meaning.

### 🪞 Basic → Intermediate → Advanced — missing data

**Basic** — count NaN per column.
```python
df.isna().sum()
```

**Intermediate** — drop or impute per column with a sensible default.
```python
df['rating']  = df['rating'].fillna(df['rating'].mean())     # numeric → mean
df['cat']     = df['cat'].fillna('Unknown')                  # categorical → sentinel
df['discount'] = df['discount'].fillna(0)                    # domain default
df = df.dropna(subset=['order_ts'])                          # drop where critical field missing
```

**Advanced** — `fillna(s.mean())` is a no-op when the whole column is NaN (`mean` of all-NaN is NaN). And blindly imputing leaks information from test into train. Production rule: fit the imputer on train, apply to test.
```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean').fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp  = imp.transform(X_test)         # uses train's mean, not test's
```

### 🪞 Basic → Intermediate → Advanced — reshape

**Basic** — melt collapses many value columns into one.
```python
pd.melt(df, id_vars=['product_id'],
            value_vars=['actual_price', 'discounted_price'],
            var_name='kind', value_name='price')
```

**Intermediate** — `pivot_table` reverses melt and aggregates duplicates.
```python
pd.pivot_table(df, index='category', columns='month',
                values='rating', aggfunc='mean', fill_value=0)
```

**Advanced** — `pivot()` is strict and errors on duplicate `(index, columns)` pairs; `pivot_table()` aggregates them silently (default `mean`). On real data, prefer `pivot_table` and pass `aggfunc=` explicitly so the choice is documented.
```python
df.pivot(index='id', columns='month', values='rating')      # ValueError if duplicates
df.pivot_table(index='id', columns='month', values='rating',
               aggfunc='first')                              # documents intent
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| `None` vs `NaN` | `None` is Python; `NaN` is `numpy.float64`. Pandas auto-converts most `None` to `NaN` |
| `isna()` / `isnull()` | Identical methods — boolean mask of missing values |
| `dropna(how='any'/'all', subset=[...])` | Remove rows/cols with NaN |
| `fillna(value)` / `fillna(method='ffill')` | Impute — constant, mean, forward-fill, back-fill |
| `pd.melt` | Wide → long. Multiple value columns collapse into rows |
| `pd.pivot_table` | Long → wide with aggregation. `aggfunc='mean'` resolves duplicates |
| `pivot` vs `pivot_table` | `pivot` errors on duplicate index/column pairs; `pivot_table` aggregates |
| `pd.cut` | Binning continuous → categorical via edges + labels |
| `pd.qcut` | Quantile binning (equal-count buckets) |
| `.str.contains('foo', case=False, na=False)` | NaN-safe substring search |
| `.str.extract(r'(pattern)')` | Pull regex matches into columns |
| `pd.to_datetime` | Parse strings to datetimes; pass `format=` for speed/clarity |
| `.dt` accessor | `.dt.year`, `.dt.month`, `.dt.day_name()`, `.dt.isocalendar().week`, `.dt.strftime('%B %Y')` |
| Univariate plot | Distribution / count of ONE variable |
| `sns.histplot(kde=True)` | Histogram with optional KDE overlay |
| `sns.countplot(x='col')` | Categorical frequency bar plot |

### ⚙️ Top APIs

```python
# Missing data
df.isna().sum().sort_values(ascending=False)
df.dropna(how='any', subset=['rating'])
df.fillna({'rating': df['rating'].mean(), 'discount': 0})

# Reshape
melted = pd.melt(
    df,
    id_vars=['product_id','category'],
    value_vars=['actual_price','discounted_price'],
    var_name='price_type', value_name='price',
)
pivot = pd.pivot_table(
    df, index='category', columns='product_name',
    values='rating', aggfunc='mean',
)

# Binning
df['price_band'] = pd.cut(
    df['actual_price'],
    bins=[0, 5000, 20000, 50000, 200000],
    labels=['Low','Medium','High','Very High'],
)
df['rating_band'] = pd.qcut(df['rating'], q=4)         # equal-count buckets

# String ops
df['has_durable'] = df['about_product'].str.contains('durable', case=False, na=False)
df['qid'] = df['product_link'].str.extract(r'qid=(\d+)')

# DateTime
df['ts'] = pd.to_datetime(df['order_timestamp'])
df['week'] = df['ts'].dt.isocalendar().week
df['day']  = df['ts'].dt.day_name()
df['mo']   = df['ts'].dt.strftime('%B %Y')

# Univariate viz
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='rating', bins=20, kde=True, color='skyblue')
plt.axvline(df['rating'].mean(), color='red', linestyle='--', label=f"Mean: {df['rating'].mean():.2f}")
plt.legend()
plt.title("Rating Distribution")

sns.countplot(data=df, x='category', palette='coolwarm')
plt.xticks(rotation=45)
```

### 🧩 Code patterns

```python
# 1. NaN audit + targeted imputation
df.isnull().sum()
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['discount'] = df['discount'].fillna(0)                    # domain default

# 2. Wide → long for plotting two metrics
melted = pd.melt(df, id_vars=['product_id'], value_vars=['actual_price','discounted_price'],
                 var_name='kind', value_name='price')

# 3. Long → wide with aggregation
pivot = pd.pivot_table(df, index='category', columns='month', values='rating', aggfunc='mean')

# 4. Bucket prices into segments
df['segment'] = pd.cut(df['actual_price'],
                       bins=[0, 5000, 20000, 50000, np.inf],
                       labels=['Low','Mid','High','VeryHigh'])

# 5. Substring filter (NaN-safe)
durable = df[df['about_product'].str.contains('durable', case=False, na=False)]

# 6. Datetime to month-year label, then sort by underlying date
df['mo_label'] = df['ts'].dt.strftime('%B %Y')
trend = df.groupby('mo_label')['rating'].mean().reset_index()
trend['_sort'] = pd.to_datetime(trend['mo_label'], format='%B %Y')
trend = trend.sort_values('_sort').drop(columns='_sort')

# 7. Distribution with mean line
sns.histplot(df['rating'], bins=20, kde=True)
plt.axvline(df['rating'].mean(), color='red', ls='--')
```

### 🎯 Q&A — Module 2

> Mix of original drills and questions adapted from `ajcr/100-pandas-puzzles`, `guipsamora/pandas_exercises`, and StrataScratch.

1. **`isna()` vs `isnull()` — different?** *(quick-fire opener)*
   No — `isnull` is an alias for `isna`. They return identical boolean masks. Pandas docs treat them as the same method; use whichever reads best.

2. **`pivot()` keeps failing — when do you switch to `pivot_table()`?**
   When index/column pairs aren't unique. `pivot()` is strict and raises; `pivot_table()` aggregates duplicates (default `aggfunc='mean'`). For real data, prefer `pivot_table`.

3. **`pd.melt` — what do `id_vars` and `value_vars` mean?**
   `id_vars` = columns kept as-is (identifiers that label each row). `value_vars` = columns that get **stacked** into one column of values plus one column naming which variable each value came from.

4. **`pd.cut` vs `pd.qcut`?**
   `cut` bins by **value range** — you pass explicit edges. `qcut` bins by **quantile** so every bucket has the same count. Use `cut` for "low/mid/high" labels; `qcut` for "quartile/decile" analysis.

5. **`.str.upper()` on a Series with NaN — error?**
   No. The `.str` accessor preserves NaN and continues. This is one of the rare safe defaults in Pandas string handling.

6. **Why does my CSV load IDs `"00123"` as `123`?**
   Pandas inferred the column as int and dropped leading zeros silently. Fix at load time: `pd.read_csv(path, dtype={'id': str})`. After load, you can't recover — the zeros are gone.

7. **`pd.to_datetime` is slow on a big column — what's the speed-up?**
   Pass `format=` explicitly. Without it, Pandas tries multiple parsers per value. With `format='%Y-%m-%d'` it goes straight to the fast path. For really huge data, parse on load via `parse_dates=['col']`.

8. **`.dt.isocalendar().week` vs `.dt.week`?**
   `.dt.week` is deprecated. Use `.dt.isocalendar().week` for the ISO 8601 week number. Heads-up: ISO weeks start Monday; a Sunday in early January can belong to week 52 of the previous year.

9. **A histogram has 100 bins — but 95% of mass is in two — what do you do?**
   Either reduce bin count (`bins=20`), switch to `pd.qcut` for equal-count buckets, or log-transform the variable. The plot is telling you the data is skewed.

10. **When do you use `countplot` vs `histplot`?**
    `countplot` is for **categorical** variables — counts unique values. `histplot` is for **continuous** variables — bins them. Don't `histplot` a categorical or you'll get a confusing bar chart.

[🔝 Back to top](#top)

---

<a id="3-module3"></a>
## 3. Module 3 — Bivariate & Multivariate Visualization (Amazon Sales Data Analysis 3)

> **What the notebook covers:** Two-variable plots (scatter, line, box, violin, grouped bar); three-or-more variable plots (pair plot, heatmap, stacked bar); correlation analysis with `.corr()`; styling with `hue`, `palette`, `alpha`, `subplots`.

### 🪜 Mental model

**Choose by question, not by chart name.** Don't ask "should I use a boxplot?" Ask "*what am I trying to see?*"
- *Shape of one variable* → histogram / KDE
- *Frequency of one category* → countplot
- *Relationship between two numerics* → scatter
- *Distribution of a numeric per group* → boxplot or violin
- *Trend over time* → line plot
- *Pairwise correlations* → heatmap
- *Composition* → stacked bar

The decision tree in [§9](#9-vizdecisions) is the canonical version. Once you internalize this, picking a chart is a 5-second decision.

**Encoding extra dimensions:** `hue` adds a categorical 3rd dimension; `size` adds a numeric 3rd; `subplots` adds an unlimited categorical 4th. Don't try to encode more than 4 — readers can't decode it.

### 🪞 Basic → Intermediate → Advanced — correlation

**Basic** — Pearson correlation in [-1, 1] for two numeric arrays.
```python
df[['price', 'rating']].corr()       # 2×2 matrix
```

**Intermediate** — full pairwise matrix visualized as a heatmap.
```python
sns.heatmap(df[['price', 'rating', 'count']].corr(),
            annot=True, fmt='.2f', cmap='coolwarm', center=0)
```

**Advanced** — Pearson `r = 0` does **NOT** mean independent — only that there's no *linear* relationship. A perfect quadratic over a symmetric range gives `r = 0`. Use **Spearman** for monotonic patterns or visualize with a scatter when in doubt.
```python
df[['x', 'y']].corr(method='spearman')   # ranks-based; catches monotonic non-linear
```

### 🪞 Basic → Intermediate → Advanced — scatter

**Basic** — two numeric columns, one point per row.
```python
sns.scatterplot(data=df, x='price', y='rating')
```

**Intermediate** — overplotting? Use `alpha`. Adding a 3rd dim? Use `hue`.
```python
sns.scatterplot(data=df, x='price', y='rating', hue='category', alpha=0.4)
```

**Advanced** — when point density is the message (not individual points), switch to a 2D KDE or hexbin. Useful at millions of rows.
```python
sns.kdeplot(data=df, x='price', y='rating', fill=True)
plt.hexbin(df['price'], df['rating'], gridsize=50, cmap='Blues')
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Univariate | One variable — distribution / count |
| Bivariate | Two variables — relationship |
| Multivariate | 3+ variables — typically encoded with hue, size, or subplots |
| `sns.scatterplot(x, y, hue, alpha)` | Two continuous vars; `hue` adds a 3rd categorical dim |
| `sns.lineplot(x, y, marker)` | Trends over a sequence (time, ordinal x) |
| `sns.boxplot(x, y)` | Quartiles, median, whiskers, outliers per group |
| `sns.violinplot(x, y)` | Box + KDE — shows density shape |
| `sns.barplot(x, y, hue, estimator=np.mean)` | Aggregated bar (default = mean + 95% CI) |
| `sns.pairplot(df, hue, diag_kind)` | Grid of pairwise scatter + diag distributions |
| `sns.heatmap(corr, annot, fmt, cmap)` | Color-encoded matrix; great for correlation |
| `df.corr(method='pearson')` | Linear correlation; `'spearman'` for monotonic |
| Pearson r = 0 means | NO **linear** relation — non-linear may still exist |
| Stacked bar | `df.groupby([a, b]).size().unstack().plot(kind='bar', stacked=True)` |
| `plt.subplots(rows, cols)` | Multi-axes figure; assign with `ax=axes[i]` |
| `figsize=(w, h)` | Inches |
| `alpha` | Transparency 0–1 — useful for overplotting in scatter |
| `hue` | Color encoding by a categorical variable |
| `palette` | Color scheme — `'coolwarm'`, `'Spectral'`, `'Blues'` |

### ⚙️ Top APIs

```python
# Univariate recap
sns.countplot(data=df, x='category')
sns.kdeplot(data=df, x='rating', shade=True)
sns.histplot(data=df, x='rating', bins=20, kde=True)

# Bivariate
sns.scatterplot(data=df, x='discounted_price', y='rating', alpha=0.5, hue='category')
sns.lineplot(data=monthly, x='month', y='rating', marker='o')
sns.boxplot(data=df, x='category', y='rating')
sns.violinplot(data=df, x='category', y='rating')
sns.barplot(data=df, x='category', y='rating', hue='has_durable')

# Multivariate
sns.pairplot(df[['rating','price','count']].dropna(), diag_kind='kde')
corr = df[['rating','price','count','discount']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')

# Stacked composition
df.groupby(['price_band','rating_band']).size().unstack().plot(
    kind='bar', stacked=True, figsize=(10, 6), colormap='Blues',
)

# Layout
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=df, x='price', y='rating', hue='band', ax=axes[0])
sns.scatterplot(data=df, x='price', y='rating', hue='durable', ax=axes[1])
plt.tight_layout()
plt.show()
```

### 🧩 Code patterns

```python
# 1. Scatter with hue + transparency (handles overplotting)
sns.scatterplot(data=df, x='price', y='rating', hue='category', alpha=0.5)

# 2. Box plot per category — quick distribution comparison
sns.boxplot(data=df, x='category', y='rating')
plt.xticks(rotation=45)

# 3. Correlation matrix → heatmap
corr = df[['rating','price','count','discount']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)

# 4. Stacked bar — composition of B inside A
df.groupby(['price_band', 'rating_band']).size().unstack(fill_value=0)\
  .plot(kind='bar', stacked=True, colormap='viridis')

# 5. Pair plot — fast first-look at relationships
sns.pairplot(df[['price','rating','discount']].dropna(),
             diag_kind='kde', corner=True, plot_kws={'alpha': 0.5})

# 6. Side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='category', y='rating', ax=axes[0])
sns.violinplot(data=df, x='category', y='rating', ax=axes[1])
plt.tight_layout()

# 7. Line plot with sorted month-year (real datetime under the hood)
trend = df.groupby('mo_label')['rating'].mean().reset_index()
trend['_sort'] = pd.to_datetime(trend['mo_label'], format='%B %Y')
trend = trend.sort_values('_sort')
sns.lineplot(data=trend, x='mo_label', y='rating', marker='o')
plt.xticks(rotation=45)
```

### 🎯 Q&A — Module 3

> Mix of original drills and questions adapted from `alexeygrigorev/data-science-interviews`, `chiphuyen/ml-interviews-book`, and `kojino/120-Data-Science-Interview-Questions`.

1. **Pearson `r = 0` — are the variables independent?** *(classic stats trap — `kojino/120-Data-Science-Interview-Questions` Q-set)*
   **No.** It means no **linear** relationship. They could still have a strong non-linear relationship (parabolic, sinusoidal, etc.). Independence is a probabilistic property that's strictly stronger than zero correlation.

2. **Why use `alpha` on a scatter plot?**
   Transparency. When points overlap, identical positions over-plot into one black blob and you lose density information. `alpha=0.3–0.5` lets you see where points are stacked.

3. **`boxplot` vs `violinplot` — when which?**
   `boxplot` summarizes the 5-number summary (min/Q1/median/Q3/max) plus outliers. `violinplot` adds a kernel-density estimate so you can see the **shape** of the distribution (bimodal? skewed?). Use violin when shape matters; box for quick comparison.

4. **`sns.pairplot` for 30 columns — good idea?**
   No — it'll make 900 subplots and crawl. Pair plot is for ~5–8 numeric columns. For bigger datasets, sample columns or use a heatmap of the correlation matrix instead.

5. **`df.corr()` skips a column — why?**
   It computes pairwise correlation only on **numeric** columns. `object`/string columns are silently dropped. Fix dtype with `pd.to_numeric(errors='coerce')` if you expected to see it.

6. **`hue` adds a categorical dimension but the legend has 200 entries — what now?**
   Your hue variable is too granular. Bucket it (`pd.cut`), pick top-N categories and lump the rest into "Other", or switch to subplots per category instead.

7. **Stacked bar shows totals — but I want proportions per group. How?**
   Normalize before plotting:
   ```python
   cross = df.groupby(['a','b']).size().unstack()
   (cross.div(cross.sum(axis=1), axis=0) * 100).plot(kind='bar', stacked=True)
   ```

8. **Why use `plt.tight_layout()`?**
   It rebalances padding so axis labels, titles, and legends don't overlap each other or the figure edge. Call it after the last plot, before `show`.

9. **`heatmap(corr, annot=True)` — what does `fmt='.2f'` do?**
   Controls the number-formatting string for the annotations. `.2f` shows 2 decimal places. Without it, you'll see noisy decimals like `0.234567`.

10. **Correlation says rating is uncorrelated with price (r ≈ -0.15). Conclusion?**
    Weak negative linear relationship — close to nothing. Don't conclude "price doesn't matter"; check non-linear patterns with a scatter or boxplot per price band, and check sample size.

[🔝 Back to top](#top)

---

<a id="4-module4"></a>
## 4. Module 4 — Probability via Pandas (Analyzing Sachin Tendulkar's ODI Career)

> **What the notebook covers:** Probability fundamentals from a refresher angle (since the curriculum is shifting toward GenAI/Agentic AI). Sample space, events, set operations, addition/multiplication/complement rules, marginal vs joint, conditional probability, and Bayes' Theorem — all computed empirically on Sachin's 360-match ODI dataset using Pandas filtering.

### 🪜 Mental model

**Two mental models. Use both.**

1. **Bayes = update belief.** Start with a prior P(A). Observe evidence B. Multiply the prior by the likelihood P(B|A), normalize by P(B). You end up with the posterior P(A|B). *Probabilities are beliefs being updated by data, not facts about the world.*

2. **2×2 contingency table.** When marginal / joint / conditional get tangled, draw the table:
   ```
                  B=true   B=false   total
       A=true       a        b       a+b      ← P(A)   = (a+b)/n
       A=false      c        d       c+d
       total       a+c      b+d       n       ← P(B)   = (a+c)/n

       P(A ∩ B) = a/n             P(A | B) = a/(a+c)
   ```
   Every probability question reduces to "which cell or row/column total am I dividing?" The numerator is always **some count**; the denominator is always **the conditioning set size**.

### 🪞 Basic → Intermediate → Advanced — conditional probability via pandas

**Basic** — what fraction of matches satisfy a condition? That's a marginal.
```python
P_50plus = (df['runs'] > 50).mean()
```

**Intermediate** — what fraction of *winning* matches had >50 runs? That's conditional — denominator changes.
```python
P_50plus_given_win = df.loc[df['Won'], 'runs'].gt(50).mean()
# equivalent to:  len(df[(df['runs']>50) & (df['Won'])]) / len(df[df['Won']])
```

**Advanced** — Bayes' theorem assembled from empirical pieces. Use this skeleton whenever the question is "posterior given evidence."
```python
P_A         = df['A'].mean()
P_B_given_A = df.loc[df['A'], 'B'].mean()
P_B         = df['B'].mean()
P_A_given_B = P_B_given_A * P_A / P_B
```

### 🪞 Basic → Intermediate → Advanced — addition rule

**Basic** — for mutually exclusive events, P(A ∪ B) = P(A) + P(B).
```python
P_lose_or_tie = (df['Won'] == False).mean() + (df['result'] == 'tie').mean()
```

**Intermediate** — when events can co-occur, subtract the overlap.
```python
P_A_or_B = P_A + P_B - P_A_and_B           # addition rule
```

**Advanced** — `concat([A_df, B_df]).drop_duplicates()` only computes the union correctly when rows are uniquely identifiable. If the source already has duplicates, `drop_duplicates` over-collapses and you'll undercount. Use the addition rule directly or join on a known unique key.
```python
either = pd.concat([df_A, df_B]).drop_duplicates(subset='match_id')
P_union = len(either) / total
```

### 🧠 Concept cheat sheet

| Concept | One-liner |
|---|---|
| Sample space (S) | Set of all possible outcomes |
| Outcome | A single result in S |
| Event | A subset of S (e.g., "Sachin scores > 50") |
| Mutually exclusive | Cannot co-occur: P(A ∩ B) = 0 |
| Independent | P(A ∩ B) = P(A) · P(B) |
| Exhaustive | P(A₁ ∪ A₂ ∪ … ∪ Aₙ) = 1 |
| Marginal P(A) | Probability of A regardless of other events |
| Joint P(A ∩ B) | Both A and B occur |
| Conditional P(A \| B) | A given B has occurred: P(A ∩ B) / P(B) |
| Complement P(A′) | 1 − P(A) |
| Addition rule | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) |
| Multiplication rule | P(A ∩ B) = P(A) · P(B \| A) |
| Bayes' Theorem | P(A \| B) = P(B \| A) · P(A) / P(B) |
| Empirical probability | Count of favorable outcomes ÷ total in your data |

### ⚙️ Top APIs (Pandas idioms for probability)

```python
total = len(df)

# Marginal P(A)
event_A = df[df['runs'] > 50]
P_A = len(event_A) / total

# Joint P(A ∩ B) — intersection via merge or compound mask
event_AB = df[(df['runs'] > 50) & (df['Won'] == True)]
P_AB = len(event_AB) / total

# Union P(A ∪ B) — concat + drop_duplicates OR addition rule
event_AorB = pd.concat([event_A, event_B]).drop_duplicates()
P_AorB    = len(event_AorB) / total
# Or: P_AorB = P_A + P_B - P_AB                                     # addition rule

# Conditional P(A | B)
P_A_given_B = len(df[(df['runs'] > 50) & (df['Won'] == True)]) / len(df[df['Won'] == True])

# Complement P(A')
P_not_A = 1 - P_A

# Sample space
df['runs'].unique()           # distinct outcomes
df['runs'].nunique()          # count of distinct outcomes
```

### 🧩 Code patterns

```python
# 1. Empirical P(event) pipeline
total = len(sachin)
event = sachin[sachin['runs'] > 50]
P = len(event) / total

# 2. Compound condition — always parenthesize, use & / |
hi_score_six = sachin[(sachin['runs'] > 50) & (sachin['sixes'] > 0)]

# 3. Set intersection via merge — equivalent to compound mask
both = pd.merge(event_A, event_B, how='inner')
P_intersection = len(both) / total

# 4. Set union via concat + drop_duplicates
either = pd.concat([event_A, event_B]).drop_duplicates()
P_union = len(either) / total

# 5. Conditional probability (definition)
P_century_given_50plus = (
    len(sachin[(sachin['runs'] > 50) & (sachin['century'] == True)])
    / len(sachin[sachin['runs'] > 50])
)

# 6. Bayes' theorem from empirical pieces
P_A      = len(df[df['A']]) / len(df)
P_B_givA = len(df[df['A'] & df['B']]) / len(df[df['A']])
P_B      = len(df[df['B']]) / len(df)
P_A_givB = (P_B_givA * P_A) / P_B
```

### 🎯 Q&A — Module 4

> Mix of original drills and questions adapted from `kojino/120-Data-Science-Interview-Questions` (probability section), `alexeygrigorev/data-science-interviews`, and `chiphuyen/ml-interviews-book` (probability chapter).

1. **Mutually exclusive vs independent — same thing?** *(`kojino` Q1 — classic opener)*
   **No** — opposite-ish. Mutually exclusive ⇒ they **cannot** co-occur (P(A ∩ B) = 0). Independent ⇒ knowing one tells you nothing about the other (P(A ∩ B) = P(A) · P(B)). Mutually exclusive events with positive probability are NEVER independent.

2. **Addition rule — why subtract P(A ∩ B)?**
   Because the rows in A ∩ B got counted twice (once in A, once in B). Subtracting the intersection corrects the double-count. Skip the subtraction only when A and B are mutually exclusive (P(A ∩ B) = 0).

3. **Conditional vs joint — when each?**
   Joint = "what fraction of all matches satisfy BOTH?" Conditional = "given B happened, what fraction satisfy A?" The denominator changes: total vs |B|.

4. **Why does `len(pd.concat([A, B]).drop_duplicates())` give P(A ∪ B) only sometimes?**
   Only when the rows are uniquely identifiable. If your DataFrame has true duplicate rows already, `drop_duplicates` will over-collapse and you'll undercount the union. Either use a unique key (e.g., match_id) before the union, or use the addition rule directly.

5. **Bayes in one English sentence?**
   Update your prior belief P(A) using the likelihood P(B|A) and the evidence P(B): P(A|B) = P(B|A) · P(A) / P(B). The flow is "update belief in A after observing B."

6. **Why is empirical probability ≠ theoretical probability?**
   Empirical = frequency in observed data. Theoretical = exact long-run probability. They converge by the law of large numbers — but small samples (360 matches) leave noise. Always state your sample size when reporting an empirical P.

7. **Without replacement vs with replacement — why does it matter for P(2nd red | 1st red)?**
   With replacement: probabilities don't change between draws → independent. Without replacement: the population shrinks, so the conditional probability differs from the marginal. Common interview trap.

8. **A factory has 60% output from skilled workers (2% defect) and 40% from unskilled (10% defect). A defective item is found — P(it came from unskilled)?**
   P(D) = 0.6·0.02 + 0.4·0.1 = 0.052. P(U|D) = (0.4·0.1)/0.052 ≈ 76.9%. Classic Bayesian reweighting — the rare-but-defective group dominates the posterior.

9. **Why does the notebook use `pd.merge(..., how='inner')` for intersection but `&` works too?**
   For two filtered DataFrames derived from the **same** rows, `&` on the original is faster and cleaner. `merge` is general — it works even when A and B come from different sources joined by a key. Use `&` for "same-table conditions" and `merge` for "two-table intersections."

10. **Sample size question: 46 centuries in 360 matches — what's a 95% CI on P(century)?**
    Roughly `p ± 1.96·√(p(1-p)/n)` = `0.128 ± 0.0344` → about (0.094, 0.162). Showing this on the spot signals you understand uncertainty, not just point estimates.

[🔝 Back to top](#top)

---

<a id="5-terms"></a>
## 5. 📚 Terms glossary

| Term | Definition |
|---|---|
| `agg` | Apply one or more aggregation functions to columns/groups |
| Aggregation | Reducing rows to a summary statistic (sum, mean, count, …) |
| Apply | Run a function per-row, per-column, or per-group |
| Bayes' Theorem | P(A\|B) = P(B\|A)·P(A) / P(B) |
| Binning | Group continuous values into discrete intervals |
| `concat` | Stack DataFrames along an axis without aligning on a key |
| Conditional probability | P(A\|B) = P(A ∩ B) / P(B) |
| Correlation (Pearson r) | Linear association in [-1, 1]; 0 means no LINEAR association |
| `cut` | Bin by value range — explicit edges |
| Datetime accessor `.dt` | Namespace for extracting components of datetime Series |
| Drop duplicates | Remove repeated rows (full or subset) |
| Event | Subset of the sample space |
| Filter (group) | Keep rows whose group passes a predicate |
| `fillna` | Replace NaN with constant / mean / forward-fill etc. |
| GroupBy | Split rows into groups by key(s) for split-apply-combine |
| Heatmap | Color-encoded matrix |
| Hue | Categorical color encoding in Seaborn plots |
| Inner join | Keep only matched keys |
| `isna` / `isnull` | Identical: boolean mask of NaN |
| Joint probability | P(A ∩ B) |
| KDE | Kernel Density Estimate — smooth distribution curve |
| Left join | All rows from the left + matches from the right |
| Marginal probability | P(A) regardless of other events |
| Melt | Wide → long: stack value columns into rows |
| Merge | Join on a key — SQL-style |
| Mutually exclusive | P(A ∩ B) = 0 |
| Outer join | Union of keys, NaN where missing |
| Pair plot | Grid of pairwise scatter + diagonal distributions |
| Pivot | Long → wide. `pivot` strict; `pivot_table` aggregates duplicates |
| `qcut` | Bin by quantile — equal-count buckets |
| Right join | All rows from the right + matches from the left |
| Sample space | Set of all possible outcomes |
| Split-apply-combine | The GroupBy pattern |
| Stacked plot | Bars or areas where the segment heights are summed |
| `.str` accessor | Vectorized string operations on a Series |
| Univariate / Bivariate / Multivariate | 1 / 2 / 3+ variables |
| Violin plot | Box plot + KDE — shows distribution shape |

[🔝 Back to top](#top)

---

<a id="6-apis"></a>
## 6. ⚙️ API cheat sheet — every applied method, one table

### Selection & updates
| Call | Purpose |
|---|---|
| `df.iloc[i:j, m:n]` | Position-based rectangular slice |
| `df.loc[mask, 'col'] = v` | Conditional one-step assignment |
| `df.set_index('col')` / `reset_index()` | Index manipulation |

### Duplicates & sort
| Call | Purpose |
|---|---|
| `df.duplicated(subset=, keep=)` | Boolean mask of duplicates |
| `df.drop_duplicates(subset=, keep=)` | Drop duplicates by some/all columns |
| `df.sort_values(by, ascending)` | Sort rows by 1+ columns |

### Combine
| Call | Purpose |
|---|---|
| `pd.concat([a, b], axis=0/1, ignore_index=True)` | Stack rows / cols, no key |
| `pd.merge(a, b, on='k', how='inner/outer/left/right')` | Join on key |

### Aggregate / apply / group
| Call | Purpose |
|---|---|
| `s.sum/mean/count/min/max/std()` | Single agg on a Series |
| `df[['c1','c2']].agg(['sum','mean','min','max'])` | Multiple aggs on multiple columns |
| `s.apply(fn)` / `df.apply(fn, axis=1)` | Element / row-wise apply |
| `df.groupby(k)[c].mean()` | Per-group mean → Series |
| `df.groupby(k).agg({c1: 'mean', c2: 'sum'})` | Per-group multi-agg → DataFrame |
| `df.groupby(k).filter(fn)` | Keep rows whose group passes `fn` |
| `df.groupby(k).apply(fn)` | Custom per-group function |

### Missing data
| Call | Purpose |
|---|---|
| `df.isna() / df.isnull()` | NaN mask |
| `df.isna().sum()` | NaN count per column |
| `df.dropna(how=, subset=, axis=)` | Remove rows/cols with NaN |
| `df.fillna(v / method='ffill')` | Impute |

### Reshape & bin
| Call | Purpose |
|---|---|
| `pd.melt(df, id_vars=, value_vars=, var_name=, value_name=)` | Wide → long |
| `pd.pivot_table(df, index=, columns=, values=, aggfunc=)` | Long → wide |
| `pd.cut(s, bins=, labels=)` | Bin by value range |
| `pd.qcut(s, q=)` | Bin by quantile |

### Strings & datetime
| Call | Purpose |
|---|---|
| `s.str.contains('p', case=False, na=False)` | Substring search |
| `s.str.extract(r'(pat)')` | Regex extraction |
| `s.str.replace(',', '')` / `.lower()` / `.strip()` | Vectorized string ops |
| `pd.to_datetime(s, format=, errors='coerce')` | Parse datetime |
| `s.dt.year/.month/.day/.day_name()` | Datetime components |
| `s.dt.isocalendar().week` | ISO week number |
| `s.dt.strftime('%B %Y')` | Format datetime as string |

### Visualization (Matplotlib / Seaborn)
| Call | Purpose |
|---|---|
| `plt.figure(figsize=(w, h))` | Set canvas size |
| `plt.subplots(rows, cols, figsize=)` | Multi-axes figure |
| `plt.xticks(rotation=45)` | Rotate axis tick labels |
| `plt.axvline(x, color, linestyle)` | Reference vertical line |
| `plt.tight_layout()` | Rebalance padding |
| `sns.histplot(data, x, bins, kde)` | Histogram |
| `sns.countplot(data, x, palette)` | Categorical counts |
| `sns.kdeplot(data, x, shade)` | Smooth density |
| `sns.scatterplot(data, x, y, hue, alpha)` | Scatter |
| `sns.lineplot(data, x, y, marker)` | Line / trend |
| `sns.boxplot(data, x, y)` | Box plot |
| `sns.violinplot(data, x, y)` | Box + KDE |
| `sns.barplot(data, x, y, hue, estimator)` | Aggregated bar |
| `sns.pairplot(df, hue, diag_kind)` | Pairwise grid |
| `sns.heatmap(df, annot, fmt, cmap)` | Color-encoded matrix |
| `df.corr(method='pearson')` | Correlation matrix |

[🔝 Back to top](#top)

---

<a id="7-gotchas"></a>
## 7. ⚠️ Gotchas & traps — Amazon / Sachin focus

The full collection. Re-read this before any technical interview involving applied pandas, visualization, or probability.

### Joins & GroupBy

1. **`pd.merge` without `on=` joins on ALL common column names.** Subtle and dangerous — explicitly pass `on=` or `left_on=`/`right_on=`.
2. **Inner join is the default, not left.** First-time merge users get smaller results than expected.
3. **`groupby` excludes NaN keys by default.** Pass `dropna=False` (modern pandas) to keep them.
4. **`groupby().apply()` operated on grouping columns in older Pandas.** Now deprecated → pass `include_groups=False` (or it'll warn).
5. **`groupby().filter(fn)` returns a DataFrame, not a GroupBy.** You're filtering groups, but rows come back.
6. **`groupby('k')[['c']].mean()` returns a DataFrame; `groupby('k')['c'].mean()` returns a Series.** One bracket vs two.

### Apply

7. **`apply(axis=1)` is row-wise, NOT vectorized.** It's slow on big data. Try `np.where`, masks, or `.assign(...)` first.
8. **`apply(fn)` on a DataFrame defaults to `axis=0` (columns).** Pass `axis=1` for rows.

### Missing data

9. **`fillna(s.mean())` on a column with all NaN gives NaN.** `s.mean()` of all NaN is NaN — the impute is a no-op.
10. **`dropna(how='all')` drops rows where ALL values are NaN; `how='any'` drops if ANY is NaN.** Default is `'any'`.
11. **`isna()` and `isnull()` are identical.** Pick one and stick with it.
12. **Forward-fill (`method='ffill'`) propagates the previous value down.** Be careful when the prior value isn't a sensible default for the row.

### Reshape & bin

13. **`pivot()` errors on duplicate (index, columns) pairs — use `pivot_table` to aggregate.**
14. **`pd.cut(s, bins=...)` left edge is NOT included by default.** Pass `include_lowest=True` if your min value is on the boundary.
15. **`pd.cut` returns a categorical; arithmetic on it fails until you cast.**
16. **`pd.melt` `value_vars` must all be the same dtype** — mixing int and string forces object dtype.

### Strings & datetime

17. **`.str.contains` on a column with NaN raises `ValueError` unless you pass `na=False`.**
18. **`pd.to_datetime` without `format=` is slow** and can mis-parse ambiguous strings (`01/02/2023` US vs UK).
19. **`.dt.week` is deprecated — use `.dt.isocalendar().week`.**
20. **`.dt.strftime` returns strings, breaking subsequent date arithmetic.** Keep a datetime version for sorting.

### Visualization

21. **`df.corr()` silently drops non-numeric columns.** Check dtypes if a column is missing from the heatmap.
22. **Pearson r = 0 ≠ independence.** It misses non-linear patterns. Show a scatter when in doubt.
23. **`sns.pairplot` on too many columns** = unreadable + slow. Cap at ~6–8 numeric columns.
24. **`figsize` units are inches, not pixels.** A `figsize=(800, 600)` will be enormous (yes, this has happened).
25. **`hue` requires the categorical column to have ≤ ~10 levels** for the legend to stay readable.
26. **`countplot` on a continuous variable** plots one bar per unique value → looks terrible. Use `histplot` instead.
27. **Stacked bar plot of group counts — small categories get squished.** Normalize to percentages if proportions are the message.

### Probability

28. **Mutually exclusive ≠ independent.** They're opposites: mutually exclusive (with positive probability) implies dependent.
29. **`P(A ∪ B) = P(A) + P(B)` only when mutually exclusive.** Otherwise subtract P(A ∩ B).
30. **`pd.concat([A, B]).drop_duplicates()` for union** silently undercounts if A or B already had duplicate rows.
31. **Conditional probability denominator is `|B|`, not `total`.** Easy to miss in interview pressure.
32. **Empirical probabilities have sampling error.** Always state n.

[🔝 Back to top](#top)

---

<a id="8-advanced"></a>
## 8. 🎯 Advanced interview Q&A

Cross-module questions a senior interviewer would actually ask. Read the question, formulate an answer, then peek.

### Joins & GroupBy

**Q1. Two tables `orders` and `products` share `product_id`. You want one row per order with product info attached, never losing an order. Which join, and what's the row count?**
Left join with `orders` on the left. Row count = `len(orders)` exactly, possibly with NaN columns for orders whose `product_id` is missing in `products`.

**Q2. `df.groupby('region')['sales'].mean()` returns?**
A **Series** indexed by region. Single column selected before aggregation → Series. Two-bracket form returns a DataFrame.

**Q3. Difference between `groupby('k').agg('mean')` and `groupby('k').apply(np.mean)`?**
`agg('mean')` is C-optimized and handles NaN cleanly. `apply` calls a Python function per group — slower and treats NaN per the function's behavior. Always prefer `agg` for built-in reductions.

**Q4. How would you keep the top-3 products by sales within each category?**
```python
df.groupby('category', group_keys=False).apply(
    lambda g: g.nlargest(3, 'sales')
)
```
`group_keys=False` prevents an extra index level being added.

**Q5. Anti-pattern: looping over groups manually with `for name, g in df.groupby('k')`. When is it OK?**
For non-aggregation work that genuinely doesn't fit `agg`/`transform`/`apply` (e.g., writing each group to a separate file, training a model per group). For aggregations, never.

### Reshape & cleaning

**Q6. You have monthly revenue columns `Jan, Feb, ..., Dec`. How do you tidy this for plotting?**
`pd.melt` with `id_vars` = identifier cols and `value_vars` = month columns:
```python
pd.melt(df, id_vars=['product_id'], value_vars=['Jan','Feb',...,'Dec'],
        var_name='month', value_name='revenue')
```

**Q7. `pd.pivot_table` produces NaN cells — why and how to handle?**
Because some (index, columns) combos have no data. Pass `fill_value=0` (or another sentinel). Be careful — a real 0 is different from missing data analytically.

**Q8. `pd.cut(s, q=5)` — what's wrong?**
`q` isn't a `cut` parameter. You meant `pd.qcut(s, q=5)`. `cut` takes `bins=`.

**Q9. Strip `'₹1,200'` to a float — fastest way?**
```python
df['price'] = df['price'].str.replace('[₹,]', '', regex=True).astype(float)
```
One pass. Avoid Python-level `apply` for simple regex cleanup.

**Q10. After `pd.to_datetime`, `.dt.year` returns int — but later filtering breaks. Why?**
You probably ran `.dt.strftime('%Y')`, which returns a **string**. Filter with `df['ts'].dt.year == 2023` (int) or compare strings consistently. Don't mix.

### Visualization & correlation

**Q11. Strong upward curve, Pearson r ≈ 0 — what's going on?**
Pearson measures **linear** dependence. A perfect quadratic `y = x²` over a symmetric range has r = 0. Use Spearman (`df.corr('spearman')`) for monotonic, or visualize.

**Q12. Heatmap of correlations shows everything ≈ 0.99 — what likely happened?**
Either the variables are scaled versions of each other (collinear), the columns are accidentally the same (a join bug), or you computed correlation on the wrong axis. Inspect `df.head()`.

**Q13. Box plot whiskers — how are they computed?**
By default, whisker = 1.5 × IQR beyond Q1/Q3. Points outside are plotted as outlier dots. Set `whis=` to adjust.

**Q14. Stacked bar isn't aligning groups — what likely went wrong with the pre-plot pipeline?**
You probably skipped `unstack()`. Pattern: `df.groupby(['a','b']).size().unstack().plot(kind='bar', stacked=True)`. Without `unstack`, you have a 1-column long Series and the plot collapses.

**Q15. `sns.pairplot` is slow — alternatives?**
Use `sns.heatmap(df.corr())` for a quick correlation view, or sample rows (`df.sample(1000)`) before pair plotting. Drop irrelevant columns first.

### Probability

**Q16. P(A ∪ B) = 0.8, P(A) = 0.5, P(B) = 0.4. P(A ∩ B)?**
`P(A) + P(B) - P(A ∪ B) = 0.5 + 0.4 - 0.8 = 0.1`. Addition rule rearranged.

**Q17. P(century) = 0.13. P(century | runs > 50) = 0.39. What does this tell you?**
Conditioning on "runs > 50" triples the probability of a century — strong dependence. Sachin almost never scores a century without first crossing 50 (almost trivially, since 100 > 50). The interesting number is the conversion rate ABOVE 50.

**Q18. Two unfair coins: A = heads with P=0.7, B = heads with P=0.3. You pick one at random and flip — heads. Posterior P(A | heads)?**
Prior P(A) = 0.5. P(H|A) = 0.7, P(H|B) = 0.3. P(H) = 0.5·0.7 + 0.5·0.3 = 0.5. P(A|H) = (0.7·0.5)/0.5 = 0.7. Heads makes A more likely (since it favors heads).

**Q19. Why does the "Monty Hall" answer feel wrong even when you know it?**
The intuition that "two doors left, so 50/50" treats both doors symmetrically, but the host's action carries information. Bayes' theorem formalizes the asymmetry: the door you didn't pick is twice as likely.

**Q20. When is `len(A ∩ B) / len(B)` not a valid estimate of P(A | B)?**
When `B` is small (high sampling error) or when your data is biased (B's selection process introduces dependence). Always check sample size and selection.

[🔝 Back to top](#top)

---

<a id="9-vizdecisions"></a>
## 9. 📊 Visualization decision tree

Pick the right chart in 10 seconds.

```
What's the goal?
├── Distribution of ONE numeric variable
│   ├── Small data (≲100)  → strip plot / swarm plot
│   ├── Medium data        → histogram (sns.histplot)
│   └── Comparing distributions → kde plot or violin
│
├── Frequency of ONE categorical variable
│   └── sns.countplot
│
├── Relationship between TWO variables
│   ├── Both numeric         → sns.scatterplot (use alpha for density)
│   ├── Numeric over time    → sns.lineplot
│   ├── Numeric by category  → sns.boxplot / violinplot
│   └── Both categorical     → grouped bar or heatmap of counts
│
├── Three or more variables
│   ├── Quick screen          → sns.pairplot (≤ 8 numeric cols)
│   ├── Correlation overview  → sns.heatmap(df.corr())
│   ├── Composition           → stacked bar (proportions if comparing groups)
│   └── 3rd dim on a scatter  → hue (categorical) or size (numeric)
│
└── Trend with seasonality, anomalies, missing periods?
    └── Group + reset_index + sort by REAL datetime, then lineplot
```

Common chart-level decisions:

- **Skewed distribution?** → log-scale axis or `pd.qcut` for equal-count buckets.
- **Overlapping points?** → `alpha=0.3–0.5`, or `sns.kdeplot` for density contours.
- **Too many categories?** → top-N + "Other", or rotate to horizontal.
- **Comparing across groups?** → `hue` (if ≤ ~6 groups) or facet via `sns.FacetGrid` / `sns.catplot`.
- **Want proportions, not counts?** → normalize before stacking; or use `sns.barplot` with a pre-computed proportion column.

[🔝 Back to top](#top)

---

<a id="10-datasets"></a>
## 10. 📦 Dataset cheat sheets

### Amazon Sales (notebooks 1–3)

Two CSVs joined on `product_id`:

**`orders.csv`** (per product, comma-joined user/review lists):

| Column | Notes |
|---|---|
| `product_id` | Key. Join on this. |
| `user_id` | Comma-joined string of user IDs (one product, many users). |
| `user_name`, `review_id`, `review_title`, `review_content` | Comma-joined string fields per review. |
| `order_timestamp` | Parse with `pd.to_datetime`. |

**`products.csv`**:

| Column | Notes |
|---|---|
| `product_id` | Key. |
| `product_name` | String. |
| `category` | 9 categories (Computers&Accessories dominates). |
| `discounted_price`, `actual_price` | Strings like `'₹399'`, `'₹1,099'`. Strip `₹`/`,` then `astype(float)`. |
| `rating` | Float; has NaN. |
| `rating_count` | Float; has NaN. |
| `about_product` | Long text; use `.str.contains` for feature flags. |
| `img_link`, `product_link` | URLs; useful for `.str.extract` of query params. |

**Derived columns introduced across the notebooks:**
- `calculated_discount` = (actual − discounted) / actual × 100
- `review_title_uppercase` (apply)
- `price_range` (`pd.cut` of `actual_price`)
- `rating_category` (Poor / Average / Excellent buckets)
- `contains_durable` (`.str.contains`)
- `month_year`, `week_number`, `day_name` (`.dt` components)
- `review_status` (groupby-apply tag)

**Standard cleanup recipe:**
```python
# 1. Load
orders   = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

# 2. Join — preserve all orders
df = pd.merge(orders, products, on='product_id', how='left')

# 3. Strip currency and comma
def to_float(s):
    try: return float(str(s).replace('₹', '').replace(',', ''))
    except: return np.nan

df['actual_price']     = df['actual_price'].apply(to_float)
df['discounted_price'] = df['discounted_price'].apply(to_float)

# 4. Calculate discount %
df['calculated_discount'] = (df['actual_price'] - df['discounted_price']) / df['actual_price'] * 100

# 5. Datetime + components
df['ts']    = pd.to_datetime(df['order_timestamp'])
df['month'] = df['ts'].dt.strftime('%B %Y')

# 6. NaN audit
df.isnull().sum().sort_values(ascending=False)
```

### Sachin Tendulkar ODI (notebook 4)

Single CSV: `Sachin_ODI.csv` — 360 matches.

| Column | Type | Meaning |
|---|---|---|
| `runs` | int | Runs in the innings |
| `NotOut` | int | 1 if not out |
| `mins` | object | Minutes batted (stored as string) |
| `bf` | int | Balls faced |
| `fours`, `sixes` | int | Boundary counts |
| `sr` | float | Strike rate |
| `Inns` | int | Innings number (1 or 2) |
| `Opp` | object | Opponent team |
| `Ground` | object | Venue |
| `Date` | object | YYYY-MM-DD |
| `Winner` | object | Winning team name |
| `Won` | bool | India won |
| `century` | bool | Sachin scored ≥ 100 |

**Common questions:**
- P(century) = 46/360 ≈ 0.128
- P(century | runs > 50) ≈ 0.39
- P(runs > 50 ∩ Won) ≈ 0.20
- Best opponent? `df.groupby('Opp')['runs'].mean().sort_values(ascending=False)`
- Best year? Parse `Date` → year, group, mean runs.

[🔝 Back to top](#top)

---

<a id="11-business"></a>
## 11. Business questions → which API

| Business question | API |
|---|---|
| "Top 10 categories by revenue?" | `df.groupby('category')['actual_price'].sum().nlargest(10)` |
| "Average discount by category?" | `df.groupby('category')['calculated_discount'].mean()` |
| "Categories with avg rating ≥ 4?" | `df.groupby('category')['rating'].mean().loc[lambda s: s >= 4]` |
| "Power users (>15 reviews each)?" | `df.groupby('user_id').filter(lambda g: len(g) > 15)` |
| "Best-value products (high rating, low price)?" | Compute `score = rating / discounted_price`, then `nlargest(N, 'score')` |
| "Monthly rating trend?" | `df.groupby(df['ts'].dt.to_period('M'))['rating'].mean()` |
| "Are durable products rated higher?" | `df.groupby('contains_durable')['rating'].mean()` |
| "Distribution of prices?" | `sns.histplot(df['actual_price'])` (likely log-x or qcut buckets) |
| "Rating vs price relationship?" | `sns.scatterplot(data=df, x='actual_price', y='rating', alpha=0.4)` |
| "Correlation of pricing & rating columns?" | `sns.heatmap(df[['rating','price','discount']].corr(), annot=True)` |
| "P(Sachin scored century)?" | `(df['runs'] >= 100).mean()` |
| "P(India wins given Sachin > 50)?" | `df.loc[df['runs']>50, 'Won'].mean()` |
| "Sachin's best year?" | `df.groupby(pd.to_datetime(df['Date']).dt.year)['runs'].mean().idxmax()` |
| "Sachin vs each opponent?" | `df.groupby('Opp')['runs'].agg(['mean','max','count'])` |

[🔝 Back to top](#top)

---

<a id="12-probability"></a>
## 12. 🎲 Probability formula cheat sheet

| Concept | Formula |
|---|---|
| Sample space | S = set of all possible outcomes |
| Probability of event | 0 ≤ P(A) ≤ 1 |
| Complement | P(A′) = 1 − P(A) |
| Mutually exclusive | P(A ∩ B) = 0 |
| Independent | P(A ∩ B) = P(A) · P(B) |
| Addition rule | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) |
| Multiplication rule | P(A ∩ B) = P(A) · P(B \| A) = P(B) · P(A \| B) |
| Conditional | P(A \| B) = P(A ∩ B) / P(B), provided P(B) > 0 |
| Bayes' Theorem | P(A \| B) = P(B \| A) · P(A) / P(B) |
| Law of total probability | P(B) = Σᵢ P(B \| Aᵢ) · P(Aᵢ), Aᵢ partition S |

**Pandas idioms:**

```python
n = len(df)

# P(A)
P_A = (df['A']).mean()                                # if df['A'] is bool
P_A = len(df[df['runs'] > 50]) / n                    # general

# P(A ∩ B)
P_AB = ((df['A']) & (df['B'])).mean()
P_AB = len(df[(df['A']) & (df['B'])]) / n

# P(A | B)
P_A_given_B = df.loc[df['B'], 'A'].mean()             # bool A
# or: len(df[(df['A']) & (df['B'])]) / len(df[df['B']])

# Bayes
P_B_given_A = ((df['A']) & (df['B'])).sum() / df['A'].sum()
P_A_given_B = P_B_given_A * P_A / P_B
```

[🔝 Back to top](#top)

---

<a id="13-conceptmap"></a>
## 13. Cross-module concept map

```
Notebook 5 (Amazon 1): COMBINE & COMPUTE
  ├── select more flexibly (iloc/loc, conditional .loc)
  ├── remove duplicates (duplicated, drop_duplicates)
  ├── compute summaries (sum, mean, agg)
  ├── sort (single & multi-col, mixed ascending)
  ├── combine tables without keys (concat)
  ├── combine tables BY keys (merge, SQL joins)
  ├── transform with .apply (col-wise, row-wise, lambdas)
  └── split-apply-combine (groupby + agg / filter / apply)

Notebook 6 (Amazon 2): CLEAN & RESHAPE
  ├── missing data (isna, dropna, fillna)
  ├── reshape (melt for wide→long, pivot_table for long→wide)
  ├── bin (cut for value range, qcut for quantile)
  ├── strings (.str.contains, .str.extract, apply for custom)
  ├── datetime (to_datetime, .dt components, strftime)
  └── univariate viz (histplot, countplot)

Notebook 7 (Amazon 3): SEE RELATIONSHIPS
  ├── bivariate (scatter, line, box, violin, grouped bar)
  ├── multivariate (pairplot, heatmap, stacked)
  ├── correlation (.corr — linear; spearman for monotonic)
  └── styling (hue, alpha, palette, subplots, figsize)

Notebook 8 (Sachin ODI): REASON ABOUT UNCERTAINTY
  ├── sample space, events, set operations
  ├── addition / multiplication / complement rules
  ├── marginal vs joint vs conditional
  └── Bayes' Theorem — update beliefs from evidence

Threads tying everything together:
  - Pandas as the common substrate for cleaning, joining, and computing.
  - Visualization as the eye-test layer over computed summaries.
  - Probability as a formal framework for the patterns we observed.
```

[🔝 Back to top](#top)

---

<a id="sourced-bank"></a>
## 🌐 Sourced interview questions

> **Real questions paraphrased from canonical interview-prep sources.** Use this as your standalone practice bank — no internet required. Each batch keeps the source's original numbering where applicable.

### Batch 1 — Probability, from [`kojino/120-Data-Science-Interview-Questions`](https://github.com/kojino/120-Data-Science-Interview-Questions)

| # | Question (paraphrased) | One-liner answer |
|---|---|---|
| 1 | A family has 2 children, at least 1 is a girl. P(both girls)? | **1/3** — sample space `{BG, GB, GG}` after conditioning. |
| 2 | Shooting star has 30% chance of appearing every 15 min. P(see one in an hour)? | `1 − 0.7⁴ ≈ 0.76`. |
| 3 | Generate a uniform random integer 1–7 using only a fair 6-sided die. | Roll twice (36 outcomes); map 35 of them to 5 copies of {1..7}, reject the 36th. |
| 4 | Simulate a fair coin using a biased coin (p ≠ 0.5). | von Neumann: flip twice. `HT → H`, `TH → T`, otherwise repeat. |
| 5 | Two children, the **oldest** is a girl. P(both girls)? | **1/2** (different from Q1 — ordering changes the conditioning). |
| 6 | Couples have kids until first daughter, then stop. Expected gender ratio in the population? | **1 : 1** — counter-intuitive but correct. |
| 7 | Hash 10 objects into 10 bins. P(any collision)? | `1 − 10! / 10¹⁰ ≈ 0.9996`. |
| 8 | Expected number of fair-coin flips until you see 2 heads in a row. | **6**. (Set up the recurrence on states "0 heads" / "1 head".) |
| 9 | St. Petersburg paradox: payout = `$2ⁿ` for heads on flip n. Fair price? | **Infinite expected value**; in practice utility theory caps it. |
| 10 | You have a fair coin and a biased coin (always heads). Pick one at random, flip twice → both heads. P(you picked the fair coin)? | Bayes: `(½)(½ · ½) / [(½)(½·½) + (½)(1·1)] = 1/5`. |
| 11 | 5 ride-share cars arrive in random order. P(all 3 Lyfts arrive before all 2 Ubers)? | `C(5,3)⁻¹ = 1/10`. |
| 12 | Two users pick 5 adjectives each from a list of 24. P(at least one match)? | `1 − C(19,5)/C(24,5) ≈ 0.71`. |
| 13 | A father is in the 99th percentile for height. Expected percentile of his adult son? | Below 99th (regression to the mean). Conditional expectations shrink toward the population mean. |
| 14 | After 10 consecutive heads from one of two coins (fair + always-heads), P(fair)? | `(½ · (½)¹⁰) / [(½)(½)¹⁰ + (½)(1)] ≈ 0.000488`. |

### Batch 2 — ML theory & evaluation, from [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews)

| # | Question | One-liner answer |
|---|---|---|
| 15 | What is the normal distribution and why do we care? | Bell-shaped, fully specified by μ and σ. Central Limit Theorem says sample means converge to it — that's why so many stat tests assume normality. |
| 16 | How do you check if data follows a normal distribution? | Visual: histogram + Q-Q plot. Statistical: Shapiro–Wilk, Kolmogorov–Smirnov. |
| 17 | Why split data into train / validation / test? | Train fits, validation tunes hyperparameters, test gives the final unbiased estimate. Using test for tuning leaks information. |
| 18 | How does K-fold cross-validation reduce evaluation bias? | Every row is in the test set exactly once across the K splits; averaging K scores smooths out unlucky splits. |
| 19 | Precision vs recall — when does each matter? | Precision = "of predicted positives, how many are right?" — costly false positives (spam). Recall = "of actual positives, how many did we catch?" — costly false negatives (cancer screening). |
| 20 | When is ROC-AUC misleading compared to PR-AUC? | On highly imbalanced datasets — ROC can look great while PR shows the model is bad at the minority class. |
| 21 | L1 vs L2 regularization — practical difference? | L1 produces *sparse* coefficients (feature selection); L2 shrinks all coefficients toward zero without zeroing them out. |
| 22 | Why does linear regression assume homoscedasticity? | Constant error variance is required for OLS standard errors to be valid; violated → wrong p-values, wrong CIs. |
| 23 | Why do tree models struggle with time series extrapolation? | They predict by averaging training leaves; they cannot extrapolate beyond observed value ranges. |

### Batch 3 — Applied pandas (advanced), from [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles) and StrataScratch patterns

| # | Question | One-liner answer |
|---|---|---|
| 24 (#23) | Subtract row mean from each element in a DataFrame. | `df.sub(df.mean(axis=1), axis=0)` |
| 25 (#27) | Sum the top-3 values per group. | `df.groupby('k')['v'].apply(lambda g: g.nlargest(3).sum())` |
| 26 (#28) | Bin continuous values and sum per interval. | `df.groupby(pd.cut(df['x'], bins=...))['v'].sum()` |
| 27 (#35) | Resample a time series by month and aggregate. | `df.set_index('ts').resample('M').agg({'sales':'sum','price':'mean'})` |
| 28 (#43) | Expand a list-valued column into rows. | `df.explode('col')` |
| 29 (Amazon, Medium) | For each category, find products with sales above the category median. | `df[df['sales'] > df.groupby('cat')['sales'].transform('median')]` |
| 30 (Meta, Medium) | Top-N events per user per day. | `df.assign(rank=lambda d: d.groupby(['user','day'])['ts'].rank()).query('rank <= N')` |
| 31 (Google, Easy) | 7-day rolling average. | `df.set_index('date').rolling('7D').mean()` |
| 32 (Common) | A left join produces duplicate rows — why? | The right table has duplicate keys; left join is m:n, so each left row matches every right row with the same key. Use `validate='m:1'` to catch this. |
| 33 (Common) | `groupby().apply()` triggers a deprecation warning — what's the modern fix? | Pass `include_groups=False` so grouping columns aren't passed to the function. |

### Batch 4 — Visualization & correlation, paraphrased common questions

| # | Question | One-liner answer |
|---|---|---|
| 34 | Pearson `r = 0` — are the variables independent? | **No** — only "no linear relationship." A perfect quadratic over a symmetric range gives r = 0. Use Spearman for monotonic, visualize for non-linear. |
| 35 | When choose Spearman over Pearson? | When the relationship is monotonic but not linear (ranks correlate). Also robust to outliers. |
| 36 | What's the difference between a box plot and a violin plot? | Box = 5-number summary (min/Q1/median/Q3/max + outliers). Violin = box + kernel density estimate → reveals distribution *shape* (bimodal, skewed). |
| 37 | Why use `alpha < 1` on a scatter plot? | To handle overplotting — overlapping points stack into one black blob and lose density info. `alpha=0.3–0.5` shows where points cluster. |
| 38 | `df.corr()` silently drops a column — what happened? | The column is non-numeric (`object` dtype). `.corr()` only computes on numeric. Cast first or pre-select. |

### Citations & where to drill more
- 🎯 [`kojino/120-Data-Science-Interview-Questions`](https://github.com/kojino/120-Data-Science-Interview-Questions) — probability + stats sections.
- 🎯 [`alexeygrigorev/data-science-interviews`](https://github.com/alexeygrigorev/data-science-interviews) — theory + coding.
- 🎯 [`chiphuyen/ml-interviews-book`](https://huyenchip.com/ml-interviews-book/) — probability chapter, ML system design.
- 🎯 [`ajcr/100-pandas-puzzles`](https://github.com/ajcr/100-pandas-puzzles) — applied pandas drills.
- 🎯 **StrataScratch** — real FAANG pandas/SQL.

[🔝 Back to top](#top)

---

<a id="14-drill"></a>
## 14. 🔁 100-question revision drill

Read each question, answer in your head, peek. Aim for under 15 seconds per question — finish all 100 in under 30 minutes.

### Block A — Joins, GroupBy, Apply (Q1–25)

1. `pd.concat` vs `pd.merge` in one word? → **concat: stack; merge: join-by-key**
2. Default `how=` in `pd.merge`? → **`'inner'`**
3. Left join row count? → **`len(left)`** (matches added where they exist)
4. Outer join semantics? → **Union of keys, NaN fills**
5. Inner join semantics? → **Intersection of keys only**
6. Right join — when use? → **When the right table is the "primary" view**
7. `df.duplicated()` returns? → **Boolean mask, first occurrence is False**
8. `keep='last'` does? → **First occurrences become True (duplicates)**
9. `keep=False` does? → **ALL occurrences marked duplicate**
10. `drop_duplicates(subset=['x'])` checks? → **Duplicates only on column `x`**
11. `drop_duplicates` modifies in place? → **No — pass `inplace=True` or re-assign**
12. `df['c'].agg(['mean','min'])` returns? → **Series indexed by stat name**
13. `sort_values` default ascending? → **True**
14. `sort_values(by=['a','b'], ascending=[True,False])` → **Multi-col mixed sort**
15. `.apply(fn)` on a Series? → **Elementwise**
16. `df.apply(fn, axis=1)`? → **Per row; row passed as a Series**
17. `df.apply(fn, axis=0)`? → **Per column (the default)**
18. Is `.apply(axis=1)` vectorized? → **No — Python loop**
19. Faster than `.apply` for elementwise math? → **Vectorized ops / `np.where`**
20. `df.groupby('k')[c].mean()` returns? → **Series**
21. `df.groupby('k')[[c]].mean()` returns? → **DataFrame**
22. `groupby().filter(fn)` returns? → **DataFrame of rows whose group passed**
23. `groupby().apply(fn)` returns? → **Whatever `fn` returned, combined**
24. Split-apply-combine in one phrase? → **Group rows, run fn per group, combine results**
25. `groupby` skips NaN keys by default? → **Yes — pass `dropna=False` to include**

### Block B — Missing, Reshape, Strings, DateTime (Q26–50)

26. `isna()` vs `isnull()`? → **Identical**
27. NaN count per column? → **`df.isna().sum()`**
28. Drop rows with any NaN? → **`df.dropna()` (default `how='any'`)**
29. Drop rows where ALL NaN? → **`df.dropna(how='all')`**
30. Fill NaN with column mean? → **`df['c'].fillna(df['c'].mean())`**
31. Forward fill? → **`df['c'].fillna(method='ffill')`** (or `.ffill()`)
32. `pd.melt` does? → **Wide → long; stack value columns into rows**
33. `pd.pivot_table` does? → **Long → wide with aggregation**
34. `pivot` vs `pivot_table`? → **`pivot` errors on duplicates; `pivot_table` aggregates**
35. `pd.cut` does? → **Bin by value range**
36. `pd.qcut` does? → **Bin by quantile (equal counts)**
37. NaN-safe substring search? → **`.str.contains('foo', na=False)`**
38. Regex extract a group? → **`.str.extract(r'(pat)')`**
39. Strip `'₹'` and `','`? → **`.str.replace('[₹,]','',regex=True)`**
40. Parse datetime? → **`pd.to_datetime(s, format='...')`**
41. Speed up `to_datetime`? → **Pass `format=`**
42. ISO week number? → **`s.dt.isocalendar().week`**
43. Day name? → **`s.dt.day_name()`**
44. Format datetime as text? → **`s.dt.strftime('%B %Y')`**
45. After `strftime`, dtype is? → **`object` (string) — breaks date arithmetic**
46. Sort by month-year correctly? → **Keep a datetime version for the sort key**
47. `pd.cut(s, bins=[0,5,10])` — is 0 included? → **No — pass `include_lowest=True`**
48. CSV with `'00123'` IDs preserved? → **`dtype={'id': str}` at load**
49. `histplot` for? → **Continuous variable**
50. `countplot` for? → **Categorical variable**

### Block C — Visualization & correlation (Q51–75)

51. `figsize=(w, h)` units? → **Inches**
52. `alpha` does? → **Transparency (0–1)**
53. `hue` does? → **Categorical color encoding**
54. Rotate x-tick labels 45°? → **`plt.xticks(rotation=45)`**
55. Multi-axes figure? → **`fig, axes = plt.subplots(rows, cols)`**
56. Reference vertical line? → **`plt.axvline(x, color, linestyle)`**
57. Fix overlapping labels? → **`plt.tight_layout()`**
58. Box plot whisker default? → **1.5 × IQR**
59. Violin plot adds? → **KDE — shows distribution shape**
60. Box vs violin? → **Box: 5-num summary; Violin: + density shape**
61. Pair plot good for? → **5–8 numeric columns — pairwise overview**
62. Heatmap of? → **`df.corr()` is the classic use**
63. `df.corr()` skips? → **Non-numeric columns silently**
64. Pearson r = 0 means? → **No LINEAR relationship — non-linear may exist**
65. r in [−1, 1]; |r| ≥ 0.7 means? → **Strong linear relationship**
66. Spearman vs Pearson? → **Spearman: monotonic; Pearson: linear**
67. Stacked bar from groupby? → **`df.groupby([a,b]).size().unstack().plot(kind='bar', stacked=True)`**
68. Stacked bar percentages? → **Normalize each row first** with `.div(row_sums, axis=0)`
69. Scatter with overplotting fix? → **Use `alpha`** or `sns.kdeplot` (2D)
70. Comparing distributions side-by-side? → **`sns.boxplot(x='cat', y='val')`** or violin
71. Adding third dim to scatter? → **`hue` (cat) or `size` (num)**
72. Heatmap annotations format? → **`fmt='.2f'`** for 2 decimals
73. Color palette example? → **`palette='coolwarm'`** or `'Blues'`, `'Spectral'`
74. Univariate vs bivariate? → **1 var vs 2 vars**
75. Multivariate plot examples? → **pairplot, heatmap, stacked bar, hue-scatter**

### Block D — Probability (Q76–95)

76. Sample space? → **Set of all possible outcomes**
77. Event? → **Subset of the sample space**
78. P(A′) = ? → **1 − P(A)**
79. Mutually exclusive condition? → **P(A ∩ B) = 0**
80. Independent condition? → **P(A ∩ B) = P(A)·P(B)**
81. Mutually exclusive ⇒ independent? → **No (with positive P, opposite)**
82. Addition rule? → **P(A∪B) = P(A) + P(B) − P(A∩B)**
83. When does subtraction vanish? → **When mutually exclusive**
84. Multiplication rule? → **P(A∩B) = P(A) · P(B\|A)**
85. Conditional formula? → **P(A\|B) = P(A∩B) / P(B)**
86. Bayes' Theorem? → **P(A\|B) = P(B\|A)·P(A) / P(B)**
87. Empirical P from pandas? → **`len(df[event]) / len(df)`** or `(mask).mean()`
88. P(A∩B) via pandas? → **`((df['a']) & (df['b'])).mean()`**
89. P(A\|B) via pandas? → **`df.loc[df['b'], 'a'].mean()`**
90. Union via pandas? → **`((df['a']) | (df['b'])).mean()`** or `concat.drop_duplicates`
91. Compound condition operator? → **`&` `|` `~`** with parentheses around comparisons
92. Why parens? → **`&` precedence is higher than `>`/`==`**
93. With replacement → events are? → **Independent**
94. Without replacement → events are? → **Dependent**
95. Posterior > prior when? → **Evidence is more likely under the hypothesis** (LR > 1)

### Block E — Cross-cutting traps (Q96–100)

96. `groupby().apply` deprecation warning fix? → **Pass `include_groups=False`**
97. `pd.merge` joins on what if you omit `on=`? → **ALL common column names**
98. `.str.contains` raises on NaN? → **Yes — pass `na=False`**
99. `pivot()` errors with duplicates → use? → **`pivot_table()`**
100. P(A∪B) when A, B are mutually exclusive simplifies to? → **P(A) + P(B)**

**Score yourself:** 90+ = strong, 75–89 = solid, 60–74 = revise, <60 = re-read modules.

[🔝 Back to top](#top)

---

<a id="15-bestpractices"></a>
## 15. ✅ Best practices

Distilled habits from the four applied notebooks.

### Pipeline hygiene

1. **Always start with `df.shape`, `df.info()`, `df.head()`, `df.isnull().sum()`** on a new dataset.
2. **Clean before joining.** Don't merge two messy tables — clean each, then merge.
3. **Audit join row counts.** A left join should preserve `len(left)`; surprises mean duplicate keys.
4. **Save intermediates with `to_csv(..., index=False)`** — cleaning is expensive to repeat.

### Correctness

5. **One-step `.loc` assignment** for conditional updates: `df.loc[mask, 'col'] = val`.
6. **Parenthesize compound masks**: `(df.x > 0) & (df.y < 5)`.
7. **`.str.contains(..., na=False)`** to avoid NaN-induced errors.
8. **Pass `format=` to `pd.to_datetime`** for speed and unambiguous parsing.
9. **Don't sort `.dt.strftime` strings** — keep a real datetime for the sort key.
10. **Use `pd.qcut` for skewed distributions**; `pd.cut` for known business thresholds.

### Performance

11. **Replace `.apply(axis=1)` with vectorized ops** wherever possible.
12. **`groupby().agg(...)` over `groupby().apply(...)`** for built-in reductions.
13. **`df.corr()` only on numeric columns** — pre-select to avoid silent drops.
14. **Avoid `pairplot` on > 8 columns** — pre-select or use a heatmap instead.

### Visualization

15. **Use `alpha`** when scatter points overplot.
16. **Rotate x-tick labels** for long category names: `plt.xticks(rotation=45)`.
17. **Call `plt.tight_layout()`** before `plt.show()` to avoid label overlap.
18. **Bucket high-cardinality categoricals** with a `top-N + "Other"` before plotting.
19. **Annotate the heatmap (`annot=True, fmt='.2f'`)** when communicating to non-technical viewers.

### Probability

20. **Always state n** alongside an empirical probability.
21. **Verify "independent" vs "mutually exclusive"** before applying addition/multiplication rules.
22. **Reach for Bayes** when you have a posterior question (probability of cause given evidence).
23. **Compute conditional probability with `.loc[mask, 'target'].mean()`** when the target is boolean — cleaner than nested filters.

### Interview-day reminders

24. **Read row counts after every join** — `len(df)` is your sanity check.
25. **Walk through the split-apply-combine narrative** when GroupBy comes up.
26. **Pearson r = 0 ≠ no relationship** — explicitly mention non-linear.
27. **Quote the formula before plugging numbers** for probability questions.
28. **When unsure, sketch a 2×2 table** — it disambiguates joint, marginal, conditional.

[🔝 Back to top](#top)

---

<a id="16-mapping"></a>
## 16. 📦 Notebook mapping

| Notebook | Title | Covers | This guide |
|---|---|---|---|
| 5 | Amazon Sales Data Analysis 1 | Combined `iloc`/`loc`, conditional updates, duplicates, aggregations, sorting, `concat` vs `merge`, SQL joins, `.apply` (col/row/lambda), `isnull().sum()`, `groupby` (split-apply-combine, filter, apply) | [§1](#1-module1) |
| 6 | Amazon Sales Data Analysis 2 | Missing data (`isna`/`dropna`/`fillna`), reshape (`melt`/`pivot_table`), binning (`pd.cut`), string ops, datetime parsing & `.dt`, univariate viz (histplot, countplot) | [§2](#2-module2) |
| 7 | Amazon Sales Data Analysis 3 | Bivariate viz (scatter/line/box/violin/grouped bar), multivariate (pairplot/heatmap/stacked), `.corr()`, hue/subplots styling | [§3](#3-module3) |
| 8 | Analyzing Sachin Tendulkar's ODI Career | Probability fundamentals, set ops on events, addition/multiplication/complement rules, marginal vs joint, conditional probability, Bayes' Theorem — applied empirically with pandas | [§4](#4-module4) |

[🔝 Back to top](#top)
