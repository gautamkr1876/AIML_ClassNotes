<a id="top"></a>
# Amazon Sales + Sachin Tendulkar EDA — Master Revision Guide

> **Interview-ready revision sheet for the four new notebooks** (Amazon Sales Data Analysis 1/2/3 + Analyzing Sachin Tendulkar's ODI Career). Covers applied pandas, data cleaning, reshaping, datetime, visualization, correlation, and probability — with cheat sheets, advanced Q&A, gotchas, and a 100-question drill.

**Companion guides:**
- 🐍 [`Data_Foundation_Revision_Guide.md`](./Data_Foundation_Revision_Guide.md) — NumPy foundations
- 🐼 [`Pandas_Revision_Guide.md`](./Pandas_Revision_Guide.md) — Pandas basics (Series, DataFrame, `iloc`/`loc`, cleanup)

**How to use:**
- **First-time learning a concept:** open the module's **📖 Guided concept walkthrough** ([M1](#1g-guided) · [M2](#2g-guided) · [M3](#3g-guided) · [M4](#4g-guided)). Each concept is introduced *what → why → how → where → related → code → gotcha* — beginner-friendly, no follow-up search needed.
- **Pre-interview:** scan the [🚀 Topic finder](#topic-finder) → open the relevant module → use the recap cheat sheet + drill the Q&A.
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
| 📖 First-time intro to a concept (what / why / how / where / related) | [M1 walkthrough](#1g-guided), [M2 walkthrough](#2g-guided), [M3 walkthrough](#3g-guided), [M4 walkthrough](#4g-guided) |
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

1. [Module 1 — Joins, GroupBy, Apply (Amazon 1)](#1-module1) · [📖 Guided walkthrough](#1g-guided)
2. [Module 2 — Missing Data, Reshape, Strings, DateTime, Univariate Viz (Amazon 2)](#2-module2) · [📖 Guided walkthrough](#2g-guided)
3. [Module 3 — Bivariate & Multivariate Visualization (Amazon 3)](#3-module3) · [📖 Guided walkthrough](#3g-guided)
4. [Module 4 — Probability via Pandas (Sachin Tendulkar ODI)](#4-module4) · [📖 Guided walkthrough](#4g-guided)
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

<a id="1g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 1 concept. Read this top-to-bottom on a first pass; the cheat sheet below is the recap surface. The depth here is on purpose — these are the concepts FAANG pandas rounds rely on.

#### `pd.merge` — joining two tables on a key

> **🪜 Mental model:** *SQL JOIN, written in pandas.* Two tables share a column (a key); `merge` lines them up row-by-row so you get one combined table with all the matching information side-by-side.

**What it is.** `pd.merge(left, right, on='key', how='inner')` is the pandas equivalent of SQL's `JOIN`. It takes two DataFrames, finds rows where a shared column (the **key**) matches, and produces a new DataFrame with columns from both. The `how=` parameter (one of `'inner'`, `'outer'`, `'left'`, `'right'`) decides which rows survive when the key matches in one table but not the other.

**Why it matters.** Real data lives in many tables — orders in one, products in another, users in a third. To answer "what's the average rating per category?", you must first join the order table to the product table. Joins are the most-asked pandas/SQL interview topic at Amazon, Meta, Airbnb because almost every analysis starts with one. Getting `how=` wrong silently drops or duplicates rows.

**How it works.**
1. Pandas builds a hash table of keys from one side (whichever is smaller).
2. It scans the other side, looks up each row's key in the hash table.
3. For every match, it emits a combined row with columns from both sides.
4. The `how=` parameter decides what to do with non-matches:
   - `inner` — drop them.
   - `outer` — keep both sides, NaN-fill the missing columns.
   - `left` / `right` — keep one side fully, NaN-fill the other.
5. Duplicate keys multiply: 3 matches on the left × 4 on the right → 12 output rows.

**Where it's used.** Joining orders to products, users to events, transactions to accounts. In sklearn pipelines, you usually `merge` before `fit` so every example has its label and features attached. StrataScratch Amazon questions almost always start with a merge.

**Related terms.**
- **Inner / outer / left / right join** — the four `how=` modes (see each below).
- **`pd.concat`** — the no-key cousin; just stacks rows or columns without matching on anything.
- **Key** — the shared column you join on; pass via `on=` (same name on both sides) or `left_on=`/`right_on=` (different names).
- **`validate='1:1' / '1:m' / 'm:1' / 'm:m'`** — pandas guard that raises if the join is not the cardinality you expected. Use to catch row explosions.
- **SQL `JOIN`** — the database-side version; semantics are identical.

```python
pd.merge(orders, products, on='product_id', how='inner')
# only orders where product_id matches a row in products
```

**Gotcha.** Calling `pd.merge(a, b)` with no `on=` silently joins on **every** shared column name — almost always wrong on real data. Always pass `on=` explicitly.

#### Inner join (`how='inner'`)

> **🪜 Mental model:** *The intersection — only the overlap survives.* Keep rows where the key exists on **both** sides; drop everything else.

**What it is.** An inner join is `pd.merge(a, b, on='k', how='inner')`. Only rows whose key value appears in **both** `a` and `b` are kept. It's the default `how=`, and it's the smallest possible result of a merge.

**Why it matters.** Inner is the "I want only complete records" mode. If you're computing per-customer revenue and a customer has no matching record in the products table, inner drops them — which is usually correct because you can't compute revenue for them. The flip side: inner silently loses rows, so if you expected `len(a)` outputs but got fewer, you might have an inner join eating data.

**How it works.** Pandas finds the **intersection** of key values present in both tables, then emits one output row per matching pair. If `a` has 3 rows with `id=5` and `b` has 2 rows with `id=5`, you get 6 output rows (`3 × 2` Cartesian product within that key).

**Where it's used.** "Show me only orders with valid products." "Show me users who also signed up for newsletter." Default in SQL `SELECT a JOIN b ON a.id = b.id`. Used in feature pipelines when you only want examples that have *every* feature available.

**Related terms.**
- **Outer join** — the opposite: keep all rows from both sides.
- **Left/right join** — keep one side fully even on no match.
- **Set intersection** — the mathematical concept inner join implements on keys.
- **Cartesian product / row explosion** — what happens when both sides have duplicates of a key.

```python
pd.merge(orders, products, on='product_id', how='inner')
# size ≤ min(len(orders), len(products)) when keys are unique
```

**Gotcha.** Inner is silent: it drops mismatched rows without warning. Always print `len(result)` and compare to `len(left)` to know what got eaten.

#### Outer join (`how='outer'`)

> **🪜 Mental model:** *The union — keep everyone, NaN-fill where missing.* Every key value from either side appears in the result; missing columns get filled with NaN.

**What it is.** An outer (a.k.a. full outer) join is `pd.merge(a, b, on='k', how='outer')`. Every key value that appears in **either** table contributes a row to the output. When a row has no match on the other side, the columns from that side come back as `NaN`.

**Why it matters.** Outer is the "I want to see what's missing" mode — it surfaces rows that don't have a match, which is exactly what you want during data audits ("which orders have no matching product? which products have no orders?"). It's the biggest possible result.

**How it works.** Pandas takes the **union** of key values from both tables. For each key, if both sides have it, you get the matched-pair rows (same as inner). If only one side has it, you get one row with the absent side's columns set to `NaN`. Pandas can optionally tag each row with an `_merge` column (via `indicator=True`) telling you which side(s) the row came from.

**Where it's used.** Data-quality audits ("show me missing on either side"). Reporting where every customer should appear even if they have no orders. Pre-modelling sanity check: outer + `indicator=True` is the standard "did my join lose anyone?" debugging trick.

**Related terms.**
- **Inner join** — the opposite: keep only matches.
- **Left / right join** — keep one side fully; outer keeps both.
- **`indicator=True`** — adds an `_merge` column with values `left_only`, `right_only`, `both`.
- **Set union** — the mathematical concept outer join implements on keys.

```python
pd.merge(orders, products, on='product_id', how='outer', indicator=True)
# inspect result['_merge'].value_counts() to see what came from where
```

**Gotcha.** Outer produces NaN columns wherever a side is missing — those NaNs propagate through subsequent math (`mean()`, `sum()`) unless you handle them.

#### Left join (`how='left'`)

> **🪜 Mental model:** *Keep all of the left table; bolt on info from the right where it matches.* The left table is the source of truth; the right is decoration.

**What it is.** A left join is `pd.merge(a, b, on='k', how='left')`. **Every** row of `a` appears in the output. Where `b` has a matching key, its columns are filled in; where it doesn't, those columns are `NaN`. The output has at least `len(a)` rows (more if `b` has duplicate keys).

**Why it matters.** Left is the most-used join in practice because it preserves the row set you started with — your "fact table" stays whole and you just enrich it with attributes. If you're computing metrics per order, you want one row per order at the end, never to lose one. Pre-modelling: left-join your label table to your feature table so every label row gets feature columns.

**How it works.** For each row in `a`, look up its key in `b`. If a match exists, emit one combined row per match (multiple matches → multiple output rows). If no match, emit one row with `b`'s columns as `NaN`. Order from `a` is preserved.

**Where it's used.** "Orders enriched with product info." "Users enriched with their last-known event." Almost every "report" query in analytics. In SQL, `LEFT JOIN` is the most common JOIN type for the same reason.

**Related terms.**
- **Right join** — mirror image: keep all of right.
- **Inner join** — drops left rows that don't match; left preserves them.
- **Outer join** — also keeps right-only rows; left does not.
- **`validate='m:1'`** — pandas guard for "left has many, right has at most one"; raises on right-side duplicates that would explode rows.

```python
pd.merge(orders, products, on='product_id', how='left')
# len(result) >= len(orders); equality only when right has no duplicate keys
```

**Gotcha.** "Left join, so result has `len(orders)` rows" is **wrong** when the right side has duplicate keys — duplicates on the right explode left rows (1 left × 3 right matches → 3 output rows).

#### Right join (`how='right'`)

> **🪜 Mental model:** *Mirror of left join.* Keep all of the **right** table, fill left columns with NaN on no match.

**What it is.** A right join is `pd.merge(a, b, on='k', how='right')` — semantically equivalent to swapping the arguments and doing a left join. Every row in `b` appears in the output, decorated with matching columns from `a` (NaN-filled where no match).

**Why it matters.** Right join exists for completeness; in practice you'll almost never write one — people prefer to swap the argument order and write `how='left'` instead, because reading "left join" left-to-right matches reading the code left-to-right. Knowing it exists is mostly an interview vocabulary check.

**How it works.** Same algorithm as left join but with the roles flipped — pandas iterates over `b` and looks up keys in `a`. Output row count ≥ `len(b)`.

**Where it's used.** Rare in practice. Sometimes used inside macros/abstractions where you can't reorder arguments. Some SQL dialects (legacy) reserve `RIGHT JOIN` for symmetry with `LEFT JOIN`.

**Related terms.**
- **Left join** — the more idiomatic mirror of this.
- **"Swap and left-join"** — the canonical rewrite that makes code readable.

```python
pd.merge(orders, products, on='product_id', how='right')
# == pd.merge(products, orders, on='product_id', how='left')
```

**Gotcha.** Two writers of the same query will disagree on argument order. Prefer `how='left'` and put your "source of truth" table on the left; you'll never need `how='right'`.

#### `groupby` — split-apply-combine

> **🪜 Mental model:** *Sort into bins, do a thing in each bin, glue the answers back.* GroupBy splits the DataFrame into one mini-DataFrame per unique key value, runs a function inside each, and stitches the results.

**What it is.** `df.groupby('key')` returns a lazy `GroupBy` object — no work happens yet. You then call an aggregation (`.mean()`, `.sum()`, `.agg()`), a transform (`.transform()`), a filter (`.filter()`), or `.apply()` on it to trigger the work. The output type depends on which operation you used. The whole pattern is named **split-apply-combine** (after Hadley Wickham's paper).

**Why it matters.** Group-level analytics — "average rating per category", "total revenue per region", "max temperature per day" — are the bread and butter of EDA and reporting. Every SQL `GROUP BY` query maps directly to a pandas `groupby`. FAANG SQL/pandas rounds almost always include at least one.

**How it works.**
1. **Split** — pandas sorts the rows into bins, one per unique value of the grouping key (or one per unique combination if you pass multiple keys).
2. **Apply** — your function runs inside each bin, independently.
3. **Combine** — pandas glues the per-bin results into a Series or DataFrame, indexed by the group keys.

The return shape depends on what you applied:
- `.agg()` → one row per group (collapses).
- `.transform()` → same shape as the input (no collapse, but each row gets its group's value).
- `.filter()` → returns rows of the original DataFrame whose group passed a predicate.
- `.apply()` → escape hatch; returns whatever your function returns.

**Where it's used.** SQL `GROUP BY` translations. Per-cohort metrics ("conversion rate per signup-month"). Feature engineering ("rolling-mean per customer"). In sklearn, `cross_val_score` uses an internal group-aware split when you pass `groups=`.

**Related terms.**
- **`.agg()`** — the collapsing operation (see below).
- **`.transform()`** — same-shape return (e.g., subtract group mean).
- **`.filter()`** — keep/drop whole groups based on a predicate.
- **`.apply()`** — most general; slowest.
- **`pivot_table`** — closely related; combines groupby + reshape in one step.
- **MultiIndex** — what you get when grouping by multiple keys.

```python
df.groupby('category')['rating'].mean()
# Series indexed by category → average rating per category
```

**Gotcha.** `df.groupby('k')` is *lazy* — it returns a GroupBy object, not a DataFrame. Printing it shows `<pandas.core.groupby...>` and that confuses beginners. Chain an aggregation to materialise.

#### `agg` — named aggregations, multiple at once

> **🪜 Mental model:** *Apply many summary functions in one call.* Instead of computing `.mean()`, `.max()`, `.std()` separately, pass them all to `agg` and get a tidy table.

**What it is.** `.agg()` accepts a list, dict, or named-aggregation spec and returns one row per group with one column per requested aggregation. List form (`agg(['mean', 'max'])`) gives generic column names. Dict form (`agg({'price': 'mean', 'rating': 'max'})`) lets you specify per-column functions. Named-aggregation form (`agg(avg_price=('price', 'mean'))`) gives you control over both the source column and the output name.

**Why it matters.** Real reports always need multiple stats per group ("mean, min, max revenue per region"). Doing them with three separate groupbys is wasteful — `agg` does them in one pass over the data. The named-aggregation form is the pandas-recommended pattern and is what FAANG interviews expect today (post-pandas 0.25).

**How it works.** Pandas iterates over each group once, applies every aggregator to every requested column, and assembles the result. Output is a DataFrame with the group key(s) as the index and one column per `(column, function)` pair. With named aggregations, you control the output column name explicitly.

**Where it's used.** Every "summary report" query. Building feature tables (per-user stats: count, mean, std of past orders). Pre-modelling diagnostics ("describe per class").

**Related terms.**
- **Named aggregation** — the `output_name=('column', 'func')` syntax; preferred over dict syntax for readability.
- **`.describe()`** — `agg` with a fixed canonical list (count/mean/std/min/quartiles/max).
- **`.transform()`** — like `agg` but returns same-shape data instead of collapsing.
- **`.apply()`** — fallback when your aggregator is too custom for `agg`.

```python
df.groupby('category').agg(
    avg_price=('price', 'mean'),
    max_rating=('rating', 'max'),
    n=('product_id', 'count'),
)
```

**Gotcha.** Old-style dict-of-dicts syntax (`agg({'price': {'avg': 'mean'}})`) is deprecated. Use named aggregations instead.

#### `apply` — row-wise / column-wise / per-group function

> **🪜 Mental model:** *The escape hatch.* When no built-in does what you want, hand pandas a Python function and let it run on each row, column, or group.

**What it is.** `.apply(fn)` runs your function `fn` on every element of a Series, every row/column of a DataFrame, or every group of a GroupBy. Direction is set by `axis=` (DataFrame only): `axis=0` (default) applies per column (each column passed as a Series); `axis=1` applies per row (each row passed as a Series with the column names as keys).

**Why it matters.** `apply` is the universal hammer — anything you can do in Python, you can wrap in `apply`. But it's also slow because it usually runs a Python-level loop, breaking pandas' compiled-C path. Interviewers love asking "is this fast?" precisely because beginners reach for `apply` when a vectorized op would be 100× faster.

**How it works.**
1. **Series.apply(fn)** — `fn` is called once per element; output is a Series of the same length.
2. **DataFrame.apply(fn, axis=0)** — `fn` is called once per column (each column passed as a Series).
3. **DataFrame.apply(fn, axis=1)** — `fn` is called once per row (each row passed as a Series).
4. **GroupBy.apply(fn)** — `fn` is called once per group, receiving the group's DataFrame; output can be a scalar, Series, or DataFrame.

In every case, pandas glues the per-call results back into a Series, DataFrame, or grouped object.

**Where it's used.** Custom row-level features (`row.actual - row.discount`). Multi-column string cleaning. Any "I'd write a Python for-loop here" moment. In groupby, `apply` is the catch-all when `agg` and `transform` can't express what you need.

**Related terms.**
- **`np.vectorize`** — looks vectorized but is also a Python loop internally; no faster than `apply`.
- **`.map()`** — element-wise like `Series.apply` but slightly different semantics (also supports dicts for replacement).
- **`.transform()`** — like `apply` on a group but constrained to return same-shape data.
- **Vectorization** — the *fast* alternative; uses array math, `np.where`, `.str.*`.

```python
df['discount_pct'] = df.apply(
    lambda r: (r['actual_price'] - r['discounted_price']) / r['actual_price'] * 100,
    axis=1,
)
# Row-wise: r is a Series with column names as keys.
```

**Gotcha.** `axis=1` does NOT mean "vectorized across rows" — it means "Python function called once per row." For numeric work, a vectorized expression (`(df['a'] - df['b']) / df['a']`) is dramatically faster.

#### `sort_values` — sort rows by one or more columns

> **🪜 Mental model:** *ORDER BY in pandas.* Reorders rows by the values of one or more columns; doesn't change which rows exist, only their order.

**What it is.** `df.sort_values(by='col')` returns a new DataFrame with rows reordered by the values in `col`. With a list of columns (`by=['a', 'b']`), pandas sorts by `a` first, breaks ties using `b`. The `ascending=` parameter controls direction and accepts a list when `by` is a list (one bool per column).

**Why it matters.** Top-N analysis ("top 10 products by revenue"), trend analysis, and any chronological inspection start with a sort. SQL `ORDER BY` maps directly here. Sorting also matters before `.head()` / `.tail()` — without an explicit sort, those return arbitrary rows.

**How it works.** Pandas uses a stable, NumPy-backed sort under the hood (Timsort for object dtypes, quicksort variant for numerics). NaNs go to the end by default (`na_position='last'`); pass `na_position='first'` to flip. By default it returns a new DataFrame; pass `inplace=True` to mutate.

**Where it's used.** Every "top N by X" query. Pre-aggregation ordering ("most-recent first"). Before computing rank, percentile, or cumulative quantities. Plot preparation (line plots need x-axis in order).

**Related terms.**
- **`sort_index()`** — sort by the index instead of a column.
- **`nlargest(n, 'col')` / `nsmallest`** — faster than `sort_values(...).head(n)` for top-N because it doesn't sort the whole table.
- **`rank()`** — closely related; assigns rank numbers based on sort order.
- **Stability** — a stable sort preserves tie-breaker order from the previous sort; pandas' sort is stable by default.

```python
df.sort_values(by=['votes', 'rating'], ascending=[False, True])
# Most votes first; ties broken by lowest rating
```

**Gotcha.** Length mismatch: if `by` is a single column, `ascending` must be a scalar bool, not a list. `ascending=[True]` with `by='col'` raises `ValueError`.

#### Duplicates — `duplicated` and `drop_duplicates`

> **🪜 Mental model:** *Find repeats, then decide.* `duplicated()` tells you **which** rows are duplicates (boolean mask); `drop_duplicates()` actually removes them.

**What it is.** `df.duplicated(subset=None, keep='first')` returns a boolean Series — True where the row's values (in the given subset of columns) have appeared earlier. `df.drop_duplicates(subset=None, keep='first', inplace=False)` returns a new DataFrame with duplicates removed. The `keep=` parameter chooses which copy to keep: `'first'` (default), `'last'`, or `False` (drop **all** copies).

**Why it matters.** Duplicates corrupt every downstream metric — `count()` is wrong, joins explode, group-bys double-count. Almost every messy CSV has accidental dupes (multiple ETL runs, manual edits). The very first cleanup step on real data is checking for them.

**How it works.**
1. Pandas computes a hash of each row (using `subset=` columns if given, else all columns).
2. Walks the rows; on each row, checks whether its hash has been seen before.
3. With `keep='first'`, the first occurrence is marked False (not a dup); all later ones are True.
4. With `keep=False`, *every* row that has any duplicate is marked True — useful when you want to inspect or remove all members of duplicated groups.

**Where it's used.** Data-quality audit: `df.duplicated().sum()` as the very first sanity check. Deduping orders that ran through ETL twice. Cleaning a join result where the join key wasn't unique.

**Related terms.**
- **`subset=`** — limit which columns count toward "duplicate." Without it, all columns are compared.
- **`keep='first' / 'last' / False`** — which copy to keep; `False` drops them all.
- **`nunique()` vs `count()`** — using these to detect duplicates without explicitly listing.
- **`validate=` on `merge`** — pre-emptive duplicate guard at join time.

```python
df.duplicated(subset=['name', 'votes']).sum()              # how many dupes?
df = df.drop_duplicates(subset=['name', 'votes'], keep='first')
```

**Gotcha.** `drop_duplicates` returns a new DataFrame by default — `df.drop_duplicates()` alone has no effect; you must reassign or pass `inplace=True`.

#### `pd.concat` — stack DataFrames with no key alignment

> **🪜 Mental model:** *Stack the boxes.* Glue DataFrames together vertically (more rows) or horizontally (more columns), without trying to match on a key.

**What it is.** `pd.concat([a, b], axis=0)` stacks DataFrames row-wise (more rows; columns must match by name); `axis=1` stacks them column-wise (more columns; aligned by index). Unlike `merge`, there's no key — concat is a positional / index-based operation, not a relational one.

**Why it matters.** When you have multiple files of the same schema (one CSV per month), you `concat` them into one big DataFrame for analysis — there's no key to join on, you just want them stacked. Confusing `concat` with `merge` is one of the top pandas interview traps; they answer different questions.

**How it works.**
1. **`axis=0` (row-wise)** — pandas takes the union of column names across the inputs, lines them up, and stacks rows. Missing columns get NaN.
2. **`axis=1` (column-wise)** — pandas aligns on the **index** (not on values), then sticks columns side by side. Rows whose index appears in only one DataFrame get NaN on the other side.
3. `ignore_index=True` resets the row index to a fresh 0..n-1 (essential when stacking — otherwise indexes collide).

**Where it's used.** Combining monthly CSV files. Appending new rows (single-row DataFrame + `concat` is the idiomatic replacement for the deprecated `.append()`). Side-by-side comparison of multiple summary tables.

**Related terms.**
- **`pd.merge`** — the keyed cousin; aligns on a column value, not on index position.
- **`ignore_index=True`** — reset row index after stacking.
- **`.append()`** — deprecated; use `concat` for the same purpose.
- **`axis=0` vs `axis=1`** — rows vs columns; same axis convention as NumPy.

```python
combined = pd.concat([jan_df, feb_df, mar_df], axis=0, ignore_index=True)
# All three months stacked; index renumbered 0..N
```

**Gotcha.** `concat([a, b], axis=1)` aligns on the **index**, not on rows in position. If `a` and `b` were sorted differently, you'll get scrambled (NaN-filled) pairings. Reset the index first if you want positional alignment.

#### `isnull` / `isna` — count missing cells

> **🪜 Mental model:** *Missing-value detector.* Returns a boolean DataFrame the same shape as the input — True where the cell is `NaN` / `None` / `NaT`.

**What it is.** `df.isnull()` and `df.isna()` are **identical** — pandas exposes both names for readability. Each returns a same-shape boolean DataFrame where True marks a missing value. The idiomatic next step is `.sum()` to count missing per column: `df.isna().sum()`.

**Why it matters.** "How much missing data?" is the single most-asked EDA question. Models don't accept NaN by default (sklearn raises), so you must quantify and handle every NaN before training. The `.isna().sum()` idiom is so common it's basically the second line of every EDA notebook.

**How it works.** Pandas treats three things as "missing": `np.nan` (float NaN), Python `None`, and `pd.NaT` (Not-a-Time, for datetime columns). `.isna()` walks the data and marks each cell True/False against this set. Pandas-2 nullable dtypes (`Int64`, `Float64`, `boolean`, `string`) also use `pd.NA` as the missing sentinel and `.isna()` catches it.

**Where it's used.** The first sanity check on any new dataset (`df.isna().sum()`). Picking imputation strategy per column (drop if rare, fill with mean/median/mode if recoverable). Building a "missingness mask" feature for tree models that benefit from knowing a value was missing.

**Related terms.**
- **`notna()` / `notnull()`** — opposite mask (True where present).
- **`fillna()`** — replace missing with a constant or computed value.
- **`dropna()`** — remove rows/cols that have missing values.
- **`NaN`** — float-only sentinel; legal in float dtype, not in int.
- **`NaT`** — datetime version of NaN.

```python
df.isna().sum().sort_values(ascending=False)
# Per-column NaN count, biggest offender first
```

**Gotcha.** `isna()` does NOT catch the **string** `'NaN'`, `'None'`, or `''` — those are real strings, not missing. Convert with `pd.to_numeric(..., errors='coerce')` first to surface them.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#1g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **`pd.merge`** | SQL-style JOIN on a shared key column. `how=` controls which side's non-matching rows survive (inner/outer/left/right). | Whenever data lives in two tables that share an ID — orders+products, users+events. The first step of almost every analysis. |
| **Inner join** | Keep only rows whose key matches on **both** sides; drop everything else. Result ≤ min(left, right). | When you want only "complete" records and don't care about orphans. SQL default. |
| **Outer join** | Keep **every** key from either side; NaN-fill columns where the other side has no match. Result = union of keys. | Data-quality audits ("what's missing on either side?"). Combine with `indicator=True` to see where rows came from. |
| **Left join** | Keep **all** of left; bolt on matching info from right (NaN where no match). Result ≥ len(left). | The most common join in practice — "enrich my fact table with attributes." |
| **Right join** | Mirror of left — keep all of right. Rare; usually rewritten as "swap args and use `how='left'`." | Almost never — but FAANG vocab quizzes ask about it. |
| **`groupby`** | Lazy split-apply-combine: split rows into bins by key, run a function in each bin, glue results back. | Every "per-X" aggregation: per-category mean, per-day count, per-user revenue. Maps 1:1 to SQL `GROUP BY`. |
| **`agg`** | One call that runs multiple aggregations on one or many columns. Named-aggregation form (`name=('col','fn')`) is preferred. | When a report needs more than one summary stat (mean + max + count). Replaces three separate groupbys. |
| **`apply`** | Universal escape hatch — runs a Python function per Series element, per DataFrame row/column, or per group. | Custom logic that no built-in covers. **Slow** — use vectorized ops when possible. |
| **`sort_values`** | `ORDER BY` in pandas — reorders rows by one or more columns; `ascending=` accepts a list when sorting by many. | Top-N analysis, trend prep, plot ordering. Pair with `.head(n)` or use `nlargest` for top-N efficiency. |
| **`duplicated` / `drop_duplicates`** | `duplicated()` marks repeats with a boolean mask; `drop_duplicates()` removes them. `keep=False` drops every copy. | First sanity check on any new dataset (`df.duplicated().sum()`). Cleanup after a merge with non-unique keys. |
| **`pd.concat`** | Stack DataFrames vertically (`axis=0`, more rows) or horizontally (`axis=1`, more columns). No key alignment. | Combining monthly CSVs into one. Appending a single new row. Side-by-side comparison of summary tables. |
| **`isnull` / `isna`** | Identical methods — boolean mask of missing cells. Pair with `.sum()` for per-column NaN count. | The first line of EDA after loading data. Quantifies how much cleanup or imputation is needed. |
| **`.iloc[i:j, m:n]`** | Position-based rectangular slice; end-exclusive, same as Python slicing. | When you know the row/column **positions**, not their labels. |
| **`.loc[mask, 'col'] = v`** | Single-step conditional assignment using a boolean mask. Avoids `SettingWithCopyWarning`. | Any "set column X to value V where condition is true." The chained-indexing alternative is buggy. |
| **`groupby().filter(fn)`** | Keep ROWS belonging to groups whose function returns True. Returns a DataFrame of original rows, not groups. | "Keep only users who placed >5 orders" — filtering at the group level, surfacing original rows. |

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

<a id="2g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 2 concept. The cheat sheet below is the recap surface.

#### Missing values overview — `NaN` in Pandas

> **🪜 Mental model:** *A blank cell with a name.* `NaN` ("Not a Number") is the placeholder pandas writes whenever a value is missing — empty CSV cells, parse failures, missing keys after an outer join.

**What it is.** Missing data appears in pandas as one of three sentinels: `np.nan` (float `NaN`, by far the most common), Python `None`, or `pd.NaT` (Not-a-Time, for datetime columns). All three are caught by `.isna()` / `.isnull()`. Pandas-2's nullable dtypes also use `pd.NA`. Importantly, `NaN` is a **float-only** value — an integer column with missing values gets promoted to float (or to the new nullable `Int64` dtype) to make room for NaN.

**Why it matters.** Real-world data has missing values. Models (sklearn estimators) refuse NaN input and raise. Math on NaN is contagious — `5 + NaN == NaN`, `[1, 2, NaN].mean() == 1.5` (NaN is skipped by default in pandas, but propagates in numpy). Mishandling missing values is the #1 source of subtle EDA bugs.

**How it works.** During parsing (`pd.read_csv`), pandas converts a configurable list of strings (`""`, `"NA"`, `"NULL"`, …) to NaN. After loading, NaN behaves like a normal value for indexing and selection, but most aggregations (`mean`, `sum`, `count`) skip NaN by default (`skipna=True`). Boolean comparisons against NaN return False (`NaN == NaN` is False — use `isna()` instead).

**Where it's used.** EDA's first call after `df.shape` is `df.isna().sum()`. Outer joins, reindex, pivot, and concat all introduce NaN where alignment fails. Before model training, every NaN must either be dropped (`dropna`) or filled (`fillna`).

**Related terms.**
- **`NaN`** vs **`None`** vs **`NaT`** — three names for "missing"; pandas treats them uniformly.
- **`pd.NA`** — the new universal missing-value sentinel in pandas-2 nullable dtypes.
- **`skipna=True`** — default in most pandas aggregations; flip to False to make NaN propagate.
- **Imputation** — replacing NaN with mean/median/mode/sentinel.
- **MNAR / MAR / MCAR** — academic taxonomy of *why* values are missing (Missing Not At Random / At Random / Completely At Random); affects which imputation is honest.

```python
df.isna().sum()                # per-column NaN counts
df['rating'].mean()            # skips NaN by default
```

**Gotcha.** `NaN != NaN` returns True (NaN is not equal to itself). Never use `==` to check for missing — use `.isna()`.

#### `isna` / `notna`

> **🪜 Mental model:** *"Is this cell empty?"* — a boolean mask the same shape as your data.

**What it is.** `df.isna()` returns a same-shape boolean DataFrame, True where the cell is missing. `df.notna()` is the opposite — True where the cell has a real value. `isnull` / `notnull` are exact aliases (the names exist for SQL-style readability).

**Why it matters.** Every missing-data workflow starts with these two: count, locate, filter. `df.isna().sum()` per-column is the standard "how clean is my data?" probe. `df[df['col'].notna()]` keeps only rows where a critical column is present.

**How it works.** Pandas walks the DataFrame and checks each cell against the missing-value sentinels (`np.nan`, `None`, `NaT`, `pd.NA`). Returns a boolean DataFrame of the same shape. The result composes naturally with boolean indexing.

**Where it's used.** EDA's per-column NaN audit. Pre-filter before `dropna(subset=…)`. Building a "missing indicator" feature for ML models that can use it.

**Related terms.**
- **`notna()`** — opposite mask.
- **`isnull()` / `notnull()`** — exact aliases.
- **`.any()` / `.all()`** — chain after `isna()` to ask "any NaN in this row/col?" (`df.isna().any(axis=1)`).
- **`fillna()`** — the next-step partner.

```python
df.isna().sum()                                    # per-column counts
df[df['order_ts'].notna()]                         # rows with non-null timestamp
df.loc[df.isna().any(axis=1)]                      # any row with at least one NaN
```

**Gotcha.** `isnull` and `isna` are identical — pandas exposes both names; choose one and stay consistent.

#### `dropna` — remove rows/cols with missing data

> **🪜 Mental model:** *Kick out the rows with holes.* `dropna()` removes any row (default) or column that has missing values.

**What it is.** `df.dropna(axis=0, how='any', subset=None, thresh=None, inplace=False)` returns a new DataFrame with rows (or columns) removed based on missing-value rules. Each parameter controls exactly which:
- **`axis=0`** (default) drops rows; **`axis=1`** drops columns.
- **`how='any'`** drops if **any** value is NaN; **`how='all'`** drops only if **all** values are NaN.
- **`subset=[…]`** restricts the NaN check to a subset of columns — drop only when those are missing, ignore NaNs elsewhere.
- **`thresh=k`** keeps rows that have **at least `k` non-NaN values**; lets you tolerate sparse rows.

**Why it matters.** When missingness is rare and the missing rows aren't recoverable, dropping is the cleanest, most honest fix — you don't add fake values. But blanket `dropna()` can silently delete most of your data if any column is mostly NaN. The four parameters above let you drop precisely.

**How it works.** Pandas builds the NaN mask, applies the rule (any/all/thresh), and returns the surviving rows or columns. By default it returns a new DataFrame; pass `inplace=True` to mutate in place. Order is preserved.

**Where it's used.** Removing rows missing a critical field (`subset=['order_id']`). Pruning columns that are >50% NaN (`axis=1, thresh=len(df)//2`). After parsing dates with `errors='coerce'`, drop rows where the date failed to parse (`subset=['date']`).

**Related terms.**
- **`fillna()`** — alternative when you don't want to lose rows.
- **`thresh=k`** — the under-used parameter; the right way to say "keep rows with at most N missing."
- **`subset=`** — the standard way to drop on one critical column.
- **`how='any'` vs `how='all'`** — easy-to-confuse modes.

```python
df.dropna(subset=['order_ts'])                     # drop rows with no timestamp
df.dropna(axis=1, thresh=len(df) * 0.5)            # drop cols ≥50% missing
```

**Gotcha.** `df.dropna()` with no args drops a row if **any** column is NaN — on real data that often deletes >90% of rows. Always pass `subset=` or `thresh=`.

#### `fillna` — replace missing with a value

> **🪜 Mental model:** *Patch the holes.* Replace each NaN with a constant, a per-column value, or a value copied from an adjacent row.

**What it is.** `df.fillna(value)` returns a new DataFrame with NaN replaced. The `value` argument can be:
- A **scalar** (`df.fillna(0)`) — same fill for everything.
- A **dict** (`df.fillna({'rating': df['rating'].mean(), 'discount': 0})`) — per-column fill, most idiomatic for real data.
- A **method** — `method='ffill'` carries the previous non-NaN value forward (forward-fill); `method='bfill'` carries the next backward. These are essential for time series. Pandas 2.x has split these into dedicated methods: `df.ffill()` and `df.bfill()`.

**Why it matters.** When you can't drop missing values (too many would die), you impute them. The right strategy depends on the column type and the business meaning: mean/median for numerics, mode or "Unknown" for categories, 0 for "no discount applied", forward-fill for time series gaps. Choosing badly biases every downstream metric.

**How it works.** Pandas walks the data and replaces each NaN according to the rule. With a scalar, every NaN gets the same value. With a dict, only the named columns are filled; others are left as NaN. With `method='ffill'`, pandas remembers the most recent non-NaN value per column and writes it into the NaN cells until a new non-NaN appears.

**Where it's used.** Numeric imputation: `df['rating'].fillna(df['rating'].mean())`. Categorical: `df['cat'].fillna('Unknown')`. Domain default: `df['discount'].fillna(0)`. Time-series gap filling: `df['price'].ffill()`. Production-grade pipelines wrap this in `sklearn.impute.SimpleImputer` so the train-time mean is reused on test data (preventing leakage).

**Related terms.**
- **`SimpleImputer` (sklearn)** — leak-free version: fit on train, apply to test.
- **`ffill` / `bfill`** — directional fills for ordered data.
- **`interpolate()`** — fill NaN with linearly (or otherwise) interpolated values between known points.
- **Sentinel** — a "magic value" that means "missing on purpose" (0 for "no discount"). Choose carefully — sentinels can be confused with real values.

```python
df = df.fillna({'rating': df['rating'].mean(), 'discount': 0, 'cat': 'Unknown'})
df['price'] = df['price'].ffill()                  # forward-fill for time-series
```

**Gotcha.** `s.fillna(s.mean())` is a **no-op when the entire column is NaN** — `s.mean()` of all-NaN is NaN, so you fill NaN with NaN. Check the column has at least one valid value first.

#### `pd.melt` — wide → long reshape

> **🪜 Mental model:** *Stack many value columns into rows.* If you have a "report-style" table with one column per metric, `melt` rotates those columns into a tall table with one row per (id, metric) pair.

**What it is.** `pd.melt(df, id_vars=[...], value_vars=[...], var_name='variable', value_name='value')` turns a wide table into a long one. Columns listed in `id_vars` stay as-is (they identify the row). Columns in `value_vars` collapse into two new columns: one named by `var_name` (which original column the value came from) and one named by `value_name` (the value itself). The result has many more rows and many fewer columns.

**Why it matters.** Most plotting and grouping libraries (seaborn, sklearn) expect data in **long** form — one row per observation with a "kind" column labelling the metric. Spreadsheet-exported data is usually **wide**. Knowing which direction to reshape (and how) is the single most common pandas-shape question in interviews.

**How it works.** Pandas iterates over each `value_var` column and emits one new row per (row, value_var) pair. If you have 100 rows and 3 value_vars, you get 300 long-form rows. The `id_vars` columns are repeated for each value_var of the same source row.

**Where it's used.** Reshaping monthly columns (`Jan, Feb, Mar`) into a `month` + `value` pair. Preparing data for seaborn (`sns.lineplot(data=long_df, x='month', y='value', hue='metric')`). Tidy-data conversion in any analysis pipeline.

**Related terms.**
- **`pd.pivot_table`** — the opposite direction (long → wide).
- **`id_vars`** — identifier columns to keep unchanged.
- **`value_vars`** — columns to stack; if omitted, every non-id column is melted.
- **`stack()`** — similar reshape, works on a MultiIndex; lower-level.
- **Tidy data** — the formal name for "one variable per column, one observation per row." Long form is tidy.

```python
pd.melt(
    df,
    id_vars=['product_id'],
    value_vars=['actual_price', 'discounted_price'],
    var_name='price_kind',
    value_name='price',
)
```

**Gotcha.** Forget `value_vars` and pandas melts every non-`id_vars` column, which usually creates a useless mega-tall table. Always specify both.

#### `pd.pivot_table` — long → wide reshape with aggregation

> **🪜 Mental model:** *Excel pivot table in pandas.* Pick rows (`index`), columns (`columns`), values (`values`), and an aggregator (`aggfunc`). Pandas reshapes and summarises in one call.

**What it is.** `pd.pivot_table(df, index='row_var', columns='col_var', values='val_col', aggfunc='mean', fill_value=...)` reshapes long data into a wide grid. Each unique value of `row_var` becomes a row; each unique value of `col_var` becomes a column. At each cell, `aggfunc` (default `mean`) is applied to all matching `val_col` values. Missing combinations come back as NaN unless `fill_value` is set.

**Why it matters.** Pivoting is how you turn raw transaction-level data into report tables ("average rating by category × month"). It's groupby + reshape in one step. In interviews, "produce a per-category-per-month average" is a guaranteed pandas question.

**How it works.** Internally, `pivot_table` is a `groupby(['index', 'columns'])` followed by `agg(aggfunc)` followed by `unstack('columns')` — pandas just packages those three steps. With duplicates, `pivot_table` silently aggregates (because `aggfunc` is defined); its stricter sibling `pivot()` would raise.

**Where it's used.** Cross-tabulation reports. Heatmap inputs (`pivot_table` first, then `sns.heatmap`). Time-series matrices (rows = entity, cols = day, values = metric). Confusion-matrix construction.

**Related terms.**
- **`pivot()`** — strict version; raises on duplicate (index, columns) pairs. Useful as a sanity check.
- **`pd.melt`** — opposite direction.
- **`crosstab()`** — pivot's cousin specialised for counts: `pd.crosstab(row, col)`.
- **`unstack()` / `stack()`** — lower-level reshape primitives `pivot_table` uses.
- **`fill_value=0`** — fill missing combinations with 0 instead of NaN (common for count tables).

```python
pd.pivot_table(
    df,
    index='category',
    columns='month',
    values='rating',
    aggfunc='mean',
    fill_value=0,
)
```

**Gotcha.** Don't forget `aggfunc=` — the default is `mean`, which silently produces wrong numbers when you wanted a `sum` or `count`. Always pass `aggfunc=` explicitly to document intent.

#### `pd.cut` / `pd.qcut` — binning continuous values into categories

> **🪜 Mental model:** *Buckets.* Turn a continuous number (price, age, score) into a labelled bucket (Low / Mid / High) so you can group, plot, or model it categorically.

**What it is.** Two functions for binning:
- **`pd.cut(s, bins, labels=...)`** — bins by **value range**. You pass explicit edges (`[0, 5000, 20000, np.inf]`) and the values fall into the corresponding bins. Bin widths are user-defined; counts per bin can be wildly unequal.
- **`pd.qcut(s, q, labels=...)`** — bins by **quantile**. You pass a number of equal-count buckets (`q=4` → quartiles) and pandas finds the cut-points so each bin has the same count. Bin widths are unequal; counts are equal.

**Why it matters.** Models and plots often want categorical buckets, not raw continuous values. "Price band" is more interpretable in a report than "$23,847.55". Quantile-based binning (`qcut`) is also a robust way to handle skewed data — equal counts per bin avoids the "all values in bin 1" trap.

**How it works.**
- `cut` evaluates each value against the bin edges and assigns it to the bin whose range it falls into. Edges are half-open by default — `right=True` (default) makes intervals like `(0, 5000]`.
- `qcut` first computes the quantile cut-points (e.g., for `q=4` it computes the 25th, 50th, 75th percentiles) and then calls `cut` under the hood.

Both return a pandas `Categorical` Series — efficient, ordered if you set `ordered=True`, and groupby-friendly.

**Where it's used.** Price bands for marketing reports. Age buckets for demographic analysis. Decile or quartile analysis ("what's the top-10% spender doing?"). Feature engineering for tree-based models that can use binned features.

**Related terms.**
- **`qcut`** vs **`cut`** — quantile-based vs value-based; same return shape.
- **Categorical dtype** — what binning produces; supports `.cat.categories`, `.cat.codes`.
- **`right=False` / `include_lowest=True`** — fine-grained control over edge handling.
- **`labels=False`** — return the bin index (0, 1, 2, …) instead of label strings.

```python
df['band']    = pd.cut(df['price'], bins=[0, 5_000, 20_000, np.inf], labels=['Low','Mid','High'])
df['quartile'] = pd.qcut(df['rating'], q=4, labels=['Q1','Q2','Q3','Q4'])
```

**Gotcha.** `pd.cut` with edges that don't cover the data range produces NaN for the out-of-range values. Use `bins=[0, ..., np.inf]` to be safe.

#### `.str` accessor — `.contains` / `.extract` / `.split` (regex on string columns)

> **🪜 Mental model:** *Vectorised Python string methods.* Instead of looping with `for s in series: s.contains(...)`, use `series.str.contains(...)` — pandas runs the operation on every element at C speed.

**What it is.** The `.str` accessor is a namespace on string Series. The three most-used methods:
- **`.str.contains(pattern, case=False, na=False)`** — boolean mask: True where the pattern is found. `pattern` is a regex by default (set `regex=False` for literal).
- **`.str.extract(pattern)`** — pulls regex capture groups into new columns. One capture group → one column; multiple → multiple columns.
- **`.str.split(sep, expand=True)`** — splits each string on `sep`. With `expand=True`, splits become new columns; without, you get a list per row.

**Why it matters.** Real-world string columns are messy: product names with embedded brand info, URLs with query parameters, addresses with city/state/zip jammed together. Vectorized `.str.*` lets you extract, classify, and clean without writing Python loops — orders of magnitude faster than `.apply(lambda s: ...)`.

**How it works.** The `.str` accessor dispatches each operation to a compiled C kernel that walks the underlying array. For regex methods, pandas compiles the pattern once and reuses it. `na=False` is the standard "don't blow up on NaN" guard — without it, NaN rows produce NaN in the result mask, which can crash boolean indexing.

**Where it's used.** Filtering by substring: `df[df['name'].str.contains('Pro', case=False, na=False)]`. Extracting structured info from URLs: `df['qid'] = df['url'].str.extract(r'qid=(\d+)')`. Splitting a `"city, state"` column into two. Lowercase/strip cleanups: `df['email'].str.lower().str.strip()`.

**Related terms.**
- **`.str.replace(pat, repl, regex=True)`** — sibling for substitution.
- **`.str.match`** — anchored version of `.str.contains` (must match from the start).
- **`.str.findall`** — returns all matches per cell as a list.
- **`na=False` / `na=True`** — how to treat NaN under boolean operations.
- **Regex** — the pattern syntax most `.str` methods accept by default.

```python
df['has_durable'] = df['about_product'].str.contains('durable', case=False, na=False)
df['qid']         = df['url'].str.extract(r'qid=(\d+)')
df[['city', 'state']] = df['location'].str.split(', ', expand=True)
```

**Gotcha.** Default `regex=True` means special characters (`.`, `*`, `?`, `(`) are interpreted as regex metacharacters. For literal matches, pass `regex=False` — otherwise `.str.contains('1.5')` matches "125" too.

#### `pd.to_datetime` — parse strings into datetimes

> **🪜 Mental model:** *Turn "2024-03-15" the string into 2024-03-15 the date.* After this, you can subtract dates, extract day-of-week, or sort chronologically.

**What it is.** `pd.to_datetime(series, format=..., errors='raise'|'coerce'|'ignore')` parses strings (or numbers, or mixed) into pandas `datetime64[ns]` values. `format=` accepts a `strftime`-style template (e.g., `'%Y-%m-%d'`) and dramatically speeds up parsing because pandas skips its multi-format guessing. `errors='coerce'` turns unparseable values into `NaT` (datetime NaN) instead of raising.

**Why it matters.** Dates loaded from CSV are usually strings — `"2024-03-15"`. As strings they sort lexicographically (`"2024-03-15" < "2024-3-9"` because `"-1" < "-3"`), and you can't compute date arithmetic. Converting to real datetimes unlocks the `.dt` accessor and date math.

**How it works.** Without `format=`, pandas tries multiple parsers per value — slow on big columns. With `format=`, it goes straight to the fast path. The result is a Series of `datetime64[ns]` (8 bytes per value, much smaller than a string). `errors='coerce'` is the standard production setting because real data has parse failures and you want NaT instead of a crash.

**Where it's used.** Right after `pd.read_csv` for any column representing time. Inside `.assign(ts=pd.to_datetime(df['ts']))` for cleanup. Before any time-series analysis or sort-by-time operation. The `parse_dates=['col']` argument to `read_csv` is a shortcut for "parse on load."

**Related terms.**
- **`.dt` accessor** — what you can do *after* converting (see below).
- **`NaT`** — Not-a-Time; pandas' datetime version of NaN.
- **`pd.to_timedelta`** — sibling for durations.
- **`parse_dates=['col']`** — `read_csv` shortcut that calls `to_datetime` under the hood.
- **`format='%Y-%m-%d'`** — `strftime` directives; massive speedup when you know the format.

```python
df['ts'] = pd.to_datetime(df['order_timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
```

**Gotcha.** Without `format=`, pandas may parse `01-02-2024` as Jan 2 (US) or Feb 1 (UK) depending on locale — silently wrong. Always pass `format=` on production data.

#### `.dt` accessor — extract datetime components

> **🪜 Mental model:** *Date-component namespace.* After `to_datetime`, `.dt` exposes `.year`, `.month`, `.day_name()`, `.hour`, `.dayofweek`, … — one call per piece you want to pull out.

**What it is.** `.dt` is to datetime Series what `.str` is to string Series — a namespace of methods/properties that operate elementwise. Common members:
- **`.dt.year`, `.dt.month`, `.dt.day`** — integer components.
- **`.dt.dayofweek`** — 0 (Monday) to 6 (Sunday).
- **`.dt.day_name()`** — string like `"Monday"`.
- **`.dt.isocalendar().week`** — ISO week number (replaces deprecated `.dt.week`).
- **`.dt.strftime('%B %Y')`** — format as a custom string ("March 2024").
- **`.dt.normalize()`** — strip the time-of-day (set to 00:00:00).

**Why it matters.** Time-based features ("is this a weekend?", "which month?", "which hour of day?") are crucial for analytics and ML. Extracting them in pandas is a one-liner; doing it by hand with string manipulation is error-prone (leap years, time zones, ISO weeks).

**How it works.** The `.dt` accessor dispatches each call to a compiled NumPy datetime kernel. Returns a Series of the requested dtype (int, str, or Categorical depending on the method). Most properties are zero-cost — they're just reading the underlying epoch-nanoseconds integer.

**Where it's used.** Feature engineering for time-series models (hour-of-day, day-of-week as features). Grouping by month/quarter (`groupby(df['ts'].dt.month)`). Filtering to weekends (`df[df['ts'].dt.dayofweek >= 5]`). Building human-readable labels (`dt.strftime('%B %Y')`).

**Related terms.**
- **`.dt.dayofweek`** — Monday=0, Sunday=6 (Python convention, NOT the SQL convention).
- **`.dt.isocalendar()` returns a DataFrame** with `year`, `week`, `day` columns.
- **`.dt.tz_localize` / `.dt.tz_convert`** — timezone handling.
- **`.str` accessor** — the equivalent namespace for string Series.
- **`pd.PeriodIndex` / `pd.DatetimeIndex`** — specialised index types for time series.

```python
df['hour']  = df['ts'].dt.hour
df['dow']   = df['ts'].dt.day_name()
df['week']  = df['ts'].dt.isocalendar().week
df['label'] = df['ts'].dt.strftime('%B %Y')          # "March 2024"
```

**Gotcha.** `.dt.dayofweek` returns Monday=0 (Python), but SQL `DAYOFWEEK` returns Sunday=1 — different conventions. State the convention in the column docstring.

#### Univariate visualisation — histogram and count plot

> **🪜 Mental model:** *Two plots for one variable.* **Histogram** is for **continuous** values (binned into ranges). **Count plot** is for **categorical** values (one bar per unique value).

**What it is.** Two seaborn (and matplotlib) plots that visualise the distribution of a single variable:
- **`sns.histplot(data=df, x='col', bins=20, kde=True)`** — bins a continuous variable into ranges and draws one bar per bin showing how many rows fall in that range. `kde=True` overlays a kernel-density curve (smoothed histogram).
- **`sns.countplot(data=df, x='col')`** — counts how many rows have each unique categorical value and draws one bar per category. Equivalent to `df['col'].value_counts().plot.bar()`.

**Why it matters.** Univariate plots are the first visual probe of every column — they reveal skew, multimodality, outliers, and unexpected categories before you build anything on top. Choosing histogram vs countplot wrong (e.g., histogramming a category) produces a confusing plot and signals "I don't actually inspect my data."

**How it works.**
- `histplot` sorts values into bins (range based on min/max, edges based on `bins=` count) and counts per bin. With `kde=True`, it additionally fits a Gaussian kernel-density estimate and overlays it.
- `countplot` runs a `value_counts()` internally and draws bars.

Both accept `hue=` to add a categorical 3rd dimension (overlay multiple distributions).

**Where it's used.** First EDA pass on every dataset: histogram each numeric column, count-plot each category. Decide if a numeric needs `log` transformation (skewed histogram), if a category has too many levels (lopsided countplot), or if there are surprise NaN gaps.

**Related terms.**
- **`sns.kdeplot`** — KDE only, no bars.
- **`sns.ecdfplot`** — empirical cumulative distribution; alternative to histogram.
- **`bins=` parameter** — too few hides shape; too many turns noise into signal. Try 20–50 to start.
- **Bivariate** — when you want **two** variables, see Module 3.

```python
sns.histplot(data=df, x='rating', bins=20, kde=True, color='skyblue')
sns.countplot(data=df, x='category', palette='coolwarm')
plt.xticks(rotation=45)
```

**Gotcha.** Histogramming a categorical column (especially numeric IDs) produces a useless near-flat plot. Always check dtype first: `df['col'].dtype` and `df['col'].nunique()`.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#2g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Missing values (`NaN`)** | Pandas' placeholder for "no value"; appears as `np.nan` (float), `None`, or `NaT` (datetime). Math on NaN is contagious unless `skipna=True` (the default). | Every dataset has them — quantify with `isna().sum()`, then drop or impute before modelling. |
| **`isna` / `notna`** | Identical-method twins. `isna()` is a boolean mask of missing cells; `notna()` is its opposite. `isnull` / `notnull` are aliases. | The first EDA call after `df.shape`. Pair with `.sum()` for per-column NaN audit. |
| **`dropna`** | Removes rows (or cols) with NaN. Parameters: `axis`, `how='any'/'all'`, `subset=[…]`, `thresh=k`. | When missingness is rare and the row isn't recoverable. Default drops too aggressively — always pass `subset=`. |
| **`fillna`** | Replaces NaN with a scalar, per-column dict, or directional fill (`ffill`/`bfill`). | When you can't afford to drop data. Pick the strategy per column: mean for numeric, "Unknown" for categorical, 0 for "no discount". |
| **`pd.melt`** | Wide → long reshape. Stacks `value_vars` into one `value` column tagged with `var_name`. | Preparing data for plotting/grouping when your source is "report-style" with one column per metric. |
| **`pd.pivot_table`** | Long → wide reshape + aggregation. Pick `index`, `columns`, `values`, `aggfunc`. | Cross-tabulation reports; heatmap inputs; "average X per row-key per col-key" tables. |
| **`pd.cut` / `pd.qcut`** | Binning continuous values into discrete categories. `cut` uses explicit value edges; `qcut` uses equal-count quantiles. | "Low/Mid/High" labels (`cut`), or quartile/decile analysis (`qcut`). Inputs to groupby and categorical plots. |
| **`.str` accessor** | Vectorized string ops — `contains`, `extract`, `split`, `replace`, `lower`, `strip`. Regex by default. | Substring filtering, extracting structured info from URLs/text, splitting concatenated columns. |
| **`pd.to_datetime`** | Parse a string column to `datetime64[ns]`. Pass `format=` for speed; `errors='coerce'` for safety. | Right after `read_csv` for any time column. Unlocks the `.dt` accessor and date arithmetic. |
| **`.dt` accessor** | Datetime-component namespace — `.dt.year`, `.dt.month`, `.dt.dayofweek`, `.dt.day_name()`, `.dt.strftime(...)`. | Feature engineering for time series; grouping by month/week; weekend vs weekday filters. |
| **Histogram** | Bins a **continuous** variable into ranges and bars one bar per range. `kde=True` overlays a smooth curve. | Inspecting a numeric column's shape — skew, multimodality, outliers, gaps. |
| **Count plot** | One bar per unique value of a **categorical** column. Equivalent to `value_counts().plot.bar()`. | Inspecting category frequency — finding rare classes, imbalance, surprise levels. |
| **`pivot` vs `pivot_table`** | `pivot()` is strict (raises on duplicate (index, col) pairs); `pivot_table()` silently aggregates them. | Use `pivot_table` on real data; reach for `pivot` as a "this should be unique" sanity check. |

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

<a id="3g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every Module 3 concept. Each plot is paired with the *question it answers* — pick by question, not by chart name. The cheat sheet below is the recap surface.

#### Scatter plot — numeric × numeric

> **🪜 Mental model:** *Cloud of dots, one per row.* Each point's `x` is one numeric column, `y` is another; the cloud's shape reveals whether the two are related.

**What it is.** A scatter plot draws one dot per row at position `(x, y)`, where `x` and `y` are two numeric columns. In seaborn: `sns.scatterplot(data=df, x='price', y='rating')`. Optional dimensions can be encoded via `hue=` (color by category), `size=` (size by numeric), or `style=` (marker shape).

**Why it matters.** Scatter is the canonical plot for **relationship between two numerics**. Before you trust a correlation number, you should look at the scatter — the correlation might be 0.8 because of three outliers, or 0.0 because the relationship is non-linear. "Always plot, don't just compute" is a senior-engineer reflex.

**How it works.** Seaborn iterates the rows of the DataFrame and calls matplotlib's `plt.scatter` once with the full arrays. With `hue=`, it splits by category and plots each subgroup separately so each gets a legend entry. `alpha=` (transparency) is essential when points overlap — without it, dense regions show only the topmost dot.

**Where it's used.** EDA's "is X related to Y?" probe. Anscombe's quartet (the classic counterexample) is the canonical case for "look at the scatter, not the correlation". Outlier hunting. Sanity-check before regression: if the cloud is curved, linear regression is wrong.

**Related terms.**
- **`alpha`** — transparency (0–1); essential against overplotting.
- **`hue`** — categorical color encoding for a 3rd dimension.
- **2D KDE / hexbin** — alternatives for millions of points where individual dots stop helping.
- **Line plot** — sibling for *ordered* x (time series); don't confuse the two.
- **Pearson correlation** — the number summarising scatter shape (only for linear shapes).

```python
sns.scatterplot(data=df, x='price', y='rating', hue='category', alpha=0.4)
```

**Gotcha.** Plotting a million rows as individual dots creates one black blob. Either subsample, set `alpha=0.05`, or switch to a 2D density (`sns.kdeplot` / `plt.hexbin`).

#### Line plot — time series or ordered numeric

> **🪜 Mental model:** *Connect the dots in x-order.* Same as scatter but with line segments joining consecutive points — only meaningful when `x` has a natural order (time, position, rank).

**What it is.** `sns.lineplot(data=df, x='month', y='value', marker='o')` draws one line connecting points sorted by `x`. With multiple groups (via `hue=` or by passing a long-form DataFrame), one line per group is drawn. Seaborn auto-aggregates duplicates (averaging y per x by default, with a 95% confidence-interval band).

**Why it matters.** Line plots are the standard for **trends over time** — daily sales, monthly users, hourly latencies. They communicate direction (going up or down?) and turning points faster than any other plot. Using a line plot on **unordered** data is a beginner red flag.

**How it works.** Seaborn sorts the data by `x`, computes the per-group aggregate (default: mean) at each unique `x`, draws a line connecting them, and overlays a confidence band (default 95% CI from bootstrap). Pass `errorbar=None` to suppress the band.

**Where it's used.** Time-series trends (`x='date', y='value'`). Learning curves in ML (`x='epoch', y='loss'`). Performance comparisons across an ordered parameter. Side-by-side trends with `hue=` for multiple cohorts.

**Related terms.**
- **Scatter plot** — same `(x, y)` data; no line; use when x is unordered.
- **`errorbar=None` / `errorbar='ci'`** — control the CI band.
- **`marker='o'`** — explicit markers at every data point (useful for sparse x).
- **Time series** — the typical use case; needs `pd.to_datetime` on x first.

```python
sns.lineplot(data=monthly, x='month', y='rating', marker='o')
plt.xticks(rotation=45)
```

**Gotcha.** Line plots imply continuity. If your x is categorical and unordered (like `'Apparel'`, `'Books'`), don't use a line plot — switch to a bar plot.

#### Box plot — categorical × numeric (5-number summary per group)

> **🪜 Mental model:** *The 5-number summary, drawn.* For each category, draw the median, the box (Q1–Q3), the whiskers (extending to 1.5× IQR), and outliers as dots.

**What it is.** `sns.boxplot(data=df, x='category', y='rating')` draws one box per category, summarising that group's distribution into five numbers: minimum (whisker bottom), 1st quartile (box bottom), median (line inside box), 3rd quartile (box top), maximum (whisker top). Points beyond 1.5× the **interquartile range (IQR = Q3 − Q1)** are drawn as outlier dots.

**Why it matters.** Box plots compress a distribution into a tiny, comparable shape — perfect for side-by-side group comparisons. They're the standard way to ask "does rating differ across categories?" in one chart. They also highlight outliers without losing the central tendency.

**How it works.** For each category in the `x` column, seaborn computes the quartiles of `y`, draws the box from Q1 to Q3, marks the median, extends whiskers to the data point nearest 1.5×IQR from the box, and dots anything beyond. The IQR rule is purely visual — it's not a formal outlier test, just a quick eye-catch.

**Where it's used.** Group comparison ("rating per category"). Outlier inspection during EDA. Pre-modelling check that the target distribution is balanced across classes. A/B test results: box per variant.

**Related terms.**
- **Violin plot** — sibling that *also* shows distribution shape (see below).
- **IQR (interquartile range)** — Q3 − Q1; the width of the box.
- **Outlier (visual)** — point beyond 1.5×IQR; not a statistical definition.
- **5-number summary** — min, Q1, median, Q3, max; what a boxplot draws.

```python
sns.boxplot(data=df, x='category', y='rating')
plt.xticks(rotation=45)
```

**Gotcha.** A box plot hides multimodality — two groups with very different shapes can have identical boxes. When shape matters, use a violin plot.

#### Violin plot — categorical × numeric with distribution shape

> **🪜 Mental model:** *Box plot + smooth density.* You see the 5-number summary **and** the shape (bimodal? skewed?) of the distribution at the same time.

**What it is.** `sns.violinplot(data=df, x='category', y='rating')` is a box plot with a kernel-density estimate (KDE) mirrored on both sides — the "violin" silhouette is wider where data is dense, narrower where it's sparse. Inside the violin, you can also see the box plot summary.

**Why it matters.** A box plot summarises into 5 numbers, throwing away shape. If you suspect bimodality ("ratings cluster at 1 and 5, not 3") or skew, a violin plot reveals it where a box plot can't.

**How it works.** Seaborn computes a Gaussian KDE on the `y` values per category, then mirrors the KDE around the vertical category axis. The horizontal width of the violin at any y-value is proportional to the density at that point. Combined with the inner box plot summary, you get summary + shape in one figure.

**Where it's used.** Distribution comparison across groups when shape matters. Quality check that an "average rating" isn't hiding bimodal love-it-or-hate-it behavior. Often paired with box plots side-by-side in a 1×2 subplot grid.

**Related terms.**
- **Box plot** — simpler sibling; use when shape doesn't matter.
- **KDE (Kernel Density Estimate)** — the smoothed density curve underlying the violin.
- **`bw_method=` / `bw_adjust=`** — bandwidth controls; smooths the violin more or less.
- **Bimodal distribution** — two peaks; visible in violin, hidden in box.

```python
sns.violinplot(data=df, x='category', y='rating', inner='box')
```

**Gotcha.** Violins are computationally heavy on big data. With millions of rows, sample first.

#### Grouped bar chart — categorical × numeric aggregate

> **🪜 Mental model:** *Bars per category showing an aggregated number.* By default seaborn's `barplot` shows the **mean** of `y` per category, with a 95% CI bar.

**What it is.** `sns.barplot(data=df, x='category', y='rating', estimator=np.mean)` draws one bar per category whose height is the chosen aggregate (default mean, change with `estimator=`). The thin black line on top is a 95% confidence interval. With `hue=`, bars are split into sub-bars per hue category, producing a grouped bar chart.

**Why it matters.** When you want a clean, simple "X per category" report — average revenue per region, conversion rate per variant — a bar plot is the cleanest visual. It's more compact than a box/violin and answers the headline question directly.

**How it works.** Seaborn groups the data by `x` (and `hue=` if given), computes `estimator` (mean by default), and draws a bar of that height. The error bar comes from a bootstrap of the per-group mean (95% CI by default; disable with `errorbar=None`).

**Where it's used.** "Average rating per category." "Conversion rate per A/B variant." "Revenue per region split by month (`hue='month'`)." Executive dashboards. Almost any "compare a number across categories" question.

**Related terms.**
- **`estimator=`** — change to `np.median`, `np.sum`, or a custom function.
- **`countplot`** — special-case bar plot where `y` is implicitly the count.
- **`errorbar=`** — `'ci'` (95% CI), `'sd'` (standard deviation), or `None`.
- **Grouped bar (via `hue=`)** — multi-level comparison.

```python
sns.barplot(data=df, x='category', y='rating', hue='month', estimator='mean')
```

**Gotcha.** A bar plot showing the **mean** with a small CI band hides the distribution. If the spread matters, box/violin is better; if the count matters, use `countplot`.

#### Pair plot — matrix of all pairwise scatter plots

> **🪜 Mental model:** *Pairwise relationships at a glance.* For N numeric columns, draw an N×N grid: scatter for off-diagonal pairs, distribution plot on the diagonal.

**What it is.** `sns.pairplot(df, hue='category', diag_kind='kde')` plots every numeric column against every other. Off-diagonal cells show scatter plots; diagonal cells show the univariate distribution of that column. With `hue=`, points and distributions are colored by category. `corner=True` shows only the lower triangle (since the matrix is symmetric).

**Why it matters.** Pair plots are the fastest first-look at relationships in a small numeric dataset. You see all pairwise scatters, all marginal distributions, and (with hue) the per-class structure — in one figure. It's the standard "quick first-pass" plot in EDA.

**How it works.** Seaborn iterates over all `N×N` cell positions, plotting a scatter for off-diagonal `(i, j)` cells with column `i` on x and column `j` on y, and a KDE/histogram on the diagonal. It's basically `for i in cols: for j in cols: subplot(...)`.

**Where it's used.** EDA on small numeric datasets (5–8 columns). Before fitting a regression: do features look linearly related to the target? Class separability check (with `hue='label'`).

**Related terms.**
- **`corner=True`** — show only the lower-triangle (avoid the duplicated upper half).
- **Heatmap of correlation** — the "compressed" alternative for many columns.
- **`diag_kind='kde' / 'hist'`** — distribution plot type on the diagonal.
- **Scatter plot matrix / SPLOM** — the same idea in matplotlib / R.

```python
sns.pairplot(df[['price','rating','count']].dropna(), diag_kind='kde', corner=True)
```

**Gotcha.** Pair plots scale O(N²) — with 30 numeric columns you get 900 subplots and a multi-minute render. Cap at ~8 columns; for more, use a correlation heatmap.

#### Heatmap — matrix of values

> **🪜 Mental model:** *Color-coded grid.* Each cell of a 2D matrix gets a color from a colormap, scaled by the cell's value.

**What it is.** `sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)` takes a 2D matrix (a DataFrame or 2D array) and draws it as a grid of colored cells. With `annot=True`, the numeric values are overlaid; `fmt='.2f'` controls their formatting. `cmap` picks the colormap; `center=0` makes diverging colormaps (red-blue) symmetric around zero.

**Why it matters.** When you have many pairs of related values (a correlation matrix, a confusion matrix, a pivot table), a heatmap shows the **whole structure at once**. The eye picks up patterns (blocks of red, single hot cells) faster than reading numbers.

**How it works.** Each cell is filled with the color from `cmap` at the position `(value - vmin) / (vmax - vmin)`. With `annot=True`, each cell's value is also written in text. The colorbar on the right shows the value-to-color mapping.

**Where it's used.** Correlation matrices (`df.corr()` → heatmap). Confusion matrices in ML evaluation. Pivot tables of "average X per row-key per col-key". Cross-tabs.

**Related terms.**
- **`cmap`** — colormap; `'coolwarm'` / `'RdBu_r'` for diverging (around 0), `'viridis'` / `'Blues'` for sequential.
- **`center=0`** — symmetric scale around 0 (essential for correlation).
- **`annot=True`** — show numbers in cells.
- **`fmt='.2f'`** — number formatting for the annotations.
- **Confusion matrix** — the ML-evaluation heatmap of predictions vs truths.

```python
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
```

**Gotcha.** Without `center=0`, a correlation heatmap's color scale is unbalanced and small negatives look like nothing. Always set `center=0` for correlations.

#### Stacked bar / 100% stacked bar

> **🪜 Mental model:** *Composition per category.* Each x-axis bar is one category; the bar is split into colored segments showing the composition by a second variable. 100%-stacked rescales each bar to total 100% so you compare proportions, not absolute totals.

**What it is.** `df.groupby(['a','b']).size().unstack().plot(kind='bar', stacked=True)` produces a stacked bar where each x is a unique value of `a` and the bar is segmented by `b`. The total height of each bar is the sum across `b`. For a **100% stacked** version, divide each row by its sum first: `cross.div(cross.sum(axis=1), axis=0) * 100`.

**Why it matters.** Stacked bars communicate **composition** — "what's the mix of B within each A?" Useful for category-share questions ("what fraction of orders in each region is Prime?"). 100%-stacked is the version when totals differ wildly and you only care about proportions.

**How it works.** First, build a cross-tab (count per `(a, b)` pair) using groupby+size+unstack or `pd.crosstab`. Then call `.plot(kind='bar', stacked=True)`. Each row of the cross-tab becomes one stacked bar.

**Where it's used.** Marketing reports: "category share by region". Survey results: "answer distribution per question". A/B test composition: "variants × outcome counts".

**Related terms.**
- **`crosstab()`** — pandas' shortcut for building the underlying count table.
- **100%-stacked** — proportion view, not absolute.
- **Grouped bar** — sibling alternative; bars side-by-side, not stacked.
- **`unstack()`** — pivots the groupby result into the wide shape `plot()` expects.

```python
# 100%-stacked bar — proportions per group
cross = df.groupby(['region','status']).size().unstack(fill_value=0)
(cross.div(cross.sum(axis=1), axis=0) * 100).plot(kind='bar', stacked=True)
```

**Gotcha.** Stacked bars with too many segments (more than ~5) become unreadable. Limit categories or aggregate small ones into "Other".

#### `.corr()` — correlation matrix

> **🪜 Mental model:** *Pairwise linear-relationship strength.* For every pair of numeric columns, a number in `[-1, +1]` summarising how linearly they move together.

**What it is.** `df.corr(method='pearson' | 'spearman' | 'kendall')` returns a square DataFrame of pairwise correlations. Pearson (default) measures **linear** correlation. Spearman measures **monotonic** correlation (works for ordinal/rank data, catches non-linear monotones). Kendall is a robust alternative based on concordant pairs. Diagonal is always 1 (a column with itself).

**Why it matters.** Correlation is the most-used summary of "how strongly are these two variables related?" It's the first quantitative probe before fitting a regression. In feature engineering, highly-correlated features signal redundancy. In interviews, "correlation ≠ causation" and "Pearson r=0 ≠ independent" are top traps.

**How it works.** Pearson r = `cov(x, y) / (σ_x · σ_y)` — covariance divided by the product of standard deviations. Range is `[-1, +1]`: `+1` perfect positive linear, `-1` perfect negative linear, `0` no linear relationship. Spearman is Pearson applied to the **ranks** of the data instead of the raw values — that's why it catches monotonic non-linear shapes.

**Where it's used.** Correlation heatmap during EDA. Feature-selection screen ("drop one of any pair with |r| > 0.9"). Time-series cross-correlation. Diagnostic for multicollinearity in linear regression.

**Related terms.**
- **Pearson** — linear; sensitive to outliers; only meaningful for numerics.
- **Spearman** — rank-based; catches monotonic non-linear; robust to outliers.
- **Kendall** — concordant-pair-based; even more robust, slower.
- **r = 0 ≠ independence** — the canonical interview trap. r=0 means no LINEAR relationship; non-linear relationships may still exist (e.g., quadratic).
- **Covariance** — the unscaled version of correlation; scale-dependent.

```python
df.select_dtypes('number').corr(method='pearson')
df[['x','y']].corr(method='spearman')
```

**Gotcha.** `.corr()` silently **drops non-numeric columns** — if a column you expected isn't in the result, check its dtype with `df.dtypes`.

#### `hue` — adds a categorical 3rd dimension via colour

> **🪜 Mental model:** *Colour-by-category overlay.* When you have a 2D plot (scatter, line, box) and want a 3rd variable, encode it as colour by setting `hue=`.

**What it is.** `hue=` is a seaborn parameter that splits a plot by a categorical (or low-cardinality) variable and colours each subset differently. In a scatter plot it colours individual points; in a line plot it draws one line per category; in a box plot it puts side-by-side boxes per category. A legend appears automatically.

**Why it matters.** Without `hue`, a 2D plot can only show two variables. With `hue`, you can answer "does the relationship differ across groups?" — e.g., is the price-vs-rating cloud the same for Electronics and Apparel? It's the simplest way to add a dimension without going to 3D (which is almost always a bad idea).

**How it works.** Seaborn groups the data by the `hue` column, then plots each group with a different color from the active palette. The legend maps colour → category. You can control the palette with `palette='coolwarm'`.

**Where it's used.** Scatter coloured by class. Multi-line trend per cohort. Side-by-side boxes per group. Any "compare across category" question in EDA.

**Related terms.**
- **`size=`** — encodes a numeric 3rd dimension via dot/line size.
- **`style=`** — encodes a categorical 3rd dimension via marker shape (use instead of hue for grayscale prints).
- **`palette=`** — colour scheme; `'coolwarm'`, `'Spectral'`, `'viridis'`.
- **Faceting / subplots** — alternative for too-many categories; one mini-plot per category.

```python
sns.scatterplot(data=df, x='price', y='rating', hue='category', alpha=0.5)
```

**Gotcha.** `hue=` with 50+ categories produces an unreadable legend. Bucket the variable first (`pd.cut`) or pick top-N and lump the rest as "Other".

#### Subplots / `plt.subplots` / faceting

> **🪜 Mental model:** *A grid of mini-plots in one figure.* When one plot can't fit all the views you need, lay out multiple `Axes` side-by-side and put one plot in each.

**What it is.** `fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))` creates a figure with a grid of empty axes. You then plot into each one by indexing (`axes[0]`, `axes[1, 2]`) and passing `ax=axes[...]` to a seaborn call. After the last plot, `plt.tight_layout()` rebalances padding so labels don't collide. Faceting (`sns.FacetGrid` or `sns.catplot`/`sns.relplot` with `col=` / `row=`) is the higher-level version where seaborn builds the grid automatically based on a categorical column.

**Why it matters.** Side-by-side plots make patterns instantly comparable. "Box vs violin" in two adjacent panels. "Same scatter, four price bands". Sometimes the cleanest answer to "how do I show this?" is "two charts together."

**How it works.** `plt.subplots(nrows, ncols)` returns a `Figure` and an array (or 2D grid) of `Axes` objects. Each Axes is an independent plotting area — its own x/y scale, title, legend. You explicitly pass which axes a plot goes into via `ax=`. `tight_layout()` runs a quick layout solver after the plots are drawn.

**Where it's used.** Comparing the same data in two views (box and violin). Showing the same plot per category (faceting). Dashboards. Pre/post-cleanup comparisons.

**Related terms.**
- **`plt.tight_layout()`** — call after the last plot, before `show`, to avoid label overlap.
- **`figsize=(w, h)`** — figure size in inches; bigger for more subplots.
- **`sns.FacetGrid` / `col=` / `row=`** — seaborn's auto-faceting layer.
- **`fig.suptitle('...')`** — overall title across all subplots.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df,    x='category', y='rating', ax=axes[0])
sns.violinplot(data=df, x='category', y='rating', ax=axes[1])
plt.tight_layout()
plt.show()
```

**Gotcha.** Forget `plt.tight_layout()` and titles/labels overlap the figure edge. Forget `ax=axes[i]` and seaborn draws on the wrong panel.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#3g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Scatter plot** | One dot per row at `(x, y)`. Shows the relationship between two numeric variables. | "Is X related to Y?" Always look at scatter before trusting a correlation number. |
| **Line plot** | Scatter + line segments connecting points sorted by `x`. Implies x has an order. | Trends over time, learning curves, anything where x is sequential. |
| **Box plot** | One box per group showing the 5-number summary (min/Q1/median/Q3/max) + outlier dots beyond 1.5×IQR. | Comparing distributions across categories. Quick outlier inspection. |
| **Violin plot** | Box plot + mirrored KDE — shows summary **and** distribution shape (bimodal? skewed?). | When the shape of the per-group distribution matters, not just the summary. |
| **Grouped bar** | One bar per category showing an aggregate (default mean) with a CI band. With `hue=`, side-by-side bars per sub-category. | Executive-style "average per region" reports. Aggregated comparison. |
| **Pair plot** | N×N grid: scatter off-diagonal, distribution on-diagonal, for every pair of numeric columns. | First-look EDA on small (~5–8 col) numeric datasets. Class separability check with `hue=label`. |
| **Heatmap** | 2D matrix drawn as color-encoded cells. `annot=True` overlays numbers. | Correlation matrices, confusion matrices, pivot tables — pattern detection in 2D structure. |
| **Stacked bar / 100% stacked** | Bars split into segments showing composition by a 2nd variable. 100%-stacked normalises to proportions. | "Mix of B inside A." Marketing share, survey composition, A/B variant outcomes. |
| **`.corr()`** | Pairwise correlation matrix. Pearson (linear), Spearman (monotonic), Kendall (rank-pair). | Quantitative summary of relationships. Feature-redundancy check. |
| **`hue` parameter** | Adds a categorical 3rd dimension by colouring points/lines/boxes per category. | Comparing the same relationship across groups (price-vs-rating per category). |
| **`subplots` / faceting** | Grid of independent mini-plots in one figure. `plt.subplots(rows, cols)` or `sns` faceting via `col=`. | Side-by-side comparisons (box vs violin, before vs after, per category). |
| **Pearson r = 0** | No **linear** relationship — non-linear (quadratic, sinusoidal) relationships may still be strong. | The canonical interview trap. Don't conclude independence from r = 0. |
| **`alpha`** | Transparency 0–1. Lets overlapping points show density instead of one black blob. | Any scatter with overplotting. Try `alpha=0.3–0.5`. |
| **`palette`** | Colour scheme — `'coolwarm'` (diverging), `'viridis'` (sequential), `'Set2'` (categorical). | Tune readability of `hue`-encoded plots. Match the data type (sequential vs diverging vs categorical). |

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

<a id="4g-guided"></a>
### 📖 Guided concept walkthrough

> Beginner-first introduction of every probability concept in this module. Translation: every formula here is followed by a plain-English sentence so the symbols stop feeling like a wall. Read top-to-bottom on a first pass; the cheat sheet below is the recap.

#### Sample space and events

> **🪜 Mental model:** *The whole list of "what could happen", and the subsets you care about.* The sample space is **everything possible**; an event is **the subset where the question is "yes"**.

**What it is.** The **sample space** (usually written **S** or **Ω**) is the set of all possible outcomes of a random experiment. Roll a die → S = {1, 2, 3, 4, 5, 6}. A Sachin ODI innings → S = the set of all match outcomes. An **event** is any subset of S — e.g., "rolled an even number" = {2, 4, 6}, or "Sachin scored a century" = the set of matches where runs ≥ 100. A single outcome (e.g., "rolled a 4") is also called an **elementary event** or just an **outcome**.

**Why it matters.** Probability is just "counting how many elementary outcomes are in this event, divided by how many are in the sample space." Until you can clearly name the sample space and the event, you can't compute anything. Interview traps almost always come from a sloppy sample space — e.g., "conditional on already knowing it's a boy" silently shrinks S to "families with at least one boy."

**How it works.** In pandas, the **sample space is your DataFrame** — each row is one elementary outcome. An event is a **subset of rows** — usually built with a boolean mask: `df[df['runs'] >= 100]`. Then `len(event) / len(sample_space)` is the empirical probability of that event. Every probability calculation in this module reduces to "filter the DataFrame, count rows, divide."

**Where it's used.** Every probability problem starts here — "what's the sample space?" should be the first sentence of any answer. In data analysis, S is the DataFrame; events are masks. In ML, S is the set of possible (X, y) pairs; events are subsets like "model predicted positive."

**Related terms.**
- **Outcome / elementary event** — a single point in S; a single row of the DataFrame.
- **Event** — any subset of S; a boolean filter of the DataFrame.
- **Universe** — synonym for sample space.
- **Trial / experiment** — one "execution" of the random process (one die roll, one match).
- **Mutually exclusive events** — events that share no outcomes (no overlap in row sets).

```python
S    = df                                       # sample space = all rows
event = df[df['runs'] >= 100]                    # event = subset of rows
P_event = len(event) / len(S)
```

**Gotcha.** If your DataFrame has duplicates, the "sample space" is corrupted — each duplicate inflates the count. Dedupe (and define what "one trial" means) before computing probabilities.

#### Set operations on events (∪, ∩, complement)

> **🪜 Mental model:** *Events are sets; combine them with set algebra.* Union = "either or both happen"; intersection = "both happen"; complement = "doesn't happen".

**What it is.** Three operations you can do on events:
- **Union** (`A ∪ B`) — at least one of A, B occurs. In pandas: `df[mask_A | mask_B]`.
- **Intersection** (`A ∩ B`) — both A and B occur. In pandas: `df[mask_A & mask_B]`.
- **Complement** (`A′` or `A^c` or `¬A`) — A does **not** occur. In pandas: `df[~mask_A]`.

In plain English: ∪ is "OR", ∩ is "AND", complement is "NOT".

**Why it matters.** Every multi-condition probability question is built from these. "P(century OR fifty)" is a union. "P(century AND won the match)" is an intersection. "P(did not score a century)" is a complement. Getting the boolean operators (`&`, `|`, `~`) right — including the parentheses — is half the battle in pandas probability code.

**How it works.** Boolean masks combine elementwise:
- `mask_A | mask_B` — element-wise OR; True where either mask is True.
- `mask_A & mask_B` — element-wise AND; True where both masks are True.
- `~mask_A` — element-wise NOT; flips True ↔ False.

Each compound mask is itself a boolean Series, ready to feed into `df[mask]` to get the rows of the corresponding event.

**Where it's used.** Every multi-condition filter in EDA. Every probability of a compound event. Every set-builder description ("matches where won AND scored century").

**Related terms.**
- **`&`, `|`, `~`** — pandas bitwise operators; ALWAYS parenthesize each comparison.
- **`and`, `or`, `not`** — Python keywords; **do not work** on Series (raise `ValueError`).
- **Mutually exclusive** — A ∩ B = ∅ (no overlap).
- **Complement rule** — P(A′) = 1 − P(A) (see below).

```python
df[(df['runs'] > 50) & (df['Won'])]    # intersection
df[(df['runs'] > 50) | (df['century'])] # union
df[~(df['runs'] > 50)]                  # complement
```

**Gotcha.** `df[mask_A and mask_B]` raises `ValueError: ambiguous truth value`. Pandas needs `&` (and parentheses around each comparison because `&` has higher precedence than `>` / `<`).

#### Probability — the basic definition

> **🪜 Mental model:** *Favourable outcomes ÷ total outcomes.* If every outcome in S is equally likely, the probability of event A is `|A| / |S|`.

**What it is.** For a finite sample space with equally-likely outcomes, **P(A) = |A| / |S|** — the number of outcomes in event A divided by the number in the sample space. P(A) is always in [0, 1]. When outcomes are NOT equally likely, probability is defined more carefully (axiomatically by Kolmogorov), but the empirical formula `count_satisfying / total_count` is still what we compute from data.

**Why it matters.** This is the operational definition we use in every Pandas probability calculation — `(df['runs'] > 50).mean()` is exactly this formula, since `mean` of a boolean Series equals `(count of True) / (total count)`. Internalising that *mean-of-bool = probability* is the single most useful shortcut in applied probability.

**How it works.**
1. Define event A as a boolean mask: `mask = df['runs'] > 50`.
2. `mask.sum()` counts the True values — the favourable outcomes.
3. `len(df)` (or `len(mask)`) is |S| — the total.
4. `mask.sum() / len(df)` = empirical P(A).
5. Equivalently and more pandas-idiomatic: `mask.mean()`. (Boolean → 1/0 → mean = fraction True.)

For non-uniform sample spaces (weighted outcomes), `(df['weight'] * mask).sum() / df['weight'].sum()`.

**Where it's used.** Every empirical probability calculation. Class-balance audits in ML (`(y == 1).mean()`). Conversion rates (`(df['clicked']).mean()`). Hit rates in models (`(y_pred == y_true).mean()`).

**Related terms.**
- **Empirical probability** — what we compute from data; converges to the true probability as n → ∞ (law of large numbers).
- **Theoretical probability** — derived from a model (P(heads) = 0.5 for a fair coin).
- **`mean()` on a bool Series** — fraction of True; equals empirical P.
- **Law of large numbers** — empirical converges to theoretical as sample size grows.

```python
P_50plus = (df['runs'] > 50).mean()    # mean of bool = empirical P
```

**Gotcha.** Probability is **always** in [0, 1]. If you compute >1 or <0, you've divided by the wrong denominator or the mask is computing something other than 0/1.

#### Addition rule of probability

> **🪜 Mental model:** *P(A) + P(B) double-counts the overlap; subtract it once.* "Anyone in A OR B" = "people in A" + "people in B" − "people in both".

**What it is.** For any two events A and B:
**P(A ∪ B) = P(A) + P(B) − P(A ∩ B)**

In plain English: the probability that A happens OR B happens (or both) equals the probability of A plus the probability of B, minus the probability of both happening together. The subtraction corrects for the fact that the intersection got counted twice — once in each individual probability.

If A and B are **mutually exclusive** (can't both happen, so P(A ∩ B) = 0), the formula simplifies to **P(A ∪ B) = P(A) + P(B)**.

**Why it matters.** This is the formula behind "what fraction of matches did Sachin score a fifty OR a century?" If you naively add P(fifty) + P(century), you double-count the centuries (which are also fifties). The addition rule corrects that. Beginners forget the subtraction constantly.

**How it works.**
1. Compute P(A) = `mask_A.mean()`.
2. Compute P(B) = `mask_B.mean()`.
3. Compute P(A ∩ B) = `(mask_A & mask_B).mean()`.
4. P(A ∪ B) = P(A) + P(B) − P(A ∩ B).

Alternatively, compute it directly: `(mask_A | mask_B).mean()`. Both should give the same answer — if they don't, your masks aren't what you think they are.

**Where it's used.** Any "P(A or B)" question. Union-of-events probabilities in survey analysis. Coverage metrics in IR / search: "fraction of queries matched by retriever A or retriever B".

**Related terms.**
- **Mutually exclusive events** — P(A ∩ B) = 0, so subtraction term drops out.
- **Inclusion-exclusion principle** — generalisation for ≥ 3 events: alternating sum.
- **`|` operator on masks** — the direct way to compute the union in pandas.

```python
P_A_or_B = P_A + P_B - P_A_and_B
# or directly:
P_A_or_B = (mask_A | mask_B).mean()
```

**Gotcha.** "P(A or B) = P(A) + P(B)" is **only correct when A and B are mutually exclusive**. Forgetting the −P(A ∩ B) is the #1 addition-rule mistake.

#### Multiplication rule of probability

> **🪜 Mental model:** *P(A AND B) = P(A) × P(B given A).* "Both happen" = "first one happens" × "second one happens given the first did".

**What it is.** For any two events A and B:
**P(A ∩ B) = P(A) · P(B | A)**

In plain English: the probability that BOTH A and B happen equals the probability that A happens, times the probability that B happens given A has already happened.

If A and B are **independent** (knowing A doesn't change P(B)), then P(B | A) = P(B), and the formula simplifies to **P(A ∩ B) = P(A) · P(B)**.

**Why it matters.** Joint probabilities are everywhere — "what's the chance both events happen?" The general form (with the conditional) is the foundation of Bayes' theorem. The independence shortcut is the formula students memorise — but it's only valid under independence, which you must verify.

**How it works.**
1. P(A) = `mask_A.mean()` — probability A happens (marginal).
2. P(B | A) = `mask_B[mask_A].mean()` — among rows where A is True, what fraction also have B? (See conditional probability below.)
3. P(A ∩ B) = product of the two.

Alternatively (and more directly): `(mask_A & mask_B).mean()`. As with the addition rule, both routes should agree.

**Where it's used.** Joint event probabilities. Building Bayes' theorem from empirical pieces. Sequential trials: "probability of two heads in a row" = P(H) · P(H | H) = 0.5 · 0.5 if independent.

**Related terms.**
- **Independence** — when P(B | A) = P(B); enables the product shortcut.
- **Conditional probability** — the P(B | A) factor; see below.
- **Joint probability** — what the rule computes (P(A ∩ B)).
- **Chain rule** — general form for many events: P(A₁ ∩ … ∩ Aₙ) = P(A₁) · P(A₂ | A₁) · … · P(Aₙ | A₁ ∩ … ∩ Aₙ₋₁).

```python
P_A      = (df['A']).mean()
P_B_givA = (df.loc[df['A'], 'B']).mean()     # condition on A
P_AB     = P_A * P_B_givA
```

**Gotcha.** "P(A and B) = P(A) · P(B)" works ONLY for independent events. If "A happened" tells you anything about B (e.g., temperature high and AC bill high), they're not independent, and you must use the conditional form.

#### Complement rule

> **🪜 Mental model:** *Everything that isn't A.* P(not A) = 1 − P(A). Always.

**What it is.** The **complement** of an event A — denoted A′ or Aᶜ or ¬A — is "A does not happen". The complement rule states:
**P(A′) = 1 − P(A)**

Plain English: the probability of A not happening equals one minus the probability of A happening. Always true, no conditions.

**Why it matters.** The complement rule is the single most powerful trick in probability. Many questions are vastly easier to compute as "1 minus the complement" than directly. *"What's the probability of at least one six in 10 rolls?"* = `1 − P(no sixes in 10 rolls)` = `1 − (5/6)^10`. Computing it directly would require inclusion-exclusion over many cases.

**How it works.** In pandas: `P_not_A = 1 - mask_A.mean()`, or equivalently `(~mask_A).mean()`. Both give the same number.

The rule holds because the events A and A′ partition the sample space (every outcome is in exactly one), and probabilities of a partition sum to 1.

**Where it's used.** "At least one" calculations (compute "none", subtract from 1). Sanity checks (P + P(complement) must equal 1). Quick interview wins on "what's the probability of at least one match?" type questions.

**Related terms.**
- **Partition** — splitting S into disjoint events whose probabilities sum to 1.
- **De Morgan's laws** — complement distributes over union/intersection: (A ∪ B)′ = A′ ∩ B′.
- **"At least one" trick** — P(at least one) = 1 − P(none).

```python
P_A     = (df['Won']).mean()
P_not_A = 1 - P_A                       # or (~df['Won']).mean()
```

**Gotcha.** Don't compute "at least one" by adding individual probabilities — you'll double-count overlaps. Use 1 − P(none) instead.

#### Marginal probability

> **🪜 Mental model:** *Probability of A alone, ignoring all other variables.* The "single-column" probability.

**What it is.** The **marginal probability** P(A) is the probability of event A occurring, *without conditioning on any other event*. It's what you get when you "marginalise over" (sum out) every other variable.

For a 2-variable joint distribution P(A, B), the marginal of A is **P(A) = Σ_b P(A, B=b)** — sum the joint over all values of the other variable. In English: to get P(A), add up the joint probability across every possible B.

**Why it matters.** Marginal probabilities are the unconditional "baseline" — "what's the overall fraction of wins?" P(win). They serve as priors in Bayes (the "P(A)" factor) and are what naive analyses default to. Distinguishing marginal from conditional is one of the most-tested concepts in interviews.

**How it works.** In a DataFrame, marginal P(A) = `(df['A']).mean()` — no filtering. The fact that you're not filtering by anything else IS the marginalisation. If you had a frequency table of joint events, you'd sum across rows or columns to get the marginal — same idea.

**Where it's used.** Class balance: marginal P(y=1). Click-through rate: marginal P(clicked). Prior probability in Bayes. Baseline metric to compare conditional probabilities against ("conditional given X is 80%, but the marginal is 50% — so X really moves the needle").

**Related terms.**
- **Joint probability** — P(A ∩ B), both happen (see below).
- **Conditional probability** — P(A | B), A given B (see below).
- **Prior** — the marginal P(A) before observing evidence, in Bayesian language.
- **Marginalisation** — the operation of summing the joint over one variable to get the other's marginal.

```python
P_win = df['Won'].mean()           # marginal — no condition
```

**Gotcha.** Don't confuse marginal P(A) with conditional P(A | B). Marginal is "overall"; conditional is "within the B group." If your "overall conversion rate" doesn't match your "per-cohort" rates, Simpson's paradox is lurking.

#### Joint probability

> **🪜 Mental model:** *Both happen.* P(A ∩ B) = the probability that A and B occur together.

**What it is.** The **joint probability** of two events is the probability they **both occur**: **P(A ∩ B)** (also written P(A, B) or P(A and B)). For independent events, it factors: P(A ∩ B) = P(A) · P(B). In general, it requires the multiplication rule: P(A ∩ B) = P(A) · P(B | A).

**Why it matters.** Joint probabilities are the "raw material" of probability: the joint distribution P(A, B) over all (A, B) pairs determines every marginal and every conditional. In a 2×2 contingency table, the four cell probabilities are joints; row/column totals are marginals; ratios are conditionals.

**How it works.** In pandas: `P_AB = ((df['A']) & (df['B'])).mean()` — the fraction of rows where both columns are True. With a contingency table (`pd.crosstab(df['A'], df['B'], normalize=True)`), each cell IS a joint probability.

For continuous variables, the joint is a 2D density rather than a single number — but the principle (probability of both happening together) is the same.

**Where it's used.** Contingency tables (joint = cell value). Confusion matrices in ML evaluation (joint of "true class" and "predicted class"). Bayesian model components.

**Related terms.**
- **Marginal probability** — sum the joint over the other variable to get it.
- **Conditional probability** — divide the joint by a marginal to get it (P(A | B) = P(A ∩ B) / P(B)).
- **Independence** — when P(A ∩ B) = P(A) · P(B).
- **Joint distribution** — the full table of all P(A, B) pairs.

```python
P_AB = ((df['runs'] > 50) & (df['Won'])).mean()
pd.crosstab(df['runs_50plus'], df['Won'], normalize=True)   # joint table
```

**Gotcha.** P(A ∩ B) is NOT P(A) + P(B). That's the addition rule territory (union, not intersection).

#### Conditional probability — P(A | B)

> **🪜 Mental model:** *Shrink the world.* Given B happened, restrict your attention to only the rows where B is True, then ask "what fraction of those have A?"

**What it is.** The conditional probability of A given B is:
**P(A | B) = P(A ∩ B) / P(B)**, defined for P(B) > 0.

In plain English: the probability that A occurs given that B has already occurred equals the probability that both A and B occur divided by the probability that B occurs. The denominator changes from the full sample space size to just |B| — the rows where B is True.

**Why it matters.** Conditional probabilities answer the most useful questions: "given the customer is a Prime member, what's the conversion rate?" "given the test is positive, what's the chance of disease?" Conditioning is also the foundation of Bayes' theorem and the whole Bayesian framework.

**How it works.** The denominator changes:
- Marginal P(A) divides by `len(df)`.
- Conditional P(A | B) divides by `len(df[mask_B])` — only the B-true rows.

In pandas:
```python
P_A_given_B = df.loc[mask_B, 'A'].mean()
# equivalent:
P_A_given_B = (mask_A & mask_B).mean() / mask_B.mean()
```

The first form is more idiomatic: filter to B, then take the mean of A. The second form makes the formula explicit.

**Where it's used.** Conversion rates ("conditional on visiting, fraction who buy"). Test diagnostics ("given positive test, P(disease)"). Recommender systems (P(click | shown)). Every Bayes problem.

**Related terms.**
- **Joint probability** — the numerator P(A ∩ B).
- **Marginal probability** — the denominator P(B).
- **Bayes' theorem** — flips the conditioning direction (see below).
- **Posterior probability** — Bayesian name for the conditional P(parameter | data).

```python
P_century_given_50plus = df.loc[df['runs'] > 50, 'century'].mean()
```

**Gotcha.** Conditional probability is **not symmetric**: P(A | B) ≠ P(B | A) in general. Confusing the two directions is the "prosecutor's fallacy" — P(positive | innocent) is NOT P(innocent | positive).

#### Independence vs mutual exclusivity

> **🪜 Mental model:** *Independence and mutual exclusivity are NEAR-OPPOSITES, not synonyms.* Independent = "knowing one tells you nothing about the other"; mutually exclusive = "if one happens, the other CANNOT".

**What it is.** Two distinct concepts beginners constantly mix up:
- **Independent**: knowing A happened doesn't change the probability of B. Formally: **P(A ∩ B) = P(A) · P(B)** (equivalently P(B | A) = P(B)). Example: two coin tosses.
- **Mutually exclusive** (disjoint): A and B cannot both happen. Formally: **P(A ∩ B) = 0**. Example: a single die roll can't simultaneously be 1 AND 6.

Crucially: **mutually-exclusive events with positive probability are NEVER independent.** If A and B are mutually exclusive and both have nonzero probability, then knowing A happened tells you B definitely did NOT — which is the strongest possible dependence.

**Why it matters.** This is one of the top-3 probability interview traps. People hear "they don't both happen" and intuitively think "they're independent" — but the formal meanings are nearly the opposite. The confusion is so common that `kojino/120-Data-Science-Interview-Questions` opens with this exact distinction.

**How it works.**
- Independence check: `(mask_A & mask_B).mean()` should equal `mask_A.mean() * mask_B.mean()`. If close, they're (approximately) independent.
- Mutual exclusivity check: `(mask_A & mask_B).mean()` should equal `0`. If both have positive marginal P and joint is 0, they're mutually exclusive AND therefore NOT independent.

**Where it's used.** Every probability interview. Designing A/B tests (treatments are mutually exclusive; not independent of the user being in the test). Setting up a Bayesian model (priors are usually independent assumptions).

**Related terms.**
- **Disjoint** — synonym for mutually exclusive.
- **Conditional independence** — A and B independent *given* C; common in ML (naive Bayes).
- **Pairwise vs mutual independence** — pairwise independence doesn't imply joint (3-way) independence.
- **Mutually exclusive events** — partition the sample space when also exhaustive.

```python
# Independence test
joint = (mask_A & mask_B).mean()
indep = mask_A.mean() * mask_B.mean()
print(joint, indep)                # ≈ equal → independent
```

**Gotcha.** "These two never happen together" does NOT mean "they're independent" — it means the opposite. Independence requires their joint to equal the product of marginals, not zero.

#### Bayes' theorem

> **🪜 Mental model:** *Update belief with evidence.* Start with a prior P(A). See evidence B. Multiply by the likelihood P(B | A), divide by the evidence's marginal P(B). Out comes the posterior P(A | B).

**What it is.** Bayes' theorem is the formula for flipping the conditioning direction:
**P(A | B) = P(B | A) · P(A) / P(B)**

Translating every symbol:
- **P(A | B)** — the **posterior**: probability of A given that we observed B.
- **P(B | A)** — the **likelihood**: probability of seeing B if A were true.
- **P(A)** — the **prior**: our initial belief in A before seeing any evidence.
- **P(B)** — the **evidence** (also called the marginal of B): how likely B is overall, summed over all possible causes.

In English: "the new probability of A given evidence B equals the old probability of A multiplied by how likely the evidence is under A, then normalised by how likely the evidence is overall."

**Why it matters.** Bayes is the single most important formula in probability and a cornerstone of ML — naive Bayes classifiers, Bayesian inference, A/B-test posteriors, medical diagnostic reasoning, spam filtering. The famous "factory defects" interview question (60% skilled / 40% unskilled, etc.) is solved by one Bayes application.

**How it works.**
1. Start with the prior: `P_A = mask_A.mean()`.
2. Compute the likelihood: `P_B_given_A = df.loc[mask_A, 'B'].mean()`.
3. Compute the evidence (marginal of B): `P_B = mask_B.mean()`. Or use the law of total probability: `P_B = P_B_given_A * P_A + P_B_given_notA * P_notA`.
4. Posterior: `P_A_given_B = P_B_given_A * P_A / P_B`.

The result is interpretable: how much does evidence B shift our belief in A? If `P(A | B) > P(A)`, B is supportive evidence; if `<`, it's contradictory.

**Where it's used.** Medical diagnostics (P(disease | positive test)). Spam filters (P(spam | "free money")). Naive Bayes classification. A/B test posteriors. Search relevance scoring. Litigation reasoning (avoiding the prosecutor's fallacy).

**Related terms.**
- **Prior** — P(A) before evidence.
- **Posterior** — P(A | B) after evidence.
- **Likelihood** — P(B | A) — how compatible the evidence is with A.
- **Evidence / marginal of B** — P(B), the normaliser.
- **Law of total probability** — P(B) = Σ_a P(B | A=a) · P(A=a); how to compute P(B) when only conditionals are known.
- **Naive Bayes** — ML classifier built on Bayes + a "features are conditionally independent" assumption.

```python
P_A      = df['A'].mean()
P_B_givA = df.loc[df['A'], 'B'].mean()
P_B      = df['B'].mean()
P_A_givB = P_B_givA * P_A / P_B
```

**Gotcha.** People confuse P(A | B) with P(B | A) — they are NOT the same. The "prosecutor's fallacy" (treating P(evidence | innocence) as P(innocence | evidence)) has sent people to jail.

#### Empirical probability via Pandas boolean filtering

> **🪜 Mental model:** *Probability ≈ "filter the rows, count them, divide".* In pandas, that's `mask.mean()` for a single condition or `(mask_A & mask_B).mean()` for joints.

**What it is.** The whole computational pattern for probability in pandas is just boolean-mask arithmetic. Three idioms cover almost everything:
- **Marginal**: `P(A) = mask_A.mean()`.
- **Joint**: `P(A ∩ B) = (mask_A & mask_B).mean()`.
- **Conditional**: `P(A | B) = df.loc[mask_B, 'A'].mean()` (filter to B, then mean of A).

Because `True == 1` and `False == 0`, `mean()` of a boolean Series IS the fraction of True values — i.e., the empirical probability.

**Why it matters.** Once you internalise these three idioms, every probability calculation in EDA is one line. You don't need scipy or a probability package — pandas + boolean masks is the toolkit. This is also exactly what interviews test: "given this DataFrame, compute P(won | scored century)" reduces to one well-written `.loc[mask, 'col'].mean()`.

**How it works.** A boolean Series is internally a uint8 array of 0s and 1s. `.sum()` is the count of True; `.mean()` is `sum / len`. For conditional probability, you `.loc[mask_B, 'A']` to restrict the world to the rows where B is True, then `.mean()` of A in that subset gives P(A | B). The denominator `len(df[mask_B])` is invisible but mathematically baked in.

**Where it's used.** Every probability question on real data. Class balance (`(y == 1).mean()`). Conversion rates. Diagnostic accuracy. Sanity-checking model predictions: `(y_pred == y_true).mean()` is accuracy, which is also `P(prediction correct)`.

**Related terms.**
- **`.mean()` on bool Series** — the universal "fraction True" calculation.
- **`.value_counts(normalize=True)`** — alternative for multi-class probabilities (returns the empirical distribution).
- **`pd.crosstab(a, b, normalize='all')`** — joint distribution table.
- **Law of large numbers** — why empirical probability approaches the true probability as n grows.

```python
P_century         = (df['runs'] >= 100).mean()
P_century_and_won = ((df['runs'] >= 100) & df['Won']).mean()
P_century_given_won = df.loc[df['Won'], 'runs'].ge(100).mean()
```

**Gotcha.** `.loc[mask, 'col']` returns a Series; calling `.mean()` is fine. But `.loc[mask]['col'].mean()` is chained indexing and may trigger `SettingWithCopyWarning` on assignment. Stick to single-step `.loc[mask, 'col']`.

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

### 🧠 Concept cheat sheet (recap)

> Recap table — every row 2–3 lines: *what it is + when you reach for it*. Full definitions are in [the guided walkthrough above](#4g-guided).

| Concept | What it is | When you use it |
|---|---|---|
| **Sample space (S)** | The set of every possible outcome of the random experiment. In pandas it's the DataFrame; each row is one outcome. | The first sentence of every probability answer: "what is S?" Defines what 100% means. |
| **Event** | Any subset of S — a boolean mask over the DataFrame. "Sachin scored > 50" = `df['runs'] > 50`. | Whenever you express a condition; every probability question is "what's the P of this event?" |
| **Set operations (∪, ∩, ¬)** | Union (`|`), intersection (`&`), complement (`~`) on boolean masks. Always parenthesize comparisons. | Any compound-condition probability. Translates set algebra into pandas code. |
| **Probability** | For equal-likely outcomes, P(A) = |A| / |S|. In pandas: `mask.mean()` — mean of a bool Series IS the fraction True. | Every empirical probability calc — class balance, conversion rate, hit rate. Always in [0, 1]. |
| **Addition rule** | P(A ∪ B) = P(A) + P(B) − P(A ∩ B). Subtraction corrects the double-count of the overlap. | "P(A or B)" questions. Subtraction drops if A and B are mutually exclusive. |
| **Multiplication rule** | P(A ∩ B) = P(A) · P(B \| A). Reduces to P(A) · P(B) under independence. | "P(both A and B)" questions. Foundation of Bayes. |
| **Complement rule** | P(A′) = 1 − P(A). Always true. | "At least one" trick: P(at least one) = 1 − P(none). Sanity check (P + complement = 1). |
| **Marginal P(A)** | Probability of A ignoring all other variables. `df['A'].mean()` with no filtering. | Baseline / overall rate — class balance, prior in Bayes. |
| **Joint P(A ∩ B)** | Probability both A and B occur. `(mask_A & mask_B).mean()`. | Contingency tables, confusion matrices, dependent-event analysis. |
| **Conditional P(A \| B)** | A given B occurred: P(A ∩ B) / P(B). The denominator shrinks from \|S\| to \|B\|. In pandas: `df.loc[mask_B, 'A'].mean()`. | "Given X happened, what's P(Y)?" questions — conversion, diagnostic accuracy, recommender hit-rate. |
| **Mutually exclusive** | A and B cannot both happen — P(A ∩ B) = 0. NOT the same as independent. | Disjoint outcomes (one roll can't be 1 AND 6). Simplifies addition rule. |
| **Independence** | Knowing A tells you nothing about B — P(A ∩ B) = P(A) · P(B). The OPPOSITE of mutually exclusive (when both have positive P). | Coin tosses, separate trials. Simplifies multiplication rule. Top interview confusion. |
| **Bayes' Theorem** | P(A \| B) = P(B \| A) · P(A) / P(B). Flips conditioning direction; turns prior + likelihood into posterior. | Diagnostics, naive Bayes, A/B test posteriors. Any "given evidence, update belief" question. |
| **Empirical probability** | What you compute from data (frequency in observed sample). Converges to theoretical P as n grows (LLN). | Every pandas probability calc. Always report sample size. |

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
| **`agg`** | A pandas method that applies one or more aggregation functions to columns or groups in a single call. The modern "named aggregation" form (`agg(name=('col', 'fn'))`) lets you control both source column and output name. Use it to compute multiple summary stats per group at once. ([walkthrough](#1g-guided)) |
| **Aggregation** | Reducing many rows down to one summary value — `sum`, `mean`, `count`, `min`, `max`, `std`, `nunique`. The "collapse" step in split-apply-combine. Every report query is built from these. ([walkthrough](#1g-guided)) |
| **`apply`** | The universal escape hatch in pandas. Runs a Python function on every element (`Series.apply`), every row/column (`DataFrame.apply` with `axis=`), or every group (`GroupBy.apply`). Powerful but slow — it usually breaks the vectorised C path. ([walkthrough](#1g-guided)) |
| **Bayes' Theorem** | The formula that flips conditioning direction: P(A\|B) = P(B\|A) · P(A) / P(B). Translated: posterior equals likelihood times prior divided by evidence. The foundation of Bayesian inference, naive Bayes classifiers, and diagnostic reasoning. ([walkthrough](#4g-guided)) |
| **Binning** | Grouping continuous values into a small number of discrete buckets. Two pandas tools: `pd.cut` (by value edges) and `pd.qcut` (by quantile, equal-count). Produces a `Categorical` ready for groupby and plotting. ([walkthrough](#2g-guided)) |
| **Box plot** | A per-group plot showing the 5-number summary (min, Q1, median, Q3, max) plus outliers beyond 1.5×IQR. Compresses a distribution into a tiny, comparable shape — perfect for side-by-side group comparisons. ([walkthrough](#3g-guided)) |
| **Complement rule** | P(A′) = 1 − P(A). The single most powerful trick in probability: when "at least one" is hard to compute directly, compute "none" and subtract from 1. ([walkthrough](#4g-guided)) |
| **`pd.concat`** | Stacks DataFrames along an axis — `axis=0` glues more rows, `axis=1` glues more columns — without aligning on any key. Use for combining files of the same schema or for side-by-side comparison tables. ([walkthrough](#1g-guided)) |
| **Conditional probability** | P(A \| B) = P(A ∩ B) / P(B) — the probability of A given that B has occurred. The denominator shrinks from \|S\| (whole sample space) to \|B\| (only rows where B is true). In pandas: `df.loc[mask_B, 'A'].mean()`. ([walkthrough](#4g-guided)) |
| **Correlation (Pearson r)** | Linear association between two numerics, range [-1, +1]. r = 0 means **no linear relationship**, but non-linear relationships may still exist — the canonical interview trap. For monotonic non-linear, use Spearman. ([walkthrough](#3g-guided)) |
| **Count plot** | A bar plot where each bar's height is the count of rows in that categorical value. Equivalent to `value_counts().plot.bar()`. Use it on **categorical** columns; for continuous columns, use `histplot` instead. ([walkthrough](#2g-guided)) |
| **`pd.cut`** | Binning by **value range** — you pass explicit edges (`[0, 100, 500, ∞]`) and pandas places each value into the matching bucket. Bin widths are user-defined; counts per bin can be very unequal. ([walkthrough](#2g-guided)) |
| **Datetime accessor `.dt`** | A namespace on datetime Series exposing `.dt.year`, `.dt.month`, `.dt.day_name()`, `.dt.dayofweek`, `.dt.isocalendar().week`, `.dt.strftime(...)`. Mirrors the `.str` accessor — vectorised, C-speed. ([walkthrough](#2g-guided)) |
| **`dropna`** | Removes rows (or columns with `axis=1`) that have missing values. Parameters: `how='any'`/`'all'`, `subset=` (only check these columns), `thresh=k` (keep rows with ≥ k non-NaN). Default is often too aggressive — always pass `subset=`. ([walkthrough](#2g-guided)) |
| **`duplicated` / `drop_duplicates`** | `duplicated()` returns a boolean mask of repeats; `drop_duplicates()` removes them. `keep='first'` (default), `'last'`, or `False` (drop **all** copies). The first sanity check on any new dataset. ([walkthrough](#1g-guided)) |
| **Empirical probability** | Probability computed by counting rows in your data (favourable ÷ total). Converges to the theoretical value as sample size grows (Law of Large Numbers). Always report your sample size when quoting an empirical P. ([walkthrough](#4g-guided)) |
| **Event** | A subset of the sample space — any condition you can express as a boolean mask on the DataFrame. Examples: "Sachin scores > 50", "order was placed on a weekend". ([walkthrough](#4g-guided)) |
| **Faceting** | Auto-generating a grid of subplots, one per category, via `sns.FacetGrid` or `col=`/`row=` parameters on `relplot`/`catplot`. The clean way to show the same plot for each level of a category. ([walkthrough](#3g-guided)) |
| **`fillna`** | Replaces NaN with a scalar, a per-column dict, or a directional fill (`ffill`/`bfill`). The leak-free production version is `sklearn.impute.SimpleImputer` (fit on train, apply to test). ([walkthrough](#2g-guided)) |
| **GroupBy** | A lazy split-apply-combine engine: split rows into bins by key(s), apply a function in each bin, combine the results. Maps 1:1 to SQL `GROUP BY`. The most-asked pandas concept in interviews. ([walkthrough](#1g-guided)) |
| **Group filter** | `groupby('k').filter(fn)` — keeps the **rows** belonging to groups whose function returns True. Returns a DataFrame, not a GroupBy object. Use for "keep only users with > 5 orders" patterns. ([walkthrough](#1g-guided)) |
| **Heatmap** | A color-encoded 2D matrix, drawn with `sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)`. Great for correlation matrices and confusion matrices. ([walkthrough](#3g-guided)) |
| **Histogram** | A plot that bins a continuous variable into ranges and shows one bar per range. `sns.histplot(data=df, x='col', bins=20, kde=True)`. Use on **numeric** columns; for categories use count plot. ([walkthrough](#2g-guided)) |
| **Hue** | A seaborn parameter (`hue='col'`) that colours points/lines/boxes by a categorical variable, adding a 3rd dimension to a 2D plot. Bucket the column first if there are too many categories. ([walkthrough](#3g-guided)) |
| **Independence** | Two events A and B are independent if P(A ∩ B) = P(A) · P(B) (equivalently P(B \| A) = P(B)) — knowing A tells you nothing about B. The OPPOSITE of mutually exclusive when both events have positive probability. ([walkthrough](#4g-guided)) |
| **Inner join** | `pd.merge(..., how='inner')` — keep only rows where the key matches on **both** sides. The default and smallest possible join result. ([walkthrough](#1g-guided)) |
| **`isna` / `isnull`** | Identical methods — return a boolean mask True where the cell is missing (NaN/None/NaT). The standard idiom is `df.isna().sum()` for per-column NaN count, the first line of EDA. ([walkthrough](#1g-guided)) |
| **Joint probability** | P(A ∩ B) — the probability that A and B occur together. For independent events factors as P(A) · P(B); in general requires the multiplication rule P(A) · P(B \| A). In pandas: `(mask_A & mask_B).mean()`. ([walkthrough](#4g-guided)) |
| **KDE (Kernel Density Estimate)** | A smoothed estimate of a continuous distribution — like a histogram but with a continuous curve instead of bars. `sns.kdeplot` for the curve alone; `histplot(kde=True)` overlays it on a histogram. ([walkthrough](#2g-guided)) |
| **Left join** | `pd.merge(..., how='left')` — keep **every** row from the left; bolt on matching info from the right (NaN where no match). The most common join type in practice. ([walkthrough](#1g-guided)) |
| **Line plot** | A plot connecting `(x, y)` points sorted by x with line segments. Only meaningful when x has a natural order (time, sequence, rank). For unordered x, use bar or scatter. ([walkthrough](#3g-guided)) |
| **Marginal probability** | P(A) — the unconditional probability of A, without conditioning on anything. The "overall rate" or "baseline." Computed in pandas as `df['A'].mean()` with no filter. ([walkthrough](#4g-guided)) |
| **`pd.melt`** | Wide → long reshape. Stacks the columns listed in `value_vars` into one tall column, leaving `id_vars` columns repeated as identifiers. Prepares data for plotting and grouping. ([walkthrough](#2g-guided)) |
| **`pd.merge`** | SQL-style JOIN on a shared key. `how=` controls the four modes (inner/outer/left/right). The first step of almost every multi-table analysis; the most-asked pandas interview topic. ([walkthrough](#1g-guided)) |
| **Multiplication rule** | P(A ∩ B) = P(A) · P(B \| A). Simplifies to P(A) · P(B) only when A and B are independent. Foundation of Bayes' theorem and the chain rule. ([walkthrough](#4g-guided)) |
| **Mutually exclusive** | A and B cannot both happen — P(A ∩ B) = 0. The OPPOSITE of independence when both events have positive probability. Classic interview trap that beginners confuse with independence. ([walkthrough](#4g-guided)) |
| **Missing values (NaN)** | Pandas placeholder for "no value", appearing as `np.nan` (float), `None`, or `NaT` (datetime). NaN is contagious in math; `NaN != NaN`. Use `.isna()` to detect, `fillna`/`dropna` to handle. ([walkthrough](#2g-guided)) |
| **Outer join** | `pd.merge(..., how='outer')` — keep **every** key from either side; NaN-fill columns where the other side has no match. Used for data-quality audits with `indicator=True`. ([walkthrough](#1g-guided)) |
| **Pair plot** | `sns.pairplot(df)` — an N×N grid: scatter for every pair of numeric columns, distribution on the diagonal. First-look EDA on small (~5–8 col) datasets; doesn't scale to 30+ columns. ([walkthrough](#3g-guided)) |
| **Pivot / Pivot table** | Long → wide reshape. `pivot()` is strict (raises on duplicate (index, col) pairs); `pivot_table()` aggregates them silently with `aggfunc`. Use `pivot_table` on real data and always pass `aggfunc=` to document intent. ([walkthrough](#2g-guided)) |
| **Probability** | For equal-likely outcomes, P(A) = \|A\| / \|S\|. In pandas, `mask.mean()` is the empirical probability because mean of a bool Series equals fraction True. Always in [0, 1]. ([walkthrough](#4g-guided)) |
| **`pd.qcut`** | Binning by **quantile** — pandas picks the cut-points so each bucket has the same row count. Use for quartile/decile analysis or robust binning on skewed data. ([walkthrough](#2g-guided)) |
| **Right join** | `pd.merge(..., how='right')` — keep every row from the right side. Almost never used in practice; people prefer to swap arguments and use `how='left'`. ([walkthrough](#1g-guided)) |
| **Sample space (S)** | The set of every possible outcome of the random experiment. In data analysis, S is the DataFrame itself — each row is one outcome. Defining S precisely is the first step of any probability calculation. ([walkthrough](#4g-guided)) |
| **Scatter plot** | One dot per row at `(x, y)`. The canonical plot for the relationship between two numeric variables. Always look at the scatter before trusting a correlation number. ([walkthrough](#3g-guided)) |
| **Set operations on events** | Union (`|` = "A or B"), intersection (`&` = "A and B"), complement (`~` = "not A"). In pandas, these are bitwise operators on boolean masks — always parenthesize each comparison. ([walkthrough](#4g-guided)) |
| **`sort_values`** | Reorders rows by one or more columns. With a list `by=['a','b']`, sorts by `a` then breaks ties with `b`. `ascending=` accepts a list when `by` is a list. ([walkthrough](#1g-guided)) |
| **Split-apply-combine** | The three-step GroupBy pattern coined by Hadley Wickham: **split** rows into bins by key, **apply** a function in each bin, **combine** results back. Understanding it disambiguates `.agg`, `.transform`, `.filter`, `.apply`. ([walkthrough](#1g-guided)) |
| **Stacked plot** | Bar (or area) plot where each bar is split into colored segments showing composition by a 2nd variable. **100%-stacked** rescales each bar to 100% so you compare proportions, not totals. ([walkthrough](#3g-guided)) |
| **`.str` accessor** | Namespace of vectorised string operations on a Series — `contains`, `extract`, `split`, `replace`, `lower`. Regex by default. Pass `na=False` for safe boolean indexing. ([walkthrough](#2g-guided)) |
| **Subplots** | A grid of independent plotting areas in one figure, built with `fig, axes = plt.subplots(rows, cols)`. Plot into each via `ax=axes[i]`. Call `plt.tight_layout()` after to avoid label overlap. ([walkthrough](#3g-guided)) |
| **`pd.to_datetime`** | Parses strings (or numbers) into pandas `datetime64[ns]`. Pass `format=` for speed and to disambiguate locale-dependent formats; use `errors='coerce'` to turn parse failures into `NaT` instead of raising. ([walkthrough](#2g-guided)) |
| **Univariate / Bivariate / Multivariate** | One / two / three-or-more variables in a plot. Univariate plots show distributions (histogram, countplot); bivariate show relationships (scatter, line, box); multivariate use hue/size/subplots. ([walkthrough](#2g-guided), [M3 walkthrough](#3g-guided)) |
| **Violin plot** | A box plot fused with a kernel-density curve — shows both the 5-number summary and the distribution shape (bimodal? skewed?). Use when shape matters; otherwise box plot is simpler. ([walkthrough](#3g-guided)) |

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
