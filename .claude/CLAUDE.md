# CLAUDE.md — AIML Class Notes

Working instructions for preparing interview-ready revision notes from class notebooks.

## Project purpose

This repo holds AI/ML class notes organized **by module**. Each module is a top-level folder under the repo root — for example `Data Foundation/`, `2.ML Coding (Supervised Learning)/`, `5.ML Coding (CV)/`, and any future module (NLP, GenAI, MLOps, …). **Module names are free-form** — sometimes prefixed with a number, sometimes not. Every convention in this file applies to **all** modules equally; nothing here is specific to any one module.

For each module we maintain:

- **Notebooks** — one per lecture, kept in a subfolder named after the lecture (free-form, may carry a short prefix like `G -`).
- **Per-notebook deep-dive guides** — `<Topic>_Interview_Prep_Guide.md` inside the notebook's own folder. Created on demand (do not auto-generate), but **when one exists it follows the same beginner-friendly Concept Definition Template standard as the master guide** — see "Per-notebook deep-dive pattern" below.
- **Master revision guides** — `<Topic>_Revision_Guide.md` at the module root, consolidating multiple notebooks for interview prep.

The audience is the repo owner — **a beginner in AI/ML** studying for interviews. They need each concept explained well enough that they don't have to leave the guide to search the web for "what is X" or "why does X matter". Definitions must be **guided and intuitive**, not one-liners. The reader should be able to read a definition once and walk away with the *what*, *why*, *how*, *where*, and *related ideas* — without follow-up search.

**Voice:** intuitive, beginner-friendly, but still scannable. Plain English first; jargon introduced *only* after the plain-English version lands. Tables, code, and bullet structure stay — what changes is that every concept's first appearance must explain itself fully.

**Tone guard:** scannable ≠ telegraphic. A definition is allowed (and expected) to be 4–8 sentences when the concept is new to the reader. What we cut is filler, hype, and tutorial fluff — *not* the explanations themselves. Terseness applies to gotchas, drill items, and Q&A answers, not to first-introduction definitions.

## File layout

**Generic structure (applies to every module):**

```
AIML_ClassNotes/
├── CLAUDE.md                              ← these instructions (auto-loaded by Claude Code)
├── README.md                              ← index of all modules
├── .claude/                               ← Claude Code settings/hooks/skills only — NOT for CLAUDE.md
│
├── <Module A>/                            ← e.g. "Data Foundation"
│   ├── <Topic>_Revision_Guide.md          ← one or more master guides at module root
│   ├── <Topic>_Revision_Guide.md          ← (more master guides as topics grow)
│   ├── <Lecture Folder 1>/                ← free-form folder name, often the lecture title
│   │   ├── *.ipynb
│   │   └── <Topic>_Interview_Prep_Guide.md  (optional deep-dive)
│   ├── <Lecture Folder 2>/
│   │   └── *.ipynb
│   └── …
│
├── <Module B>/                            ← e.g. "2.ML Coding (Supervised Learning)"
│   └── … same shape as Module A …
│
├── <Module C>/                            ← e.g. "5.ML Coding (CV)"
│   └── … same shape …
│
└── <future modules>/                      ← NLP, GenAI, MLOps, etc. — add freely
```

**Concrete example (Data Foundation today):**

```
Data Foundation/
├── Data_Foundation_Revision_Guide.md             ← NumPy master guide
├── Pandas_Revision_Guide.md                      ← Pandas master guide
├── Amazon_Sachin_EDA_Revision_Guide.md           ← applied EDA + probability master guide
├── Food Delivery Data Exploration and analysis 1/
│   ├── *.ipynb
│   └── NumPy_EDA_Interview_Prep_Guide.md         ← per-notebook deep dive
├── Food Delivery Data Exploration and analysis 2/
│   └── *.ipynb
├── G -Amazon sales data analysis 1/
│   └── *.ipynb
└── …
```

This shape transfers verbatim to every other module. For `5.ML Coding (CV)/` you'd expect e.g. `CV_Revision_Guide.md` at its root plus per-notebook deep dives inside each lecture folder — same conventions, different topics.

**Naming rules:**
- **Module folders** — free-form. Numeric prefixes (`2.`, `5.`) are fine; spaces and parentheses are fine.
- **Lecture folders** — free-form, usually the lecture title. Short prefixes like `G -` are fine.
- **Master guides** — `<Topic>_Revision_Guide.md` at the module root. Multiple topic-scoped guides per module are encouraged (don't cram everything into one).
- **Deep-dive guides** — `<Topic>_Interview_Prep_Guide.md` inside the lecture folder. Optional.
- **Don't rename existing files unless the user asks** — sibling guides cross-link by path/anchor.

## Revision-guide pattern (REQUIRED)

All revision guides follow this structure. Match it when creating or updating a guide.

1. **Top anchor**: `<a id="top"></a>` then `# <Topic> — Master Revision Guide`.
2. **Intro blurb** explaining what it covers + bullet list of companion guides.
3. **"How to use"** — 2–3 bullets (pre-interview, just-before-coding-round, deep-dive pointer).
4. **🚀 Topic finder** — a table mapping "need to revise X" → anchor link.
5. **📑 Table of contents** — numbered sections, each with an anchor.
6. **Numbered modules** (one per notebook or topic cluster). Each module has, in this order:
   - One-line summary of what the notebook covers.
   - **🪜 Mental model** — 1–3 sentence analogy / visual that captures the *why* (see "Mental models" section below).
   - **📖 Guided concept walkthrough** — for every headline concept the module introduces, a full **Concept Definition Template** entry (see "Guided definitions" section). This is the section a beginner reads *first* — it answers what / why / how / where / related, so they never have to leave the guide to understand a term.
   - **🧠 Concept cheat sheet** — recap table summarising each concept introduced above into a **2–3 line** plain-English row (no naked one-liners — at minimum "what it is + when you use it"). This is the *recall* table after the walkthrough has done the teaching; it is never the first place a concept appears.
   - **🪞 Basic → Intermediate → Advanced** — for each headline concept, three layered examples (see "Basic → Advanced ladder" below).
   - **⚙️ Top APIs** — code block listing the calls.
   - **🧩 Code patterns** — numbered idiomatic snippets (5–8).
   - **🎯 Q&A — Module N** — 5–10 numbered questions with bold-question + plain answer. **At least 5 must be fetched & adapted from the source list** (cite inline, e.g., *"(adapted from `alexeygrigorev/data-science-interviews`)"*).
   - Trailing `[🔝 Back to top](#top)`.
7. **Cross-cutting sections** after the modules:
   - **📚 Terms glossary** — alphabetical table. **Each entry is a 2–4 sentence beginner-friendly definition**, not a one-liner; if a term is non-obvious, include a tiny "why it matters" tail. Treat the glossary as a "no-Google-needed" safety net.
   - **⚙️ API cheat sheet** — every method, grouped tables.
   - **⚠️ Gotchas & traps** — numbered list, each item leads with the trap in bold.
   - **🎯 Advanced interview Q&A** — bold-Q + plain-A, optionally sub-grouped. **≥ 10 sourced + 5 original** (auto-fetched per "Interview question sources" section).
   - **🌐 Sourced interview questions** — dedicated practice bank with **≥ 20 fetched-and-paraphrased questions** from the canonical sources, organised by source/sub-topic. Each batch carries a header naming its source. This is the section the user opens when they want to "do interview reps without browsing."
   - **🔁 Revision drill** — 50–100 flash-card Q&A items in blocks (A, B, C…). **70% sourced / 30% original.** End with "Score yourself" thresholds.
   - **✅ Best practices** — numbered, optionally sub-grouped (Performance, Correctness, Workflow, Interview-day).
   - **📦 Notebook mapping** — table of notebook ↔ what it covers ↔ which section in this guide.

Cross-link liberally between guides using relative links. Whenever a guide depends on another (e.g., the EDA guide assumes Pandas basics), add a pointer to the companion guide at the top.

## Per-notebook deep-dive pattern (REQUIRED when a deep dive exists)

A **per-notebook deep-dive guide** (`<Topic>_Interview_Prep_Guide.md` in a lecture folder) is the companion to a master guide's module. Where the master guide gives the *cross-notebook* synthesis, the deep dive gives the *what-this-particular-notebook-does* walkthrough. Both must follow the same beginner-friendly Concept Definition Template standard.

### When to create one

Do not auto-generate deep-dive guides. Create one only when the user asks for it, when a notebook is too rich to fit into a master-guide module, or when the lecture folder already contains a deep dive (in which case **maintain it to this standard whenever you touch the notebook**).

### Required structure

1. **Top anchor + title:** `<a id="top"></a>` then `# <Module/Notebook Name> — <Topic> (Deep Dive)`.
2. **Companion pointer:** one-line link to the master guide section that covers this notebook (`[../<Master>_Revision_Guide.md §N](../<Master>_Revision_Guide.md#N-anchor)`).
3. **"What this notebook actually demonstrates"** — 3–6 bullets summarising the dataset, the experiments, and the headline takeaway (e.g., "tiny CNN beats massive MLP with 67% fewer params").
4. **🪜 Mental anchors** — bulleted list of the 1-sentence mental models specific to this notebook. Reuse named anchors from CLAUDE.md's canonical list (tea-room, axis-that-disappears, right-align shapes, etc.) where applicable; coin a new one when needed.
5. **📖 Concept walkthroughs** — for **every concept this notebook touches** (whether new or assumed), a **full Concept Definition Template entry** (mental model → what / why / how / where / related → code → gotcha). This is the **most important section** — it's why the deep dive exists.

   **NO LINK-ONLY BLOCKS.** A deep-dive is not a thin wrapper over the master guide; it is a **standalone** beginner-friendly companion. Even if a concept is covered in the master, **re-give the full template here**, with the notebook's specific shapes, numbers, and code substituted in. The reader of a deep-dive should never have to click away to the master to understand a term.

   - The notebook-specific spin (the actual shapes, dataset, hyperparameters, failure mode demonstrated) goes into the **"Where it's used"** and **"How it works"** sections of each template entry.
   - You *may* end an entry with a one-line `→ For broader cross-notebook context, see [master §Ng](...)` pointer, but only **after** the full template is in place — never as a substitute for it.
   - If a concept is genuinely identical to its master entry word-for-word, copy the template entry over and add the notebook-specific paragraph to "How it works" or "Where it's used" — duplication is the right trade-off for beginner comprehension.
6. **🧠 Cell-by-cell walkthrough** — code-block-anchored explanation of what each meaningful cell does. Plain-English first; reference the Concept Definition Template entries above instead of re-defining concepts here.
7. **⚠️ Gotchas** — numbered list of notebook-specific traps (off-by-one shapes, dataset quirks, framework footguns).
8. **🎯 Notebook-specific Q&A** — **at least 10 interview-style questions** tied to the experiments this notebook runs (e.g., "if the CNN has 67% fewer params, why does it still outperform the MLP?"). **Each concept introduced in the 📖 walkthroughs section must have at least one Q&A item tied to it.** A minimum of **5 items must be sourced/paraphrased from canonical interview banks** (see "Interview question sources") with inline citations; the rest can be original "design & judgment" questions. The Q&A is what makes the deep-dive *interview-ready*; under-shipping here defeats the purpose of the file.
9. **🔝 Back-to-top anchor** at the end of each major section.

### Style guards (same as master guide)

- Beginner-friendly definitions are mandatory. No one-liner concept introductions.
- Define jargon inline on first use, even if defined earlier in the master guide — the deep dive is often read standalone.
- Where a concept already has a Concept Definition Template entry in the master guide, link to it (`[Conv2D](../<Master>_Revision_Guide.md#1g-guided)`) and add only the notebook-specific nuance.
- The cell-by-cell walkthrough is the deep dive's signature section, but **the 📖 Concept walkthroughs section comes first** — a beginner reading top-to-bottom needs the concept definitions before they read the code.

### What NOT to do in a deep-dive guide

- ❌ **Don't use link-only blocks for concepts.** Every concept that earns a header in the 📖 walkthroughs section must have a full Concept Definition Template entry (mental model + what + why + how + where + related + code + gotcha). The deep-dive is **standalone** — beginners should never have to click through to the master to understand a term. (Duplication of content with the master is the right trade-off for beginner comprehension. Add the notebook's specific shapes/numbers/code in the "How it works" and "Where it's used" sections.)
- ❌ Don't ship a deep-dive whose 📖 walkthroughs are missing **mental model** lines. Every concept entry leads with `> **🪜 Mental model:** ...`. Always.
- ❌ Don't ship fewer than 10 notebook-specific Q&A items, or fewer than 5 sourced/cited ones. Each walkthrough concept needs at least one Q&A item tied to it.
- ❌ Don't replace the cell-by-cell walkthrough with concept definitions — both serve different purposes.
- ❌ Don't skip the 📖 Concept walkthroughs section even if the notebook seems "small." Every notebook touches multiple concepts and beginners need each one explained.

## Style conventions

- **Emojis as section icons** (consistent set): 🚀 📑 🧠 ⚙️ 🧩 🎯 ⚠️ 📚 🔁 ✅ 📦 🔝.
- **Tables** for cheat sheets and quick references.
- **Code fences** for every API/snippet — always Python.
- **Bold** for key terms, gotcha leads, and Q&A questions.
- **Numbered lists** for gotchas, Q&A, code patterns, drill items.
- **Anchors** (`<a id="N-name"></a>`) before every major section; use them in TOC and Topic-finder links.
- **Beginner-friendly definitions are the default.** When a concept is *first* introduced, use the full Concept Definition Template (what / why / how / where / related — see "Guided definitions"). Terseness is reserved for *recall surfaces* (cheat sheets, drill, Q&A answers) — never for first explanations.
- **No tutorial fluff, but plain-English depth is required.** Cut hype, filler, motivational language, and "as we've seen above"; keep every sentence that builds understanding. If a reader would have to Google a term to understand the sentence after it, the term must be explained inline (parenthetical or follow-up sentence).
- **Jargon is introduced, not assumed.** First use of any ML term: state the plain-English meaning first, then the formal term in bold. Acronyms get expanded on first use.
- **Always include** a `[🔝 Back to top](#top)` at the end of each major section.

## Workflow for new notebooks

When the user drops new notebooks into **any** module folder (existing or brand-new — `Data Foundation`, `2.ML Coding (Supervised Learning)`, `5.ML Coding (CV)`, or a future one):

1. **Move/keep** each notebook in its own folder under the module (e.g., `Data Foundation/G - <Title>/`).
2. **Extract notebook text** for analysis — never read large `.ipynb` files directly; convert to text first:
   ```python
   import json, os
   with open(path) as f: nb = json.load(f)
   for cell in nb['cells']:
       src = ''.join(cell.get('source', []))
       # also pull cell.outputs[*].text or .data['text/plain'] for code cells
   ```
   This avoids burning context on base64-encoded images.
3. **For multi-notebook scans**, dispatch parallel `Explore` agents on the extracted `.txt` files and ask each to return a structured summary (agenda, APIs, concepts, patterns, dataset, business questions, interview-style cells).
4. **Decide guide placement**:
   - If the notebook fits a topic that already has a master guide → extend that guide's modules / mapping.
   - If it introduces a new topic cluster → create a new `<Topic>_Revision_Guide.md` at the module root and add it to the README + cross-link from sibling guides.
5. **Update the parent module's master guide mapping table** so it points to the new guide/section.
6. **Add an "Open in Colab" badge** as the first markdown cell of every new notebook (see "Open in Colab badge" below). Run `python3 scripts/add_colab_badge.py` after dropping new notebooks in — the script is idempotent and processes the whole repo by default.

## Open in Colab badge — required on every notebook

GitHub renders `.ipynb` files as static HTML — no collapsible outputs, no interactivity. The fix is a one-click jump into Google Colab, where the notebook becomes fully interactive (collapsible outputs, runnable cells, save-to-Drive). Every notebook in the repo carries an **Open in Colab** badge as its first markdown cell so the reader is one click away from the Colab UI.

**The badge cell looks like this** (URL points back to this notebook's path on GitHub via Colab's `/github/<user>/<repo>/blob/<branch>/<path>` route):

```markdown
<a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/<URL-encoded-path>" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
```

**Don't hand-edit notebooks for this.** Use the helper script — it discovers notebooks, URL-encodes paths correctly (spaces → `%20`, parentheses kept as-is to match GitHub's style), and is idempotent (skips notebooks that already have the badge).

```bash
# Process every .ipynb in the repo
python3 scripts/add_colab_badge.py

# Process a specific module
python3 scripts/add_colab_badge.py "Data Foundation"
python3 scripts/add_colab_badge.py "5.ML Coding (CV)"

# Process specific files
python3 scripts/add_colab_badge.py path/to/notebook.ipynb
```

**Rules:**
- Every new notebook gets the badge before being committed — run the script as the last step of the "new notebook" workflow above.
- The badge **must be the first cell** (a markdown cell). The script inserts at index 0; don't reorder it afterward.
- If a notebook is **renamed or moved**, the embedded URL is now wrong. Delete the existing badge cell and re-run the script so it regenerates the URL against the new path.
- The script's `GH_USER`, `GH_REPO`, and `BRANCH` constants live at the top of `scripts/add_colab_badge.py` — update them if the repo is ever forked or renamed.

## Mental models — required for every concept

Every concept introduced in a guide must come with a **mental model**: a one-sentence analogy, visual, or rule-of-thumb that the brain remembers when the formal definition won't. Mental models go in the module's 🪜 section *and* are sprinkled inline next to any concept that earns one in a cheat sheet, Q&A, or gotcha.

**Conventions:**
- Lead with the analogy, then the formal definition, then code. Never the reverse.
- Keep it short — one sentence, one image, one rule.
- Use proper-noun names when one already exists ("the axis-that-disappears mantra", "right-align shapes", "tea-room analogy"). Don't reinvent.
- Mental models should be transferable — the reader should be able to recite it under interview pressure.

**Canonical mental models already in the repo (extend, don't duplicate).** This list grows as new modules land. When a new module is created (e.g., Supervised Learning, CV, NLP, GenAI), add a new category here with its own canonical analogies before — or as — the first guide for that module is written.

NumPy
- **Tea-room analogy** — A Python list scatters ingredients across the room (pointer-chase); NumPy lines them up on one shelf (contiguous memory). Explains the 10–100× speedup.
- **Race ends *at* the finish line, not after it** — slicing end is exclusive. `arr[0:5]` is 5 elements.
- **Axis-that-disappears** — `axis=k` is the dimension that collapses. `axis=0` collapses rows → one value per column.
- **Right-align shapes** — for broadcasting, line up shapes from the right; dims must be equal or one of them must be 1.
- **View = window, copy = clone** — basic slicing returns a view (shared memory); fancy/boolean indexing returns a copy.
- **Strides are bytes-per-step** — transpose just swaps strides, so `.T` is O(1).

Pandas
- **Series is a column with a name; DataFrame is a spreadsheet with row labels.**
- **`iloc` is the row number, `loc` is the row name.** They diverge after `set_index`, sort, or filter.
- **Split-apply-combine** — GroupBy = "shuffle into bins, run a function in each bin, glue results back."
- **Wide is human-readable; long is computer-readable.** `melt` goes wide→long, `pivot_table` goes long→wide.
- **One-step `.loc` assignment is the safe one** — `df.loc[mask, 'col'] = v` avoids `SettingWithCopyWarning`.

Probability
- **Bayes = update belief.** Start with a prior, multiply by likelihood, normalize.
- **Mutually exclusive ≠ independent — they're closer to opposites.** Mutually exclusive (with positive P) implies dependent.
- **Conditional probability changes the denominator** from "total" to "given event."
- **2×2 contingency table disambiguates everything.** When marginal/joint/conditional are tangled, draw the table.

Visualization
- **Choose by question, not by chart name** — "what's the distribution?" → histogram; "what's the relationship?" → scatter; "what's the spread per group?" → boxplot.
- **Histogram shows shape; scatter shows relationship; boxplot shows the 5-number summary at a glance.**
- **Hue = categorical 3rd dim; size = numeric 3rd dim.**

When a new topic comes in (e.g., supervised learning), invent or borrow a mental model on the way in — don't write a guide without one.

## Basic → Advanced ladder

For each headline concept, provide a **3-step ladder**. This lets the same guide serve both first-pass review and senior-interview pressure.

1. **Basic** — the one-liner truth. Smallest possible example. Shows what the concept *does*.
2. **Intermediate** — a real-world usage on actual data. Shows how to *apply* it.
3. **Advanced** — an edge case, performance variant, or interview-trap version. Shows what *breaks* and what a senior reader would catch.

**Format template:**

````markdown
### Broadcasting

**Basic** — match shapes by stretching the smaller one (no copy).
```python
np.array([1, 2, 3]) + 10        # (3,) + scalar → (3,)
```

**Intermediate** — subtract per-column mean from a data matrix.
```python
data - data.mean(axis=0)        # (n, k) - (k,) → (n, k)
```

**Advanced** — per-row centering needs `keepdims` or `[:, None]`, else broadcasting fails.
```python
data - data.mean(axis=1, keepdims=True)   # (n, k) - (n, 1) → OK
data - data.mean(axis=1)                  # (n, k) - (n,)   → ValueError
```
````

Rules of thumb:
- Every concept in the cheat sheet earns a 3-step ladder *somewhere* in the guide (module body or appendix), even if the cheat sheet itself stays one-line.
- "Advanced" must surface a real failure mode, not just a fancier example. If it's just "with bigger data," it isn't advanced.
- Don't ladder concepts that don't need it (trivial syntax like `df.head()`).
- Where the existing guides already follow the ladder (e.g., broadcasting in `Data_Foundation_Revision_Guide.md`), match the format.

## Guided definitions — the Concept Definition Template (REQUIRED)

The repo owner is **a beginner** in AI/ML. A definition has failed if the reader has to open a second tab to understand it. Every headline concept (the ones that appear in 📖 Guided concept walkthrough, the glossary, or get their own 🪞 ladder) must be introduced with the **Concept Definition Template** — a five-part block that answers, in order, *what / why / how / where / related*.

This template **replaces and supersedes** the older "intuition-first" 5-line format. It exists because one-sentence definitions force the reader to re-search the concept elsewhere, which is exactly what these notes are meant to prevent.

### The template

````markdown
### <Concept name>

> **🪜 Mental model:** <one-sentence analogy or rule-of-thumb the brain remembers under pressure.>

**What it is.** <2–4 sentences. Plain English first, then the precise/formal definition. If the term has an alternate name or acronym, introduce it here. Use no jargon that hasn't been defined earlier in the guide — and if you must, define it parenthetically on the spot.>

**Why it matters.** <2–3 sentences. What problem does this concept solve? What would go wrong, or what would you have to do by hand, if it didn't exist? Tie it to a real interview/work scenario when possible.>

**How it works.** <3–6 sentences or a short bulleted breakdown. Walk the reader through the mechanics in plain English: inputs → what happens → outputs. If there's a formula, write the formula AND translate every symbol into words. If there's an algorithm, list the steps.>

**Where it's used.** <2–3 sentences or 3–5 bullets. Concrete real-world / ML-pipeline examples: "in EDA, you use this to ___"; "in scikit-learn, this shows up as ___"; "FAANG interview rounds ask this when ___".>

**Related terms.** <Bulleted mini-glossary, 3–6 entries, each one line. Cover: (a) sibling concepts the reader will confuse this with, (b) parent/umbrella terms it belongs under, (c) the named gotcha or trap. Each entry: `**Term** — one-line plain-English meaning + (link to its own definition if it has one elsewhere in the repo).`>

```python
# Smallest possible code example — 1–4 lines.
```

**Gotcha.** <1 line. The single most common way beginners trip over this.>
````

### Worked example — `np.where`

> **🪜 Mental model:** *vectorized if-statement* — pick element-wise between two options based on a condition, with no Python loop.

**What it is.** `np.where` is a NumPy function that walks through an array and, at each position, picks one of two values depending on whether a condition is true or false there. Think of it as the array-friendly version of Python's `if/else`. It has two distinct call forms: `np.where(cond, a, b)` returns a new array of the chosen values, while `np.where(cond)` (single argument) returns the *indices* of the True positions.

**Why it matters.** Looping over arrays in Python is slow and verbose; for any conditional transformation on numeric data, `np.where` lets you express the logic in one line that runs in compiled C. It's the canonical answer to "how do I apply an if/else across a whole column" in interviews, and the foundation for `df['col'].where()` / `np.select` later.

**How it works.**
1. Broadcasts `cond`, `a`, and `b` to a common shape (so scalars and arrays mix freely).
2. For every index, evaluates `cond` at that index.
3. If True, picks the value from `a`; if False, picks from `b`.
4. Returns a new array (never modifies inputs in place).
5. In single-argument form, it instead returns a tuple of index arrays (one per dimension) — useful with fancy indexing.

**Where it's used.**
- Building target labels: `np.where(score > 0.5, 1, 0)`.
- Imputing values: `np.where(np.isnan(x), median, x)`.
- Conditional column creation in pandas: `df['bucket'] = np.where(df['age'] >= 18, 'adult', 'minor')`.
- Finding hits: `np.where(arr == target)[0]` to locate matches.

**Related terms.**
- **`np.select`** — multi-branch version of `np.where` (more than two outcomes).
- **`np.clip`** — special case for "if value is outside [lo, hi], replace it."
- **Boolean masking** (`arr[arr > 0]`) — sibling technique; returns only matching elements rather than picking between two values.
- **`pd.Series.where`** — pandas variant; *keeps* the value where the condition is True, replaces where False (note: opposite mental default from `np.where`).
- **Vectorization** — the umbrella idea that lets `np.where` exist; "no Python loop, run in C."

```python
import numpy as np
x = np.array([-1, 2, -3, 4])
np.where(x > 0, x, 0)       # → array([0, 2, 0, 4])
np.where(x > 0)             # → (array([1, 3]),)   ← indices, not values
```

**Gotcha.** Single-argument `np.where(cond)` returns *indices*, not values — and it's a tuple, so you usually need `[0]`. Confusing it with the three-argument form is the #1 trip-up.

### Style guards for definitions

- **Plain English first, formal definition second.** If the reader can't restate the *what* in their own words after the first sentence, the sentence is wrong.
- **Define jargon inline the first time it appears, every time.** Even if the term was defined two sections earlier — beginners read non-linearly. A short parenthetical (`(loss = how wrong the model's prediction is)`) costs nothing.
- **One analogy per concept.** Pick one and commit. Mixing two confuses more than it clarifies.
- **No "as we saw above" / "trivially" / "obviously".** These are tells that the writer skipped the explanation.
- **Every formula gets a word-by-word translation.** "Let $\hat{y} = Wx + b$" must be followed by "i.e., the prediction is the input multiplied by the weights, plus a bias."
- **Where-it's-used must be concrete.** "Used in many ML applications" is banned. Name the pipeline step or the sklearn class.
- **Related terms must include the easy-to-confuse sibling.** If a concept has a famous lookalike (precision vs recall, bias vs variance, `loc` vs `iloc`), the lookalike is always in the related-terms block with a one-line distinction.

### When to use the full template vs a short form

| Where the concept appears | Treatment |
|---|---|
| 📖 Guided concept walkthrough (first appearance) | **Full template** — all five parts. |
| 📚 Terms glossary | **2–4 sentence definition** (what + a hint of why). Link back to the full walkthrough entry. |
| 🧠 Concept cheat sheet | **2–3 line row** — what + when-you-use-it. Never a naked one-liner. |
| 🪞 Basic → Intermediate → Advanced | Code-focused; assumes the template has already done the teaching above. |
| 🎯 Q&A / 🔁 Drill | Terse — questions and short answers. The template is what got the reader *ready* for these. |
| ⚠️ Gotchas | One-line trap leads. The full "why" lives in the related-terms block of the template. |

### Coverage rule

Every concept that earns its own anchor, glossary entry, or ladder entry **must** have a Concept Definition Template entry somewhere in the guide (almost always in the module's 📖 walkthrough section). If a term is introduced in passing without one, either (a) add the template, or (b) inline a one-sentence plain-English gloss right where the term is used.

## Interview question sources — MANDATORY auto-fetch on every notes request

**Why this matters.** The repo owner wants every revision guide to be a *self-contained* practice surface. They should never need to leave the guide to browse GitHub, StrataScratch, or LeetCode for the questions related to that topic. **Save them the trip.** Every time the user asks for notes on a topic, the workflow must:

1. **Fetch** real interview questions on that topic from the canonical sources below (use `WebFetch`, `WebSearch`, or `Bash`+`curl` against the GitHub raw URLs when your training data is stale or thin).
2. **Paraphrase** each to interview-pressure tightness — *never paste long verbatim text*.
3. **Embed** them inside the guide's 🎯 Q&A blocks, 🔁 Drill, and a dedicated 🌐 Sourced interview questions sub-section (see revision-guide pattern §6).
4. **Cite** every borrowed question inline with the source.

If a new guide ships without sourced questions baked in, the convention has been broken — the user's whole reason for keeping these notes is to *not* have to browse.

**Minimums per new guide:**
- Each module's 🎯 Q&A: ≥ 5 sourced questions (in addition to original ones).
- Cross-cutting 🎯 Advanced Q&A: ≥ 10 sourced + 5 original ("design & judgment") questions.
- 🔁 Drill: target 70% sourced / 30% original, total 50–100 items.
- Dedicated 🌐 Sourced interview questions section per topic-cluster: ≥ 20 sourced questions, with the source name in the header of each batch.

**Canonical sources to fetch from:**

**GitHub repos — general DS / ML interview prep:**
- `alexeygrigorev/data-science-interviews` — Alexey Grigorev's curated questions, theory + coding.
- `khanhnamle1994/cracking-the-data-science-interview` — broad coverage, FAANG-flavored.
- `kojino/120-Data-Science-Interview-Questions` — classic 120-question bank.
- `andrewekhalel/MLQuestions` — ML-focused Q&A.
- `Sroy20/machine-learning-interview-questions` — ML interview pool.
- `chiphuyen/ml-interviews-book` — Chip Huyen's open ML-interviews book.
- `eugeneyan/applied-ml` — applied-ML case studies (use for systems / design questions).

**GitHub repos — library-specific drills:**
- `rougier/numpy-100` — 100 NumPy exercises with difficulty levels.
- `guipsamora/pandas_exercises` — graded pandas problems.
- `ajcr/100-pandas-puzzles` — 100 short pandas puzzles.
- `donnemartin/data-science-ipython-notebooks` — pandas/NumPy/Matplotlib notebooks for inspiration.

**Online platforms (paid + free tiers):**
- **StrataScratch** — real pandas/SQL questions from Amazon, Google, Meta, Microsoft, Airbnb.
- **LeetCode** — "Database" + "Pandas" tracks; algorithm bank for DSA rounds.
- **HackerRank** — SQL, Statistics, ML tracks.
- **InterviewBit** — DSA + ML mix, India-flavored.
- **DataLemur** — SQL-focused, FAANG questions.
- **Kaggle Learn** — short courses + competitions for applied ML.

**Books / open guides:**
- *Chip Huyen — Machine Learning Interviews Book* (free online).
- *Andrew Ng — Machine Learning Yearning* (free PDF).
- *Aman Chadha — interview prep notes* (search "aman.ai").

**Citation convention:**
- When borrowing a question, append the source: *"(adapted from `kojino/120-Data-Science-Interview-Questions`, Q47)"* or *"(StrataScratch, Amazon round)"*.
- When summarizing a *type* of question that's common across sources, write *"(common FAANG pandas question)"*.
- Don't paste long question text verbatim — paraphrase and condense to interview-pressure length.

**Sourcing routine when building a new guide (MANDATORY — not optional):**

1. **Pick the topic-matched sources** from the list above (e.g., for pandas → `guipsamora/pandas_exercises` + `ajcr/100-pandas-puzzles` + StrataScratch; for probability → `kojino/120-Data-Science-Interview-Questions` + `chiphuyen/ml-interviews-book`).
2. **Fetch live content** when training data is thin or stale:
   ```python
   # GitHub raw README/content fetch
   WebFetch("https://raw.githubusercontent.com/<owner>/<repo>/main/README.md", prompt="...")
   # Topic search across the web
   WebSearch("pandas groupby interview question site:github.com OR site:stratascratch.com")
   ```
   Don't fabricate plausible-sounding questions — fetch real ones, then paraphrase.
3. **Pick 20–40 questions** that map to concepts in the cheat sheet. Distribute across difficulty (easy/medium/hard).
4. **Rewrite to terse Q&A style** matching the guide's voice. Strip filler, keep the trap.
5. **Cite inline**: `*"(adapted from `kojino/120-Data-Science-Interview-Questions`, Q47)"*`.
6. **Add 5+ original "design & judgment" questions** the repos don't cover — these are the senior-interviewer questions and they're rare in public banks.
7. **Embed in three places**: each module's 🎯 Q&A (≥ 5 sourced), the cross-cutting 🎯 Advanced Q&A (≥ 10 sourced), and the dedicated 🌐 Sourced interview questions block (≥ 20 sourced).
8. **Tag with company/difficulty** when the source provides it: `*"(StrataScratch, Amazon, Hard)"*`.

**If a source is paywalled** (LeetCode premium, StrataScratch paid tier): fetch only the question titles + types from free pages, paraphrase the *pattern*, and note the platform. Don't reproduce paywalled answer content.

## Content conventions

- **APIs** are presented Python-first, with the canonical pattern in code, then a one-line gloss.
- **Concepts** (anything more than an API call) follow the **Concept Definition Template** (what / why / how / where / related) the first time they appear. See "Guided definitions".
- **Gotchas** lead with what fails or surprises: *"`.iloc[0]` ≠ `.loc[0]` after `set_index` — use position vs label deliberately."*
- **Q&A** answers are tight — one paragraph, sometimes a tiny code block. (But: a Q&A answer is allowed to *link back to* a fuller walkthrough section instead of re-explaining from scratch.)
- **Drill** items are flash-card terse: `→ **answer**`. No paragraphs.
- **Cross-references**: use relative links with URL-encoded spaces (`%20`) or the simpler `./<Folder>/file.md` form.
- **Dataset cheat sheets** — when a notebook uses a specific dataset, include a column table (`column | what it is | cleaning needed`) and 5–7 business questions answered by it.

## What NOT to do

- ❌ Don't **auto-create** new per-notebook deep-dive guides unless asked — the master guide is the default deliverable. *But* when a deep-dive guide already exists, maintain it to the same beginner-friendly standard as the master (see "Per-notebook deep-dive pattern").
- ❌ Don't ship a guide where a headline concept's first appearance is a one-line definition. First appearance = full Concept Definition Template (what / why / how / where / related). Cheat-sheet one-liners come *after* the template, never instead of it.
- ❌ Don't assume the reader already knows ML jargon. First use of any term: plain-English meaning first, formal term second, acronym expanded.
- ❌ Don't write *tutorial fluff* (hype, motivational filler, "as we've seen above") — but DO write the actual explanation. "Terse" applies to recall surfaces (cheat sheet, drill, gotchas), not to first explanations.
- ❌ Don't rename or delete existing files without explicit instruction.
- ❌ Don't include base64 images / huge outputs from notebooks in any markdown — quote a short text excerpt at most.
- ❌ Don't add emojis to code or to user-facing terminal output unless the user explicitly requests.
- ❌ Don't drop existing anchors or change anchor IDs in already-published guides — sibling guides link to them.

## Quick-reference checklist before finishing a new guide

- [ ] Top anchor + title + intro blurb + companion-guide pointers
- [ ] 🚀 Topic finder table
- [ ] 📑 Table of contents
- [ ] Every module has: 🪜 mental model + **📖 guided concept walkthrough (with Concept Definition Template entries)** + cheat sheet + 🪞 basic→advanced ladder + APIs + patterns + Q&A
- [ ] Every headline concept has a one-sentence mental model
- [ ] **Every headline concept has a full Concept Definition Template entry** (what / why / how / where / related) somewhere in the guide — typically in the 📖 walkthrough
- [ ] **No jargon goes undefined on first use** (plain English first, formal term second, acronyms expanded)
- [ ] **Cheat-sheet rows are 2–3 lines minimum** (what + when-to-use), never naked one-liners
- [ ] **Glossary entries are 2–4 sentences** (beginner-friendly, not telegraphic)
- [ ] Every non-trivial concept has a basic→intermediate→advanced ladder
- [ ] **Each module's 🎯 Q&A has ≥ 5 sourced questions** (fetched + paraphrased + cited from the canonical sources)
- [ ] **Cross-cutting 🎯 Advanced Q&A has ≥ 10 sourced + 5 original questions**
- [ ] **🌐 Sourced interview questions section present with ≥ 20 fetched-and-paraphrased questions, batched by source**
- [ ] **🔁 Revision drill is 70% sourced / 30% original**, 50–100 items total
- [ ] Sourced questions are paraphrased, not pasted; every one has an inline citation
- [ ] Explanations follow the **Concept Definition Template** order (mental model → what → why → how → where → related → code → gotcha)
- [ ] Cross-cutting: glossary, API cheat, gotchas, advanced Q&A, 🌐 sourced bank, drill, best practices, mapping
- [ ] Every section ends with `[🔝 Back to top](#top)`
- [ ] Companion guides cross-linked
- [ ] Parent README / master mapping updated to reference the new guide
- [ ] Anchors used in every TOC / Topic-finder link match the actual `<a id="...">` in the body
