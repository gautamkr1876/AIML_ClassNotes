# CLAUDE.md — AIML Class Notes

Working instructions for preparing interview-ready revision notes from class notebooks.

## Project purpose

This repo holds AI/ML class notes organized **by module**. Each module is a top-level folder under the repo root — for example `Data Foundation/`, `2.ML Coding (Supervised Learning)/`, `5.ML Coding (CV)/`, and any future module (NLP, GenAI, MLOps, …). **Module names are free-form** — sometimes prefixed with a number, sometimes not. Every convention in this file applies to **all** modules equally; nothing here is specific to any one module.

For each module we maintain:

- **Notebooks** — one per lecture, kept in a subfolder named after the lecture (free-form, may carry a short prefix like `G -`).
- **Per-notebook deep-dive guides** (optional) — `<Topic>_Interview_Prep_Guide.md` inside the notebook's own folder.
- **Master revision guides** — `<Topic>_Revision_Guide.md` at the module root, consolidating multiple notebooks for interview prep.

The audience is the repo owner studying for AI/ML interviews. Tone: terse, scannable, anchored in real interview pressure. Not tutorial prose.

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
   - **🧠 Concept cheat sheet** — table of concept ↔ one-liner.
   - **🪞 Basic → Intermediate → Advanced** — for each headline concept, three layered examples (see "Basic → Advanced ladder" below).
   - **⚙️ Top APIs** — code block listing the calls.
   - **🧩 Code patterns** — numbered idiomatic snippets (5–8).
   - **🎯 Q&A — Module N** — 5–10 numbered questions with bold-question + plain answer. **At least 5 must be fetched & adapted from the source list** (cite inline, e.g., *"(adapted from `alexeygrigorev/data-science-interviews`)"*).
   - Trailing `[🔝 Back to top](#top)`.
7. **Cross-cutting sections** after the modules:
   - **📚 Terms glossary** — alphabetical table.
   - **⚙️ API cheat sheet** — every method, grouped tables.
   - **⚠️ Gotchas & traps** — numbered list, each item leads with the trap in bold.
   - **🎯 Advanced interview Q&A** — bold-Q + plain-A, optionally sub-grouped. **≥ 10 sourced + 5 original** (auto-fetched per "Interview question sources" section).
   - **🌐 Sourced interview questions** — dedicated practice bank with **≥ 20 fetched-and-paraphrased questions** from the canonical sources, organised by source/sub-topic. Each batch carries a header naming its source. This is the section the user opens when they want to "do interview reps without browsing."
   - **🔁 Revision drill** — 50–100 flash-card Q&A items in blocks (A, B, C…). **70% sourced / 30% original.** End with "Score yourself" thresholds.
   - **✅ Best practices** — numbered, optionally sub-grouped (Performance, Correctness, Workflow, Interview-day).
   - **📦 Notebook mapping** — table of notebook ↔ what it covers ↔ which section in this guide.

Cross-link liberally between guides using relative links. Whenever a guide depends on another (e.g., the EDA guide assumes Pandas basics), add a pointer to the companion guide at the top.

## Style conventions

- **Emojis as section icons** (consistent set): 🚀 📑 🧠 ⚙️ 🧩 🎯 ⚠️ 📚 🔁 ✅ 📦 🔝.
- **Tables** for cheat sheets and quick references.
- **Code fences** for every API/snippet — always Python.
- **Bold** for key terms, gotcha leads, and Q&A questions.
- **Numbered lists** for gotchas, Q&A, code patterns, drill items.
- **Anchors** (`<a id="N-name"></a>`) before every major section; use them in TOC and Topic-finder links.
- **No tutorial-style prose.** Prefer "one-liner truth + example" over paragraphs.
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

## Intuition-first explanation style

When explaining a concept, write in this order:

1. **Analogy / mental model** — 1 sentence.
2. **Formal definition** — 1 sentence, precise.
3. **Smallest code example** — 1–3 lines.
4. **What breaks / what surprises** — 1 line.
5. **When does it actually matter** — 1 line.

**Example — explaining `np.where`:**

> Think of `np.where` as a *vectorized if-statement*. It returns an array where each position takes one value if a condition is true and another value if false: `np.where(cond, a, b)`. With one argument it returns the *indices* where the condition is true instead. The single-argument form trips people up because it's a different return type entirely. Reach for `np.where` whenever you'd otherwise write a Python loop with a per-element `if`.

The point is to land the intuition in the first 10 words. The formal precision and the gotcha come right after — both still in the first 60 seconds of reading.

**Style guard:**
- Don't start with the API signature. Start with what it *does*.
- One analogy per concept. Mixing two analogies confuses more than it clarifies.
- If you can't write the analogy in one sentence, the concept needs more thinking, not more words.

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
- **Gotchas** lead with what fails or surprises: *"`.iloc[0]` ≠ `.loc[0]` after `set_index` — use position vs label deliberately."*
- **Q&A** answers are tight — one paragraph, sometimes a tiny code block.
- **Drill** items are flash-card terse: `→ **answer**`. No paragraphs.
- **Cross-references**: use relative links with URL-encoded spaces (`%20`) or the simpler `./<Folder>/file.md` form.
- **Dataset cheat sheets** — when a notebook uses a specific dataset, include a column table (`column | what it is | cleaning needed`) and 5–7 business questions answered by it.

## What NOT to do

- ❌ Don't generate per-notebook deep-dive guides unless asked — the master guide is the default deliverable.
- ❌ Don't write tutorial-style explanations; this is revision material, not first-time learning.
- ❌ Don't rename or delete existing files without explicit instruction.
- ❌ Don't include base64 images / huge outputs from notebooks in any markdown — quote a short text excerpt at most.
- ❌ Don't add emojis to code or to user-facing terminal output unless the user explicitly requests.
- ❌ Don't drop existing anchors or change anchor IDs in already-published guides — sibling guides link to them.

## Quick-reference checklist before finishing a new guide

- [ ] Top anchor + title + intro blurb + companion-guide pointers
- [ ] 🚀 Topic finder table
- [ ] 📑 Table of contents
- [ ] Every module has: 🪜 mental model + cheat sheet + 🪞 basic→advanced ladder + APIs + patterns + Q&A
- [ ] Every headline concept has a one-sentence mental model
- [ ] Every non-trivial concept has a basic→intermediate→advanced ladder
- [ ] **Each module's 🎯 Q&A has ≥ 5 sourced questions** (fetched + paraphrased + cited from the canonical sources)
- [ ] **Cross-cutting 🎯 Advanced Q&A has ≥ 10 sourced + 5 original questions**
- [ ] **🌐 Sourced interview questions section present with ≥ 20 fetched-and-paraphrased questions, batched by source**
- [ ] **🔁 Revision drill is 70% sourced / 30% original**, 50–100 items total
- [ ] Sourced questions are paraphrased, not pasted; every one has an inline citation
- [ ] Explanations follow intuition-first order (analogy → definition → code → gotcha → when-it-matters)
- [ ] Cross-cutting: glossary, API cheat, gotchas, advanced Q&A, 🌐 sourced bank, drill, best practices, mapping
- [ ] Every section ends with `[🔝 Back to top](#top)`
- [ ] Companion guides cross-linked
- [ ] Parent README / master mapping updated to reference the new guide
- [ ] Anchors used in every TOC / Topic-finder link match the actual `<a id="...">` in the body
