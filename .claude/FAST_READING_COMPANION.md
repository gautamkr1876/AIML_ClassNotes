# Fast Reading Companion — pattern recipe

This is a **recipe** Claude can follow whenever the user asks for a "fast reading companion", "reading brief", "jargon card", or any phrasing meaning *"help me read this notebook faster as a beginner."*

**Trigger phrases** (any of these means: build this pattern):
- *"Make a Fast Reading Companion for `<notebook>`."*
- *"Build a Jargon Card + Reading Brief for this notebook."*
- *"Do the same thing as the CV/CNN one for `<notebook>`."*
- *"This notebook is too long — make me a pre-read."*

**Reference implementation** (the gold-standard example):
- `5.ML Coding (CV)/1.Intro to CV and CNN Fundamentals/CV_CNN_Jargon_Card.md`
- `5.ML Coding (CV)/1.Intro to CV and CNN Fundamentals/CV_CNN_Reading_Brief.md`

When in doubt about format or tone, **read those two files** before writing new ones.

---

## What this pattern is

A **two-file pack** that lives in the same folder as the notebook, designed to be read **before** the notebook so the actual notebook read becomes much faster.

The user is an **ML beginner** (per `MEMORY.md`). Their biggest cost is hitting unknown AI/CV jargon mid-notebook and breaking flow to Google it. This pattern **front-loads** every term and the big-picture lesson so by the time they open the notebook, nothing is new.

This pattern is **NOT the same as a deep-dive guide** (per `CLAUDE.md`'s "Per-notebook deep-dive pattern"). A deep-dive is comprehensive (full Concept Definition Template per concept, cell-by-cell walkthrough, ≥10 Q&A items). The Fast Reading Companion is **tight, pre-read, beginner-onboarding-only**. Don't conflate them.

| | Deep-dive guide | Fast Reading Companion |
|---|---|---|
| File count | 1 | 2 |
| Length | ~5,000-10,000 words | ~3,500-5,000 words total |
| Read time | 60-120 min | 20-30 min |
| When read | After notebook (for revision) | Before notebook (for prep) |
| Concept format | Full Concept Definition Template | Condensed primer (mental model + plain English + tiny example) |
| Code coverage | Cell-by-cell | None — code lives in the notebook |
| Q&A | ≥10 items, ≥5 sourced | 5-question self-check, original |

---

## The two files

### File A — `<Topic>_Jargon_Card.md`

**Role:** Alphabetical jargon dictionary. Skimmed once (~5 min), then kept open as a side reference while reading the notebook.

**Structure:**
```
<a id="top"></a>
# <Topic> Jargon Card

> [intro: how to use this file, link to the Reading Brief]

## A
**Term** — 2-4 sentence plain-English definition. Why it matters in this notebook.

## B
...

[🔝 Back to top](#top)
```

**Entry rules:**
- Bold term + 2-4 sentence definition.
- **Plain English first**, formal term / Greek / acronym expansion second.
- **All acronyms expanded** on first use (CNN → Convolutional Neural Network).
- **Include a hint of "why it matters in THIS notebook"** where natural (e.g., "MLP gets 27% in your notebook; CNN gets 51%").
- **Disambiguate famous twins** when the term has one (max pool vs avg pool, sigmoid vs softmax, kernel vs filter — actually same thing, say so).
- **No formulas or code.** Those live in the Brief.
- Alphabetical, grouped by first letter under `##` headers.

**Coverage rule:** Every AI/ML/domain term the notebook uses must have an entry. After writing, scan the notebook's markdown cells one more time — if you find a term that's not in the card, add it.

**Length target:** 1,500-2,000 words (~30-50 entries, depending on notebook size).

### File B — `<Topic>_Reading_Brief.md`

**Role:** The pre-read brief. User reads it **once, top-to-bottom, before opening the notebook**. After this they know the punchline, the agenda, and every concept at the "I get what it does" level.

**Required sections, in this exact order:**

1. **🎯 30-second TL;DR** — the punchline. The single most important number or insight the notebook proves. Quote the notebook's own conclusion language where possible.

2. **🗺️ Agenda — what the notebook teaches, in order** — numbered list of ~8-12 items mirroring the notebook's section flow. This is the "table of contents" the user gets BEFORE entering the notebook.

3. **🧠 The big idea** — the core insight in plain English with **one transferable analogy**. Pick one analogy; don't mix two. The analogy should survive interview pressure.

4. **📖 Core concept primers** — 4-7 entries covering the heart of the notebook. Each primer = mental model + plain-English what + tiny worked example or formula + "why it matters in this notebook." These are **condensed** — NOT the full Concept Definition Template from CLAUDE.md. Aim for ~200-300 words per primer.

5. **🔥 The headline experiment / takeaway — at a glance** — a small comparison table or summary block highlighting the notebook's main result. Use **real numbers from the notebook**, not hypothetical ones.

6. **🧮 Formulas to memorise** — only the 2-4 most load-bearing ones. Each formula gets:
   - The formula itself, in a code fence.
   - **A word-by-word translation** ("output side = input side + twice padding − filter size, divided by stride, plus one").
   - One worked numeric example.

7. **🗺️ Notebook reading map** — cell-range table telling the user where to skim/focus/skip. Columns: `Cells | What it teaches | How to read`.

8. **✅ Walk-away checklist** — 5-7 checkboxes of "after the notebook, you should be able to say in your own words..."

9. **🎯 5-question self-check** — 5 questions a beginner can answer using only the Brief. Mix conceptual (2) + formula-based (2) + synthesis (1). **Answers in a `<details>` collapsible at the bottom** so they're not visible by accident.

**Length target:** 2,500-3,000 words (~18-22 min read at beginner technical pace, ~150-170 wpm).

---

## Workflow — how to build the pack

### Step 1: Extract notebook text (don't read the .ipynb directly)

Notebooks contain base64 images that burn context. Always extract text first:

```python
import json
path = "<path-to-notebook>.ipynb"
with open(path) as f: nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print(src[:2000])
```

Run via Bash. Optionally save to a `.txt` file for repeated reference.

### Step 2: Scope the notebook

Either yourself (if small) or by dispatching an `Explore` agent (if large), produce:
- **Agenda** — ordered list of topics taught.
- **Headline concepts** — every distinct concept introduced.
- **Code experiments** — what's actually demoed.
- **Jargon list** — every term a beginner would need to look up.
- **Approximate length** — word count, cell count.
- **Headline takeaway** — the ONE big lesson. Quote the notebook's conclusion text if it has one.

### Step 3: Verify numeric facts

If the notebook quotes accuracies, parameter counts, or specific architecture details, **extract them and verify**. Don't trust agent summaries on numbers — re-grep the cells. The reference implementation got bitten by mixing a hypothetical 921,600 number (1280×720 example) with the actual experiment numbers (128×128 input) — they're different scenarios. Be precise about which number applies where.

### Step 4: Write File A (Jargon Card)

- One pass for the structure, then one pass to fill entries.
- Cross-check against the jargon list from Step 2 — every term must have an entry.
- Tone: plain English first. The user is a beginner.

### Step 5: Write File B (Reading Brief)

- Follow the required section order exactly.
- For each concept primer, include a **tiny example** with real numbers from the notebook.
- For the reading map, scan the notebook's section breaks to pick the cell ranges.
- For the self-check, make sure every question is answerable using only the Brief — if a question requires the notebook, either fix the question or add to the Brief.

### Step 6: Verify

Run these checks before declaring done:

```bash
wc -w <Topic>_Jargon_Card.md <Topic>_Reading_Brief.md
```

- File A: 1,500-2,000 words (broader vocab notebooks → upper end).
- File B: 2,500-3,000 words. **If File B exceeds 3,200, trim.** Beginner read pace caps at ~25 min for the Brief; >3,200 words risks blowing the budget.
- **Jargon coverage check:** scan the notebook's markdown cells; every AI term must be in File A.
- **Self-contained check:** read File B cold (without the notebook). Every primer should be at least "I get what it does" level.
- **Self-check answerable:** the 5 questions at the end of File B must be solvable using only the Brief.
- Combined word count should be **≤ 60% of the notebook's word count** — otherwise reading both ≈ reading the notebook, defeating the purpose.

---

## Style guards (apply to both files)

Per `MEMORY.md` (`feedback_beginner_definitions.md`) and `CLAUDE.md` (the project standard for beginner-friendly explanation):

- **Plain English first, formal term second.** First use of any term: plain meaning, then bold formal name.
- **All acronyms expanded on first use.** CNN → Convolutional Neural Network. MLP → Multi-Layer Perceptron. RGB → Red, Green, Blue. Etc.
- **One analogy per concept.** Pick one; commit. Mixing two confuses more than it clarifies.
- **All formulas get a word-by-word translation.** Example: `o = (n+2p-f)/s + 1` is followed by "output side length equals input side plus twice padding minus filter size, divided by stride, plus one."
- **"Where it's used" must reference THIS notebook concretely** — name the dataset, quote the actual numbers, point at the actual experiment. Never "used in many ML applications."
- **No tutorial fluff** — no "as we've seen above", no hype, no motivational filler.
- **Reuse canonical mental models** from `CLAUDE.md`'s mental-models section (tea-room, right-align shapes, etc.) where they apply.
- **No base64 images.** If a diagram is essential, describe it in words or skip it.
- **No code dumps.** The notebook has the code. Tiny code examples (1-4 lines) are OK when they make a formula concrete.

---

## File locations

- Files go in the **same folder as the notebook**, not at the module root.
- Naming: `<Topic>_Jargon_Card.md` and `<Topic>_Reading_Brief.md` where `<Topic>` is short and matches the notebook's theme (e.g., `CV_CNN`, `RNN_LSTM`, `Transformer_Attention`).
- Cross-link the two files at the top of each.

---

## What NOT to do

- ❌ Don't auto-create the standard deep-dive guide (per `CLAUDE.md`: "Do not auto-create new per-notebook deep-dive guides unless asked"). The Fast Reading Companion is a **separate** deliverable; don't merge the two.
- ❌ Don't use the **full** Concept Definition Template (mental model → what → why → how → where → related → code → gotcha) for every concept here. That's the deep-dive standard. The Brief uses **condensed primers** (mental model + plain-English what + tiny example + "why in this notebook"). Mixing them blows the time budget.
- ❌ Don't paraphrase the notebook section-by-section — the goal is to front-load understanding, not to re-tell.
- ❌ Don't pad with code. The notebook has the code.
- ❌ Don't trust agent summaries on numeric facts. Re-extract and verify.
- ❌ Don't skip the self-check section — it's how the user validates comprehension before/after the notebook.
- ❌ Don't write entries longer than necessary in the Jargon Card. 2-4 sentences is the sweet spot. Anything longer belongs in a primer.

---

## Reusing the Jargon Card across notebooks

The Jargon Card is **reusable**. When building a companion for the **next** notebook in the same module:

1. Start from the previous notebook's Jargon Card as a base.
2. Add new terms that appear in the new notebook.
3. Update existing entries if the new notebook adds nuance (e.g., a new context for "kernel" in a different CV setting).
4. Save as a new file in the new notebook's folder (don't share files across folders — keep each notebook self-contained).

This is how the user builds a permanent, ever-growing CV / ML dictionary over time.

---

## Quick checklist before declaring done

- [ ] File A and File B both created in the notebook's folder
- [ ] File A has `<a id="top"></a>`, alphabetical headers, ≥30 entries (for a typical notebook)
- [ ] File B has all 9 required sections in order
- [ ] Self-check answers are inside a `<details>` collapsible
- [ ] Combined word count is ≤60% of the notebook's word count
- [ ] Every AI term in the notebook has an entry in File A
- [ ] All formulas in File B have word-by-word translations
- [ ] All examples use **real numbers from the notebook**, not hypothetical ones (or are clearly labelled as hypothetical)
- [ ] Both files cross-link to each other
- [ ] Tone is beginner-friendly per `MEMORY.md`'s `feedback_beginner_definitions.md`
