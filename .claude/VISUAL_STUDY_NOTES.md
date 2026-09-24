# Visual Study Notes — pattern recipe

A **recipe** for turning one notebook into a *learning system*: a single Markdown file with
purpose-built diagrams that carries the reader from **beginner → practitioner → advanced →
senior → staff-level thinking**.

## 📋 How to ask for this (copy-paste)

```text
Create visual study notes for <notebook path>, following .claude/VISUAL_STUDY_NOTES.md
```

That one line is enough — the rest of this file supplies the standard. Useful modifiers:

| Add this | Effect |
|---|---|
| `preview it in a new file first` | writes `<name>_v2.md` + `images_v2/` so you can compare before replacing |
| `just regenerate the diagrams` | images only; prose untouched |
| `where does <number> actually come from?` | applies Rule 7 to notes that already exist |
| `keep it under N words` | overrides the 7,000-9,000 default |
| `then push and merge to main` | commit on a branch, merge, push |

**What you get without asking:** structure chosen per concept · Why-layers · dense-explainer
diagrams with syntax-highlighted code · every quoted number derived and verified · 🧠 Remember and
Mental Model boxes · a decision guide · a production-architecture extension · 28-ish interview
questions across 5 tiers · active-recall prompts · a one-page cheat sheet.

---

**Trigger phrases** (any of these means: build this pattern):
- *"Turn this notebook into visual, interview-ready study notes."*
- *"Teach me the material through the Markdown — don't just convert the notebook."*
- *"Do the same thing as the Hybrid Search notes for `<notebook>`."*
- *"I want diagrams that teach the mechanism, not decoration."*
- *"Where does that number actually come from?"* (asks for Rule 7 on an existing set of notes)

**Reference implementation** (the gold standard — read it before writing a new one):
- `8. Agentic AI Systems/12.  Hybrid Search & Advanced Retrieval/L11_and_L12_Hybrid_Search_&_Advanced_Retrieval.md`
- its 15 diagrams in the sibling `images/` folder

**Drawing toolkit:** [`scripts/diagram_kit.py`](../scripts/diagram_kit.py) — the palette, card/arrow
primitives and layout rules used by every diagram in the reference. **Use it.** Do not reinvent a
design system per notebook; visual consistency across lectures is half the value.

---

## How this differs from the other two patterns

The repo now has three notebook-derived deliverables. Do not conflate them.

| | Fast Reading Companion | **Visual Study Notes** | Deep-dive guide |
|---|---|---|---|
| Recipe | `FAST_READING_COMPANION.md` | **this file** | `CLAUDE.md` § per-notebook |
| Files | 2 (Brief + Jargon Card) | **1 `.md` + an `images/` folder** | 1 |
| When read | **before** the notebook | **instead of / after** — it replaces re-reading | after, for revision |
| Length | ~4,000–6,000 words total | **~7,000–9,000 words** | 5,000–10,000 |
| Diagrams | ≥1 mermaid | **10–15 generated PNGs** | optional |
| Structure | fixed 12 sections | **chosen per concept** | fixed template |
| Ends with | self-check | **5-tier interview bank + cheat sheet** | Q&A |

**The defining difference: this pattern has no fixed section template.** The Fast Reading Companion
is deliberately uniform so it is quick to produce and quick to skim. Visual Study Notes are the
opposite — the structure is a *teaching decision* made per concept.

---

## Rule 1 — Think before you write

Read the **entire** notebook first and build an internal map before choosing any structure:

```text
Concepts → Relationships → Algorithms → Implementation
        → Trade-offs → Architecture → Real-world usage → Interview questions
```

Then decide the shape. **Do not force every concept into the same structure.**

| The concept is… | Teach it with… |
|---|---|
| a pipeline / process | a **flow diagram**, stage by stage |
| an algorithm | a **worked example** — formula → real numbers → intuition |
| two competing options | a **comparison matrix** + evidence from the notebook |
| a system | an **architecture diagram** |
| a mathematical idea | a **visual + the formula + a word-by-word translation** |
| a coding idea | **code + what each meaningful line does** |
| a trade-off | a **decision framework**, not a recommendation |
| something genuinely hard | **two simple diagrams**, never one overloaded one |

---

## Rule 2 — The spine is the "Why" layer

The single thing that turns a list of techniques into an argument. Use it on every **load-bearing**
concept (roughly 5–7 per notebook), as a compact table:

| | |
|---|---|
| **Problem** | what situation exists |
| **Naive approach** | what you'd do without this technique |
| **Limitation** | why that isn't enough — be specific and nameable |
| **Solution** | what this technique introduces |
| **Trade-off** | what it costs or complicates |

**Why it matters:** it produces the chain that becomes the opening concept map — *sparse misses
meaning → dense blurs exact terms → two incomparable score scales → a shortlist is not an answer →
it always returns something*. The reader learns that you never *choose* a technique, you are
**forced into it** by a failure you can demonstrate.

---

## Rule 3 — Teach in layers, but only where the concept earns it

Progress naturally through these. **Skip levels that add nothing** — a trivial concept does not
need five headings.

| Level | Covers |
|---|---|
| **1 Intuition** | what it is, in plain language, with a tiny example |
| **2 Mechanics** | what happens internally; the data flow; **a diagram** |
| **3 Implementation** | the notebook's real code + what the important lines do |
| **4 Engineering** | latency, quality, cost, scaling, failure modes, trade-offs |
| **5 Staff thinking** | architecture, evaluation, observability, reliability, design decisions |

Levels 1–3 come from the notebook. Levels 4–5 are usually **extensions** — mark them (Rule 6).

---

## Rule 4 — Diagrams that teach, not decorate

> Generate a visual **when it reduces cognitive load** — when it makes something click faster than
> the text alone would.

### The rule of thumb

Ask *"what picture would make this concept click in someone's head?"*, never *"what image should I
put here?"*

### Required diagram types (roughly this set, per notebook)

| # | Type | Purpose |
|---|---|---|
| 1 | **Concept map** | the whole lesson as a chain of forced moves. Put it near the top. |
| 2 | **Mental model** | the one-sentence idea, made visual |
| 3+ | **Worked examples** | algorithms, with real numbers from the recorded run |
| — | **Comparison** | side-by-side, with evidence |
| — | **Pipelines** | processes, stage by stage |
| — | **Decision tree** | which technique, when |
| — | **Production architecture** | marked as an extension |
| last | **Final mental model** | the whole flow on one page, for the cheat sheet |

### Hard requirements

- **Real numbers from the recorded run.** A diagram quoting `4.501 vs 1.553` teaches; one saying
  "higher score" does not.
- **Every diagram ends with a `keybar()`** — the one sentence to take away.
- **Short text inside diagrams.** If it needs a paragraph, it belongs in the Markdown.
- **Descriptive alt text** on every image, written for someone who cannot see it.
- **Consistent visual language** — use `scripts/diagram_kit.py`, same palette throughout.
- **Numbered filenames** in reading order: `01_big_picture.png`, `02_mental_model.png`, …

### Never

❌ generic AI/brain/robot illustrations ❌ decorative or stock imagery ❌ giant text posters
❌ excessive icons ❌ one overloaded diagram where two simple ones would do ❌ unreadable tiny text
❌ a diagram that would fit any notebook on the topic — it must show *this* notebook's mechanism

### The dense explainer layout (preferred)

The house style for a concept diagram. **More information, fewer words.** Built with the v2
primitives in `diagram_kit.py`.

```text
┌──────────────────────────────────────────────────────────────────────┐
│ [N] Title                                       topic glyph  │ header()
│     italic one-line subtitle                                 │
├───────────────┬──────────────────────────────────────────────┤
│ railcard()    │ ① code panel        ② code panel             │ codeblock()
│  icon+bullets │   syntax-highlighted, dark, numbered steps    │ + stepbadge()
│ railcard()    │                                              │
│  icon+bullets │ ③ result / evidence rows with real numbers   │
│ panel()       │                                              │
│  formula      │ failure strip (rose) — where it breaks       │
├───────────────┴──────────────────────────────────────────────┤
│ ✦ Key Takeaway   one sentence        │ 3 short fragments     │ takeaway()
└──────────────────────────────────────────────────────────────┘
```

**Rules for this layout**

- **Code, not prose about code.** A syntax-highlighted `codeblock()` teaches more per square inch
  than three sentences describing it. Use real notebook code, trimmed to what fits.
- **Left rail = fragments.** `railcard()` bullets are noun phrases, not sentences.
  *"Cheap, CPU-only, no training."* ✅  *"It is cheap because it requires no GPU."* ❌
- **Numbered `stepbadge()`s** carry the reading order so the eye knows where to start.
- **Evidence rows with real numbers and bars** beat a table of adjectives.
- **A rose failure strip** showing where the technique breaks — every concept has one.
- **`takeaway()` is dark navy**, with an optional right-hand column of 3 short fragments.
  It replaces the lighter `keybar()` for these dense layouts.

**Worked example to copy:** [`scripts/_example_dense_explainer.py`](../scripts/_example_dense_explainer.py)
builds the BM25 figure end to end — header, two rail cards, a formula panel, two code panels,
evidence rows, a failure strip and a takeaway bar.

**Text-wrap gotcha:** use `wrap_chars(ax, avail_width, fs)` for sans text and
`wrap_chars(ax, w, fs, mono=True)` for code. Using the mono width for sans text wraps at roughly
half the correct column and silently overflows the card.

### RENDER AND LOOK AT EVERY DIAGRAM

Non-negotiable. Collisions, clipping and overflow are **invisible in the code and obvious in the
PNG**. In the reference build, **five of fifteen** diagrams had layout bugs on first render —
overflowing the canvas, an arrow routed through a card, text overlapping a callout, a node clipped
by the key bar. Every one was caught only by opening the image.

See the LAYOUT RULES docstring in `diagram_kit.py` for the specific traps.

---

## Rule 5 — Examples aggressively, formulas never alone

Every formula gets **three** things, in this order:

1. the formula,
2. a **word-by-word translation** into plain English,
3. a **worked numeric example** using real values.

```text
Formula → Example → Intuition        ✅
Formula → move on                    ❌
```

**And check your own worked examples arithmetically.** In the reference build the first RRF example
produced a **tie**, which made the ranking it displayed arbitrary and contradicted the numbers
printed beneath it. Recompute every example in Python before shipping it.

---

## Rule 6 — Source of truth, marked on every page

Two kinds of content, always visually distinguished. Put the legend near the top of the document:

| Marker | Meaning |
|---|---|
| *(unmarked)* | **From the notebook** — concepts, code, numbers and outputs as they actually ran |
| `> **⊕ Engineering context**` | **Added** — production material the notebook did not cover |

- **Never** let added context read as though the notebook taught it.
- Do not fabricate algorithms, code, benchmarks, results or claims.
- If a diagram is an extension, **put the badge inside the image** (the reference's production
  architecture carries `EXTENSION · beyond notebook`), so it survives being viewed out of context.

### The honesty rule

**Where the notebook's prose contradicts its own recorded output, say so and trust the output.**

These are the most valuable paragraphs in the document, not an embarrassment. In the reference, two
such contradictions — an eval table showing reranking *regressing* while the text claims it improves,
and a confidence gate that never fires — became the basis for three staff-level questions. Smoothing
them over is a correctness failure.

---

## Rule 7 — Derive every number you quote

**The rule:** if a number appears in the notes as evidence, the reader must be able to see **where
it came from**. A score quoted without its derivation teaches nothing — it is trivia.

Most notebooks call a library and print a result (`BM25.get_scores()`, `DOC_MATRIX @ q`). The
derivation is therefore almost always an **⊕ extension** — mark it as such. But an extension is not
a licence to guess:

### Verify, never assert

**Recompute the number yourself and check it against the notebook's recorded output.**

- If the library is installed, call it and print the intermediate parts.
- If it isn't, **re-implement the formula** and confirm it reproduces the recorded value.
- Save the check as `scripts/verify_<thing>.py` and link it from the notes, so the claim is
  reproducible rather than trusted.

*Worked precedent:* `rank_bm25` was not installed, so Okapi BM25 was re-implemented from the
published formula; it reproduced the notebook's `4.501 / 1.553 / 1.553` exactly, and only then was
the derivation written up. See [`scripts/verify_bm25_derivation.py`](../scripts/verify_bm25_derivation.py).

### What a derivation must contain

1. **The formula**, with every symbol named *and* given this corpus's actual value (`N = 20`,
   `avgdl = 27.75`, `k1 = 1.5`…).
2. **Each intermediate step** as its own small table — IDF per term, then the length penalty, then
   the per-term contribution.
3. **The arithmetic summing to the printed number**, including the rounding
   (`4.5015 → printed 4.501`).
4. **The interpretation** — which part dominated and why. *"`inc-2847` supplies 71% of the score off
   a single extra occurrence, purely because its IDF is 1.6× higher"* is the sentence that makes the
   formula mean something.

### When a number genuinely cannot be decomposed — say so

Some numbers have no meaningful breakdown, and **that fact is itself the lesson**. A cosine score is
384 learned products; no dimension means anything on its own. Do not fake a decomposition — instead:

- show the mechanism (a dot product) and a **fully worked miniature** in 3-4 dimensions where every
  product is visible,
- show **real components** from the notebook's own output if it printed any,
- then state the asymmetry plainly. *BM25's 4.501 decomposes and you can name the responsible word;
  cosine's 0.503 does not.* That single contrast justifies "sparse is explainable, dense is not" and
  explains why cosine cannot be thresholded across queries.

### Which numbers to derive

| Derive it | Don't |
|---|---|
| algorithm outputs (BM25 score, RRF score, IDF, MRR) | measured timings — nothing to derive, just label the hardware |
| anything the notes use as **evidence for a claim** | model internals that genuinely aren't decomposable (say why) |
| any metric the reader might have to compute in an interview | incidental numbers used once in passing |

---

## Rule 8 — Connect the concepts

Do not teach concepts as isolated pages. Make the links explicit:

- the **opening concept map** showing each limitation forcing the next technique,
- **backward references** ("this is the `Part 4` failure, now fixed"),
- **forward references** ("this comes back in `Part 11`"),
- a **decision guide** so the reader knows *when* to use each one — and never present one technique
  as universally superior.

---

## Required components

Beyond the per-concept structure, every Visual Study Notes document contains:

| Component | Detail |
|---|---|
| **Source-of-truth legend** | near the top |
| **Big-picture concept map** | diagram + the forced-move chain in text |
| **Contents table** | with the *teaching method* named per part |
| **Why-layer tables** | on 5–7 load-bearing concepts |
| **🧠 Remember boxes** | end of each major section — the few things to actually retain |
| **🧠 Mental Model boxes** | where a one-liner does more work than a paragraph |
| **Decision guide** | tree diagram + text |
| **Production architecture** | marked as extension |
| **Interview bank** | 5 tiers: 🟢 Beginner · 🔵 Intermediate · 🟠 Advanced · 🔴 Senior · 🟣 Staff/System Design |
| **Explain It Yourself** | 6–8 active-recall prompts |
| **One-Page Revision** | the night-before cheat sheet |

### The interview bank

- Questions must test **explaining, reasoning, designing and debugging** — not recall.
- Derive them **from the notes themselves**, so every answer is findable above.
- Give **concise** answers — something you could actually say out loud, not an essay.
- For 🟠/🔴/🟣 questions use:

  ```text
  Question → What the interviewer is testing → Expected reasoning → Strong answer
  ```

### The cheat sheet

Compress the whole lesson into: core concepts → key formulas → the numbers worth memorising →
architecture → trade-offs → common failure modes. Optimised for the night before. End with a final
visual summary if it genuinely helps.

---

## Length discipline

**Optimise for understanding, not completeness.**

```text
Understanding > completeness      Clarity > verbosity
Mental models > paragraphs        Examples > abstract explanation
Reasoning    > memorization
```

Target **7,000–9,000 words**. The reference is ~8,100 — *shorter* than the templated version it
replaced (13,100) while carrying considerably more teaching structure. If it is getting long, cut
prose, not diagrams or the Why-layers.

---

## File layout

Everything for one lecture lives **in that lecture's folder**:

```text
<Module>/<Lecture folder>/
├── <Notebook>.ipynb
├── <Notebook>.md              ← the notes, same stem as the notebook
└── images/
    ├── 01_big_picture.png
    ├── 02_mental_model.png
    └── …                       numbered in reading order
```

- Reference images relatively: `./images/07_rrf_worked_example.png`.
- **Never** base64-inline an image (`CLAUDE.md` bans it; it also breaks GitHub's render limit).
- Add a 📝 entry under the lecture in the root `README.md`.

---

## Workflow

1. **Extract the notebook to text** (never read a large `.ipynb` directly — base64 burns context).
   Strip `data:image/...` URIs. Capture markdown, code **and outputs**.
2. **Read it all**, then build the concept → relationship → trade-off → architecture map.
3. **Decide the structure per concept** using the Rule-1 table. Write the contents table first — it
   is the outline.
4. **Design the diagram set.** List concepts, decide which earn a visual, number them in reading
   order. Build with `scripts/diagram_kit.py`.
5. **Render every diagram and LOOK AT IT.** Fix collisions. Repeat.
6. **Write the notes**, weaving code into the story: *concept → visual → small example → notebook
   code → what it does → what to watch out for*.
7. **Derive the interview bank and cheat sheet from the finished notes**, not from the notebook.
8. **Validate** (below), then update the README.

---

## Final validation

- [ ] Markdown renders; fences balanced and language-tagged
- [ ] **Every image exists**; every path is relative and valid; no unused files in `images/`
- [ ] **Every image has descriptive alt text**
- [ ] No broken links; internal anchors all resolve; heading hierarchy has no level jumps
- [ ] **Every diagram was rendered and visually inspected** for collisions and clipping
- [ ] Every worked example **recomputed** and consistent with its diagram
- [ ] **Every quoted number either derived, or explicitly explained as non-decomposable**
- [ ] Derivations **verified against the recorded output**, with the check saved under `scripts/`
- [ ] No important notebook content lost — key functions and formulas preserved verbatim
- [ ] Staff-level material **clearly marked** as extension; nothing fabricated
- [ ] Prose-vs-output contradictions documented, not smoothed over
- [ ] Questions derived from the notes; answers concise and sayable
- [ ] Cheat sheet usable for rapid revision on its own
- [ ] Understandable to a beginner **and** deep enough for an advanced reader
- [ ] Secret scan clean; no credentials copied from the notebook

---

## What NOT to do

- ❌ **Don't apply a fixed template.** The structure is a teaching decision, per concept.
- ❌ **Don't ship a diagram you haven't looked at.** Five of fifteen were broken on first render.
- ❌ **Don't generate decorative images.** Every diagram must teach a mechanism.
- ❌ **Don't let added context masquerade as notebook content.**
- ❌ **Don't smooth over the notebook's own contradictions** — they're the best material you have.
- ❌ **Don't pad for length.** Cut prose before cutting diagrams or Why-layers.
- ❌ **Don't reinvent the visual system** per notebook — use `diagram_kit.py`.
- ❌ **Don't write interview answers nobody could say out loud.**
