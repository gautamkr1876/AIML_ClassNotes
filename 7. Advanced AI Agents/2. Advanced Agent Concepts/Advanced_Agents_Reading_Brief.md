<a id="top"></a>
# Advanced Agent Concepts — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~25 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Advanced_Agents_Jargon_Card.md`](./Advanced_Agents_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebook:** `Copy of Advanced Agent Concepts.ipynb` (~179 cells — large, which is why this pre-read exists).
> **Prereq:** the 5 agent components + ReAct loop from `../1. Agents: Foundations & Planning/`.

---

## 🎯 30-second TL;DR

Lecture 1 built *one* agent (a ReAct loop). This lecture asks: **how do you compose LLM calls into reliable systems?** The answer is the **five canonical agentic-workflow patterns** from Anthropic's *Building Effective Agents*, each demonstrated generically and then applied to the running **Text2SQL** task:

> **1. Prompt Chaining · 2. Routing · 3. Parallelization · 4. Evaluator-Optimizer · 5. Orchestrator-Worker**

Plus a **self-improvement layer**: an agent that critiques and rewrites its own SQL at inference time (Self-Refine / Reflexion). The throughline: **decompose the work and add feedback.** A single monolithic prompt asked to understand + recall schema + write SQL + verify all at once fails in predictable ways; splitting those concerns across structured stages — with validation between them — is what makes agentic systems reliable.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Foundation rebuild** — re-create Lecture 1's Text2SQL agent (persona, tools, planner, ReAct loop) as a clean base.
2. **Self-improvement layer** — self-reflection, self-correction, iterative refinement (Self-Refine, Reflexion); a `SelfImprovingSQLAgent` that critiques → executes → refines.
3. **Intro to agentic workflows** — the complexity spectrum (single-shot → static chains → workflows → agents); 5 recurring components.
4. **Workflow 1: Prompt Chaining** — fixed sequence of single-job calls; marketing-copy and data-extraction examples; 5-stage Text2SQL chain.
5. **Workflow 2: Routing** — classify then dispatch; support-ticket routing.
6. **Workflow 3: Parallelization** — sectioning + voting; safety guardrail, multi-reviewer moderation, parallel SQL critics.
7. **Workflow 4: Evaluator-Optimizer** — generator/critic feedback loop; policy drafting; Text2SQL with dual (LLM + execution) validation.
8. **Workflow 5: Orchestrator-Worker** — runtime decomposition + worker synthesis; report generation; complex Text2SQL queries.

---

## 🧠 The big idea — decompose, then add feedback

A single LLM call asked to do everything at once is like one person trying to write, fact-check, translate, and proofread a document in a single pass — they drop something. Agentic workflows are **org charts for LLM calls**: split the job so each call has one responsibility, and add checkpoints where work is verified before it flows on.

**The transferable analogy: a publishing house.** A book doesn't come from one person doing everything once. There's an **assembly line** (writer → translator → proofreader = *prompt chaining*), a **front desk** routing manuscripts to the right editor (*routing*), **multiple reviewers** reading in parallel and voting (*parallelization*), an **editor-author back-and-forth** until it passes (*evaluator-optimizer*), and a **managing editor** who breaks a big report into sections, assigns specialists, and stitches it together (*orchestrator-worker*). Each pattern is a different organizational shape for the same goal: **reliable output from fallible individual steps.**

Two distinctions run through every pattern. First, **workflow vs agent**: workflows put LLM calls on *predefined* code paths (you decide the sequence); agents let the *LLM* decide. Most real systems are hybrids. Second, **the failure mode each pattern fights**: chaining fights overloaded single prompts (but risks error propagation), routing fights one-prompt-can't-serve-all-inputs, parallelization fights slow/shallow single verification, evaluator-optimizer fights self-anchoring bias, and orchestrator-worker fights tasks too complex to decompose in advance.

---

## 📖 Core concept primers

The five patterns are the heart of the notebook. Each primer has a **mental model**, plain-English meaning, the notebook's concrete demo, and when to reach for it.

### 1. Prompt Chaining (Workflow 1)

> **🪜 Mental model:** an assembly line — each station does one job and hands the part to the next.

Decompose a task into a **fixed sequence** of single-responsibility LLM calls, each feeding the next, with optional **validation gates** between steps. The notebook's generic demos: marketing copy → translate → tone-check; and messy-ticket → extract fields → validate schema → format. The Text2SQL chain is 5 stages: **intent extraction → schema grounding → SQL generation → verification → conditional repair**. **When to use:** the task splits cleanly into ordered subtasks you can name in advance. **Watch out:** **error propagation** — a bad step N poisons everything after it, so insert validation gates. It's a *workflow*, not an agent (sequence fixed at design time). **Why it matters here:** it directly kills schema hallucination — Stage 2 grounds entity names against the *real* database, so Stage 3 can't invent columns.

### 2. Routing (Workflow 2)

> **🪜 Mental model:** a hospital triage nurse — doesn't treat you, just sends you to the right specialist.

**Classify** the input, then **dispatch** it to a specialized handler (different prompt/tools/model). It's the *control plane* — decides *where* work goes, not *how*. Classification can be rule-based (regex/keywords, fast, no LLM) or LLM-based (handles ambiguity). The notebook routes support tickets into billing / technical / general, each with a tuned handler; adding a category = one new handler + one dispatch entry. **When to use:** distinct input categories that each need different handling, where one prompt would degrade across all of them. **Watch out:** the classifier is a single point of failure — add a confidence threshold and fall back to a human below it.

### 3. Parallelization (Workflow 3)

> **🪜 Mental model:** a panel of independent reviewers reading the same paper at the same time.

Run **independent** LLM calls **concurrently** (scatter-gather, via `ThreadPoolExecutor`), then aggregate. Two sub-patterns: **sectioning** (different jobs in parallel — e.g., generate a response *while* a separate call screens the input for safety; wall-clock = the slower of the two, not the sum) and **voting** (the *same* job multiple times, aggregate by majority — e.g., 3 SQL critics for schema / aggregation / join correctness; flag an issue only if ≥2 agree). **Prerequisite:** subtasks must be independent — if B needs A's output, that's a chain, not parallel. **When to use:** independent checks/perspectives where you want speed (sectioning) or robustness via redundancy (voting). **Cost:** linear in #calls, but latency stays bounded by the slowest call.

### 4. Evaluator-Optimizer (Workflow 4)

> **🪜 Mental model:** an author and a separate editor passing drafts back and forth until it passes review.

A **generator** produces output; a *separate* **evaluator** scores it against explicit criteria; structured feedback flows back to the generator for **bounded retries**. The key reason the roles are split: a model critiquing its *own* output suffers **anchoring bias** (reluctant to contradict itself) — a separate evaluator is genuinely adversarial. The notebook's Text2SQL version adds **dual validation**: the LLM evaluator checks 5 dimensions (syntax, schema, joins, aggregation, semantic alignment) *and* the query is actually **executed** against the DB; both must pass to stop. **When to use:** clear evaluation criteria + iterative LLM improvement available. **Why it matters here:** it's the first pattern with a *true feedback loop* — and dual validation catches what neither check alone would (the notebook shows a query that *executes fine* but is *logically wrong*, caught only by the LLM evaluator's "missing HAVING clause").

### 5. Orchestrator-Worker (Workflow 5)

> **🪜 Mental model:** a managing editor who reads the brief, decides *which* specialists are needed, assigns sections, and stitches the final report.

A central **orchestrator** LLM analyzes a complex task, **decomposes it into subtasks at runtime**, delegates each to a specialized **worker** LLM, then **synthesizes** the results. The defining trait vs prompt chaining: the decomposition is **LLM-decided per input**, not hardcoded — a simple "list all employees" invokes few workers; a complex multi-constraint query invokes many (table/join planner, aggregation planner, subquery handler), and the orchestrator fuses their outputs into one query. **When to use:** tasks too complex for one prompt *and* too unpredictable for a fixed pipeline. **Why it matters here:** it's the most "agentic" workflow — the structure itself is a product of model reasoning.

---

## 🔥 The five patterns — at a glance

| # | Pattern | Shape | Fights | Text2SQL demo |
|---|---|---|---|---|
| 1 | **Prompt Chaining** | Sequential, fixed | Overloaded single prompt | intent → ground → generate → verify → repair |
| 2 | **Routing** | Branch (pick one path) | One prompt can't serve all inputs | classify ticket → specialized handler |
| 3 | **Parallelization** | Concurrent (use all) | Slow/shallow single check | 3 SQL critics vote; safety screen in parallel |
| 4 | **Evaluator-Optimizer** | Feedback loop | Self-anchoring; one-shot errors | generate ↔ evaluate (LLM + execution) until pass |
| 5 | **Orchestrator-Worker** | Delegate + synthesize | Task too complex *and* unpredictable | orchestrator splits query → workers → synthesize |

**Two more axes to keep straight:**
- **Sequential vs branch vs concurrent vs loop vs delegate** — chaining is sequential, routing branches to *one* path, parallelization runs *all* paths, evaluator-optimizer *loops*, orchestrator-worker *delegates* to a runtime-chosen set.
- **Patterns compose.** The notebook stacks them: a chain generates SQL, parallel critics verify it, an evaluator-optimizer loop refines it. They're building blocks, not either/or choices.

---

## 🧠 Bonus: the self-improvement layer

Before the 5 patterns, the notebook builds a `SelfImprovingSQLAgent` — worth knowing as its own concept:

- **Self-reflection** (Reflexion, Shinn et al. 2023): the agent writes a natural-language critique of its own output — a **"semantic gradient"** (verbal direction) rather than a numeric reward.
- **Self-correction:** detect a specific defect → targeted fix.
- **Iterative refinement** (Self-Refine, Madaan et al. 2023): Generate → Feedback → Refine with the *same* model in all roles; reported 5–40% gains.
- The agent's loop: generate SQL → **critique** (LLM → structured JSON: syntax/schema/logic/confidence) → **execute** (DB = ground truth) → if failure, **refine** with critique + DB error → repeat to `max_retries`. All at **inference time** — no weight updates, no human.

This is the conceptual bridge from one ReAct agent (Lecture 1) to the feedback-driven Evaluator-Optimizer pattern.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–48 | **Foundation rebuild**: API key, MySQL+BIRD, persona, 3 tools, planner, ReAct `run_agent` | **Skim** if Lecture 1 is fresh — it's a recap. |
| 50–61 | **Self-improvement layer**: critique → execute → refine; `SelfImprovingSQLAgent`; error memory | **Focus.** New material; bridges to Workflow 4. |
| 62–64 | **Intro to agentic workflows**: complexity spectrum; 5 components | **Focus.** Frames everything. |
| 65–95 | **Workflow 1: Prompt Chaining** — definition, 2 generic demos, 5-stage Text2SQL chain | **Focus.** Note the validation gates. |
| 96–104 | **Workflow 2: Routing** — classify + dispatch; support tickets | **Focus.** |
| 105–131 | **Workflow 3: Parallelization** — sectioning + voting; parallel SQL critics | **Focus + slow down.** Two sub-patterns. |
| 132–155 | **Workflow 4: Evaluator-Optimizer** — generator/critic loop; dual validation | **Focus.** The feedback-loop payoff. |
| 156–176 | **Workflow 5: Orchestrator-Worker** — runtime decomposition; report + complex SQL | **Focus.** Most agentic pattern. |

---

## ✅ Walk-away checklist

After the notebook, you should be able to say, in your own words:

- [ ] Name the 5 workflow patterns and the shape of each (sequential / branch / concurrent / loop / delegate).
- [ ] The failure mode each pattern is designed to fight.
- [ ] Why prompt chaining risks **error propagation**, and the countermeasure.
- [ ] The two parallelization sub-patterns (**sectioning** vs **voting**) and when each fits.
- [ ] Why Evaluator-Optimizer **separates** generator and evaluator (anchoring bias).
- [ ] Why **dual validation** (LLM + execution) catches more than either alone.
- [ ] How Orchestrator-Worker differs from prompt chaining (runtime vs design-time decomposition).
- [ ] What self-improvement (self-reflection / correction / refinement) means at inference time.

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. You have a support inbox where billing, technical, and general questions each need a different tone and toolset. Which workflow pattern fits, and what are its two stages?
2. A query *runs without error* against the database but returns the wrong answer (it's missing a `HAVING` clause). Which validation catches this — execution validation or the LLM evaluator — and why does Workflow 4 use both?
3. In parallelization, what's the difference between **sectioning** and **voting**? Give the notebook's example of each.
4. Why does Evaluator-Optimizer use a *separate* evaluator LLM instead of asking the generator to critique its own output?
5. Both Prompt Chaining and Orchestrator-Worker break a task into subtasks. What is the one key difference in *how* the subtasks are determined?

<details>
<summary>Answers</summary>

1. **Routing (Workflow 2).** Two stages: **classification** (determine the ticket category) and **dispatch** (send it to the specialized handler for that category). Add a confidence threshold so low-confidence cases go to a human.
2. The **LLM evaluator** catches it — it's a *logical/semantic* error, and the query is syntactically valid so it executes fine. Workflow 4 uses **dual validation** because execution catches *structural* errors (bad columns, syntax) that the LLM might miss, while the LLM catches *logical* errors (wrong aggregation, missing HAVING, semantic mismatch) that execution can't see. Both must pass to stop.
3. **Sectioning** = different independent jobs run concurrently (e.g., generate a response *while* a separate call screens the input for safety). **Voting** = the *same* job run multiple times and aggregated by majority (e.g., 3 SQL critics — schema, aggregation, join — where an issue is confirmed only if ≥2 agree).
4. To avoid **anchoring bias** — a model is reluctant to contradict its own output, so self-critique is weak. A separate evaluator has no investment in the generated text and evaluates more objectively, creating genuine adversarial tension.
5. **When the decomposition is decided.** Prompt chaining's stages are **fixed at design time** (the developer hardcodes the sequence). Orchestrator-Worker decides the subtasks **at runtime** — the orchestrator LLM analyzes each input and chooses which workers to invoke, so a simple query and a complex query get different decompositions.

</details>

[🔝 Back to top](#top)
