<a id="top"></a>
# Advanced Agent Concepts — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every term and every pattern will already make sense — you'll be confirming what you know, not learning blind.
>
> **Side reference:** keep [`Advanced_Agents_Jargon_Card.md`](./Advanced_Agents_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `L2_ Advanced Agent Concepts.ipynb` in this folder — a big one (~19.6k words). This brief is your map.

---

## 🎯 30-second TL;DR

A single LLM call gets a hard SQL query right maybe **70% of the time**. This notebook shows how to push that toward **~95%** — not by training a bigger model, but by **wrapping the same model in smart control flow** at inference time. The cost is 2–4× more LLM calls per query; the payoff is reliability.

It does this in two acts, both built around a **Text2SQL agent** (turning "average salary per department" into runnable SQL):

1. **Self-improvement** — the agent critiques and rewrites its *own* output using natural-language feedback and a memory of past mistakes (Self-Refine + Reflexion). No weight updates.
2. **The 5 agentic-workflow patterns** from Anthropic's *Building Effective Agents* — **Prompt Chaining, Routing, Parallelization, Evaluator-Optimizer, Orchestrator-Worker** — each explained generically, then applied to Text2SQL.

The one sentence to remember: **you buy reliability with extra LLM calls arranged in the right control structure.**

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Foundation setup** — MySQL + BIRD benchmark, an `employees` demo table, and the agent's building blocks: **persona**, **tools** (`list_tables`, `get_table_schema`, `execute_sql`), **planner**, **ReAct** loop.
2. **Self-improvement — concepts** — self-reflection vs self-correction vs iterative refinement vs memory-based improvement vs feedback-conditioned prompting.
3. **Self-improvement — code** — a critique function (JSON verdict), a refinement function, execution-aware feedback, a self-improving agent with an **error memory**.
4. **Failure-case demos** — force a schema hallucination (`dept` vs `department`) and watch the correction loop fix it.
5. **Intro to agentic workflows** — the spectrum from single-shot → static chains → agentic loops; workflows vs agents; the 5 recurring components.
6. **Workflow 1 — Prompt Chaining** — marketing-copy chain, data-extraction chain, then Text2SQL as a 4-stage chain (intent → schema grounding → generation → verification).
7. **Workflow 2 — Routing** — customer-support ticket routing; then routing Text2SQL by complexity.
8. **Workflow 3 — Parallelization** — sectioning (safety guardrail) + voting (multi-critic moderation); then parallel SQL critics + safety screener.
9. **Workflow 4 — Evaluator-Optimizer** — compliance-document loop; then a generate→evaluate→execute→repair Text2SQL loop.
10. **Workflow 5 — Orchestrator-Worker** — multi-section report generation; then an orchestrator that decomposes a query and dispatches specialized SQL workers.
11. **Cross-pattern comparisons** — how each pattern differs from the others (this is the interview gold).

---

## 🧠 The big idea — control flow beats brute force

**Analogy — the newsroom.** A single LLM call is a lone freelancer asked to research, write, fact-check, and publish in one draft — roughly-right-most-of-the-time. A **newsroom** gets the same job right far more often, not because any journalist is smarter, but because of *structure*: an editor sends easy stories one way and investigations another (**routing**), several fact-checkers vet claims in parallel (**parallelization/voting**), an editor and writer trade drafts until it passes (**evaluator-optimizer**), and a managing editor breaks a big investigation into beats and assigns specialists (**orchestrator-worker**). Even one journalist revising their own draft (**self-improvement**) beats publishing draft one.

That's the whole notebook. The model never gets smarter — you arrange the *workflow* around it so mistakes get caught before reaching the reader. The five patterns are five newsroom structures; pick the one whose shape matches your problem's shape.

**The spectrum the notebook draws:**
- **Single-shot prompting** — one call, one chance. No iteration, tools, or verification.
- **Static chains** — a *fixed* sequence (outline → draft → edit). Predictable, but no way to revisit an earlier step if a later one fails.
- **Agentic workflows** — *close the loop*: generate, evaluate against a criterion, then conditionally refine or re-route. The model becomes a participant in a control loop, not a stateless function.

---

## 📖 Core concept primers

Six primers cover the notebook's heart: self-improvement, then the five patterns. Each has a **mental model**, plain-English *what*, a tiny Text2SQL example, and *why it matters here*.

### 1. Self-improvement (Self-Refine + Reflexion)

> **🪜 Mental model:** the model marks its own homework, then rewrites the wrong answers — before handing the test in.

**What it is.** The agent evaluates its own SQL against structure, meaning, and *actual execution*, then rewrites it conditioned on that feedback — all at **inference time, no weight updates**. Five flavors: **self-reflection** (critique *why* it's wrong), **self-correction** (surgically fix a specific defect), **iterative refinement** (Generate → Feedback → Refine — the **Self-Refine** loop, Madaan et al. 2023), **memory-based improvement** (store `(failed_query, error, corrected_query)` triples and reuse them — the **Reflexion** idea, Shinn et al. 2023), and **feedback-conditioned prompting** (put original query + critique + DB error in one prompt so attempt 2 is better-informed).

**Tiny example.** Attempt 1: `SELECT dept, AVG(salary) FROM employees GROUP BY dept`. Execution error: *"column 'dept' does not exist."* Critique + error fed back → Attempt 2 uses `department`. Fixed, no human, no retraining.

**Why it matters here.** Part 1 of the notebook and the foundation later patterns build on. Key number: **Self-Refine improves outputs 5–40% with just 2–3 iterations**; Reflexion's verbal feedback acts as a **"semantic gradient"** — a *worded* direction for improvement, richer than a numeric pass/fail.

### 2. Prompt Chaining (Workflow 1)

> **🪜 Mental model:** an assembly line — each station does one job and passes the part to the next.

**What it is.** Decompose a task into a **fixed sequence** of small single-responsibility LLM calls, each feeding the next. You can drop **validation gates** between stages to catch errors early. It's a *workflow, not an agent* — steps are hardwired at design time, no dynamic tool choice, no going back.

**Tiny example (the notebook's Text2SQL chain).** Four stages: **(1) intent/entity extraction** → **(2) schema grounding** (query the real DB for table/column names) → **(3) SQL generation** (using only confirmed names) → **(4) verification**. Isolating stage 2 *kills schema hallucination*: the model can't invent `dept` because stage 3 only sees names stage 2 confirmed.

**Why it matters here.** The simplest pattern and the baseline the others are compared against. Its weakness — **error propagation** (a bad step corrupts everything downstream, no backflow) — is exactly what motivates Evaluator-Optimizer.

### 3. Routing (Workflow 2)

> **🪜 Mental model:** a receptionist — reads your request, sends you to the right desk, and is done.

**What it is.** Classify the input, then **dispatch** it to a specialized handler (a different prompt, tool chain, or model). Two stages: **classification** (rule-based keywords *or* LLM-based) and **task dispatch** (a lookup table mapping category → handler). Routing is the **control plane** — it decides *where* work goes, not *how*, and does **not** reason, reflect, or iterate.

**Tiny example.** A ticket is classified `billing` / `technical` / `general` and sent to a handler tuned for that category. For Text2SQL, route by **complexity**: a simple lookup → cheap fast path; a multi-join aggregation → heavy pipeline.

**Why it matters here.** Introduces separation of concerns and the key twin: **Routing picks *one* path; Orchestrator-Worker (Workflow 5) delegates to *many* workers.** Risk: **misclassification** — a wrong route has no recovery inside the routing layer.

### 4. Parallelization — Sectioning & Voting (Workflow 3)

> **🪜 Mental model:** a jury — many members judge at once, then you tally the verdicts. (Sectioning = different specialists; Voting = same question asked of many.)

**What it is.** Run multiple LLM calls **concurrently** (scatter) and **aggregate** their outputs (gather). Prerequisite: subtasks must be **independent** — if B needs A's output, that's a chain. Two sub-patterns: **sectioning** (independent *different* jobs — generate a response *while* a separate call screens it) and **voting** (the *same* job run several times for consensus by **majority rule**).

**Tiny example (the notebook's Text2SQL).** *Voting:* three SQL critics — schema, aggregation, joins — run in parallel; an issue is confirmed only if enough agree (filters any single critic's hallucination). *Sectioning:* a safety screener evaluates the *user's intent* in parallel with SQL generation and can block a risky query — it never sees the SQL, so it can't be influenced.

**Why it matters here.** Cost scales linearly with fan-out width (4 verification calls instead of 1), but **wall-clock latency is bounded by the slowest single call**, not the sum. The pattern for "thorough verification without waiting serially."

### 5. Evaluator-Optimizer (Workflow 4)

> **🪜 Mental model:** a writer and a strict editor passing drafts back and forth until the editor stamps APPROVED.

**What it is.** A two-role **iterative loop**: a generator (**optimizer**) produces output, a *separate* **evaluator** judges it against explicit criteria and returns PASS/FAIL + per-dimension repair instructions; failures flow back for **bounded** retries. Splitting the roles beats self-critique because of **anchoring bias** — a model won't honestly contradict itself, but a fresh evaluator has no stake in the output.

**Tiny example.** Generate SQL for "average salary per department, >3 employees, sorted." Evaluator checks 5 dimensions + runs the query. If GROUP BY is missing *or* execution errors, feed the critique back and regenerate. Loop until **evaluator PASS *and* execution succeeds**, or retries run out.

**Why it matters here.** The **first pattern with genuine backflow** — the evaluator's output becomes the generator's next input. The notebook contrasts it with Workflow 1's one-shot repair (Cell 151 table): Evaluator-Optimizer re-evaluates *every* revision with both LLM judgment (logic errors) *and* DB execution (**ground truth** — structural errors). Both must pass.

### 6. Orchestrator-Worker (Workflow 5)

> **🪜 Mental model:** a project manager who reads the brief, decides *at runtime* which specialists are needed, assigns them, then stitches their work into one deliverable.

**What it is.** A two-tier delegation pattern: an **orchestrator** LLM analyzes a complex task, **dynamically decomposes** it into subtasks *based on the input*, delegates each to a specialized **worker** LLM, then **synthesizes** the outputs. The decomposition is **LLM-driven, not hardcoded**.

**Tiny example.** A complex query ("highest-paid employee per department where dept avg > 50k, subquery + HAVING") — the orchestrator spawns a *tables/joins* worker, an *aggregation* worker, and a *subquery* worker, each with full attention on one concern; then it synthesizes one query. A simple "List all employees" invokes *none* of those — the plan adapts to the input.

**Why it matters here.** The most flexible (and most expensive) pattern. Nail the twins: **vs Prompt Chaining** — stages chosen at runtime, not fixed; **vs Routing** — dispatches to *many* workers, not one; **vs Parallelization** — orchestrator *chooses* which workers to run vs a fixed set. Risk: the result rides on the orchestrator's decomposition quality.

---

## 🔥 The five patterns — at a glance

Real details from the notebook. Use this table to answer "which pattern fits this task?" — the #1 interview question for this material.

| Pattern | Shape | Iterates? | Best when… | Text2SQL use in the notebook | Main risk |
|---|---|---|---|---|---|
| **1. Prompt Chaining** | Fixed linear sequence | No (no backflow) | Task has predictable, decomposable structure | 4 stages: intent → schema grounding → generation → verification | **Error propagation** — bad step corrupts all downstream |
| **2. Routing** | Classify → dispatch to one handler | No | Different input *types* need different handling | Route by complexity (cheap path vs heavy pipeline) | **Misclassification** — wrong route, no recovery |
| **3. Parallelization** | Concurrent fan-out → aggregate | No | Subtasks are **independent** & you want thoroughness/speed | 3 parallel critics (schema/aggregation/join) + safety screener | Cost × fan-out width |
| **4. Evaluator-Optimizer** | Generate ⇄ evaluate **loop** | **Yes** | Clear criteria + output improves with feedback | Generate → evaluate (5 dims) → execute → repair, bounded retries | Loop cost; needs a hard stopping condition |
| **5. Orchestrator-Worker** | Decompose → delegate → synthesize | No (decomposes once) | Subtasks **can't be predicted** in advance | Orchestrator picks tables/aggregation/subquery workers per query | Rides entirely on decomposition quality |

**Decision shortcut:** steps known & fixed, one pass → **Prompt Chaining**; just pick *where* it goes → **Routing**; independent checks *at once* → **Parallelization**; one artifact to *iterate to correctness* → **Evaluator-Optimizer**; work structure *depends on the input* → **Orchestrator-Worker**.

---

## 🧮 Formulas to memorise — (there are none; this is a *pattern-based* lecture)

This lecture has **no load-bearing math formulas** — it's about control-flow patterns and design judgment, not equations. Memorise these **decision rules and distinctions** instead; they're what interviews probe.

**Rule 1 — the reliability/cost trade.**
`single-shot ≈ 70% correct → agentic workflow ≈ 95% correct, at 2–4× the LLM calls.`
In words: you buy reliability with extra calls arranged in control flow. Always ask whether the extra cost is justified for the use case.

**Rule 2 — parallelization economics.**
`cost = O(fan-out width);  latency = slowest single call (not the sum).`
In words: running N critics costs N× the tokens, but because they run at the same time, you only wait as long as the slowest one — not N× the wait.

**Rule 3 — the famous twins (say these out loud):**
- **Sectioning vs Voting** — *different* jobs in parallel vs the *same* job repeated for consensus.
- **Routing vs Orchestrator-Worker** — pick *one* path vs delegate to *many* workers.
- **Parallelization vs Orchestrator-Worker** — *fixed* worker set vs orchestrator *chooses* workers at runtime.
- **Prompt Chaining vs Evaluator-Optimizer** — one-way pipeline (no backflow) vs closed feedback loop.
- **Self-reflection vs self-correction** — open-ended "*why* is this wrong?" vs surgical fix of a specific defect.

**Rule 4 — when Evaluator-Optimizer is justified (all three must hold):**
`(1) clear articulable criteria  AND  (2) output demonstrably improves with feedback  AND  (3) one pass isn't reliably enough.`

**Rule 5 — dual validation (Evaluator-Optimizer's edge):**
`stop only when  LLM-evaluation = PASS  AND  DB-execution = success.`
The LLM catches *logic* errors, execution catches *structural* errors (bad columns, syntax) — need both; execution is the un-foolable **ground truth**.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–25** | MySQL + BIRD setup, `employees` demo table, DB connectors | **Skim** — ~4 min. Plumbing; note `employees` exists for demos. |
| **26–49** | Foundation agent: persona, tools, tool schema, planner, **ReAct** loop | **Read normally** — ~8 min. The base agent every later pattern reuses. |
| **51–54** | Self-improvement **concepts** (the 5 distinctions) | **FOCUS** — ~6 min. Map each term to the Jargon Card. |
| **55–62** | Self-improvement **code**: critique, refine, feedback, error memory, forced-hallucination demo | **Read carefully** — ~10 min. Watch the `dept`→`department` trace. |
| **63–65** | Agentic-workflows intro: spectrum, workflows-vs-agents, 5 components | **FOCUS** — ~5 min. The framing for everything after. |
| **66–96** | **W1 Prompt Chaining** — demos + 4-stage Text2SQL chain | Definition + Text2SQL; skim demos — ~8 min. |
| **97–105** | **W2 Routing** — ticket + complexity routing | Definition + Text2SQL — ~5 min. |
| **106–132** | **W3 Parallelization** — sectioning + voting, parallel SQL critics | **FOCUS** — ~9 min. The sectioning-vs-voting twin lives here. |
| **133–156** | **W4 Evaluator-Optimizer** — loop + Cell 151 comparison table | **FOCUS** — ~9 min. The Cell 151 table is interview fuel. |
| **157–end** | **W5 Orchestrator-Worker** — report demo + SQL orchestrator/workers | **FOCUS** — ~8 min. Nail why it ≠ routing and ≠ parallelization. |

**Total notebook read time:** ~80 min if you skim the generic demos and focus on definitions + Text2SQL + comparison tables. Add this brief's ~22 min ≈ **~100 min**, vs the 150+ min of reading the raw 19.6k-word notebook cold.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **What "inference-time self-improvement" means** — critique + rewrite at run time, no weight updates; name Self-Refine and Reflexion.
- [ ] **The five patterns** and each one's *shape* (linear / branch / concurrent / loop / delegate).
- [ ] **Which pattern fits a task** — the decision shortcut (fixed steps → chaining, pick-a-path → routing, independent checks → parallelization, iterate-to-correct → evaluator-optimizer, input-dependent structure → orchestrator-worker).
- [ ] **The five twin distinctions** — sectioning/voting, routing/orchestrator-worker, parallelization/orchestrator-worker, chaining/evaluator-optimizer, self-reflection/self-correction.
- [ ] **Why splitting generator and evaluator beats self-critique** — anchoring bias.
- [ ] **Why execution is the ground-truth signal** — the DB can't hallucinate; it catches structural errors an LLM misses.
- [ ] **The core trade-off** — reliability (70% → ~95%) bought with 2–4× more LLM calls.

If any feel shaky, revisit the matching primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. A task has a fixed, predictable set of steps and you only need one pass. Which pattern, and what's its main risk?
2. You want three independent reviewers to check a SQL query for schema, aggregation, and join errors *at the same time*. Which pattern and sub-pattern is this — and how does it differ from its sibling sub-pattern?
3. Why does Evaluator-Optimizer split generation and evaluation into two separate LLM calls instead of asking one call to self-critique?
4. In one sentence each, how does **Orchestrator-Worker** differ from (a) **Routing** and (b) **Parallelization**?
5. You have a single SQL query that's sometimes wrong and you have clear pass/fail criteria plus a database to run it against. Which pattern fits, and what are the two conditions that must *both* hold before the loop is allowed to stop?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **Prompt Chaining (Workflow 1).** Fixed linear sequence of single-responsibility LLM calls, no backflow. Main risk: **error propagation** — a flawed output at step N corrupts every downstream step (countered with validation gates between stages).
2. **Parallelization (Workflow 3), voting sub-pattern** — the *same* verification job run by several critics concurrently, aggregated by majority rule. Its sibling is **sectioning**, which runs *different* independent jobs in parallel (e.g., generate a response while a separate call screens it for safety). Voting = same job repeated; sectioning = different jobs at once.
3. Because of **anchoring bias**: a single LLM call is reluctant to contradict its own output, so self-critique is weak. A *separate* evaluator has no investment in the generated answer, creating genuine adversarial tension and more honest evaluation.
4. (a) vs **Routing**: routing makes one classification and dispatches to a *single* handler; the orchestrator dispatches the same input to *multiple* workers. (b) vs **Parallelization**: parallelization runs a *fixed* set of workers regardless of input; the orchestrator *decides at runtime* which workers to invoke based on the specific query.
5. **Evaluator-Optimizer (Workflow 4).** The loop may stop only when **LLM evaluation = PASS *and* database execution = success** — LLM judgment catches logic errors, execution (the ground-truth signal) catches structural errors like bad columns or syntax. (Plus a bounded retry budget so it can't loop forever.)

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Advanced_Agents_Jargon_Card.md)
