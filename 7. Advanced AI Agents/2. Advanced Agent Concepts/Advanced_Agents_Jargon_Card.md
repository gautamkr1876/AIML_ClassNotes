<a id="top"></a>
# Advanced Agent Concepts — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min), then keep it in a side tab and look up unknown words in 20 seconds.
>
> **Companion:** read [`Advanced_Agents_Reading_Brief.md`](./Advanced_Agents_Reading_Brief.md) FIRST for the big picture and pattern primers. This card is just the vocabulary.
>
> **Scope:** self-improvement (self-reflection, self-correction, Self-Refine, Reflexion) + the 5 agentic-workflow patterns from Anthropic's *Building Effective Agents* (Prompt Chaining, Routing, Parallelization, Evaluator-Optimizer, Orchestrator-Worker) — all on a **Text2SQL** agent.

---

## A

**Agent (vs Workflow)** — In Anthropic's terminology, an **agent** lets the language model itself decide *which tools to call, in what order, and when to stop*; a **workflow** hardcodes those decisions in developer code. The notebook is mostly workflows with one agentic loop (the ReAct agent). Most real systems are hybrids.

**Agentic workflow** — A system where the model doesn't just answer once but plans, acts, observes, and adapts — a structured process with defined stages and a stopping condition. The notebook tours five of them. Contrast with **single-shot prompting** (one call, one chance).

**Aggregation (in parallelization)** — How you merge several parallel LLM outputs into one answer. Four strategies named: **concatenation** (glue), **comparison-and-selection** (pick best), **voting / majority rule** (consensus), **synthesizer LLM** (a model merges). The choice defines the parallelization variant.

**Anchoring bias** — An LLM's reluctance to contradict its *own* previous output. The core reason **Evaluator-Optimizer** splits generation and evaluation into two separate calls: a fresh evaluator has no stake in the answer, so it critiques honestly. Self-critique in a single call is weak — separate the roles.

## B

**BIRD benchmark** — A public Text2SQL benchmark. The notebook loads a BIRD database into MySQL as the realistic backend, plus a small `employees` demo table for quick testing.

**Bounded loop (retry budget)** — Any self-improving loop needs a **maximum retry count** (e.g., `max_retries=3`) so it can't spin forever on a query it can never fix. Used by both the refinement and Evaluator-Optimizer loops.

## C

**Classification (routing stage 1)** — Routing's first stage: decide *what kind* of input this is (billing vs technical vs general). Can be **rule-based** (keywords/regex — fast, no LLM call) or **LLM-based** (handles ambiguity — costs a call). Stage 2 is **task dispatch**.

**Complexity-based routing** — One of Routing's four types: route by *how hard* the task is (simple lookup vs multi-join), often to pick a cheap-vs-powerful model. (Others: task-based, tool-based, model-based — by what the user wants, what tools are needed, or cost/capability.)

**Critic / Critique** — An LLM call that reviews another output and reports what's wrong. The notebook's critique function returns structured JSON (`syntax_ok`, `schema_ok`, `logic_ok`, `confidence`, `issues`, `suggestion`) about a SQL query. Same idea as **evaluator**, different pattern.

## E

**Episodic memory** — A store of specific past experiences (individual events), as opposed to learned weights. In **Reflexion**, verbal self-critiques are saved in episodic memory and replayed later. The notebook's version: storing `(failed_query, error, corrected_query)` triples and injecting relevant ones into future prompts as few-shot examples ("last time `dept` failed; the fix was `department`").

**Evaluator (Evaluator-Optimizer)** — The critic half of Workflow 4: scores the generator's output against explicit criteria and returns structured PASS/FAIL + per-dimension repair instructions. Distinct from the **optimizer** (generator). Unlike parallelization's *voting* critics, the evaluator feeds a regeneration loop.

**Evaluator-Optimizer** — Workflow 4: a two-role **iterative loop** — a generator (optimizer) produces output, a separate evaluator judges it, failed evaluations flow structured feedback back for bounded retries. The **first pattern here with true iterative improvement of a single artifact**. Twin: like Prompt Chaining's repair stage but with a real feedback *loop* (chaining has no backflow).

**Execution-aware feedback** — Feeding the *actual database error* (ground truth from running the SQL) back into the next generation, not just an LLM's opinion. Used in the self-improving agent and the Evaluator-Optimizer's **execution validator**. The DB never hallucinates — a real "column doesn't exist" beats an LLM guess.

## F

**Fan-out / fan-in (scatter-gather)** — The shape of parallelization: **fan-out** = split one task into many concurrent LLM calls; **fan-in** = aggregate their outputs into one. Cost scales linearly with fan-out width; latency is bounded by the *slowest single call*, not the sum.

**Feedback-conditioned prompting** — Structuring a refinement prompt so the LLM sees, in one context window, the original query + structured critique + DB error. The feedback *conditions* the next generation, making attempt 2 strictly better-informed. The mechanism behind all the self-improvement here.

**Function calling (tool schema)** — The OpenAI format for describing a tool to the model (name, description, JSON parameter schema) so the LLM can request to call it. The notebook defines `list_tables`, `get_table_schema`, and `execute_sql` this way, plus a **tool dispatch map** routing the model's request to the real Python function.

## G

**Ground truth signal** — A source of correctness independent of the LLM's judgment. Here, **database execution** is ground truth: run the SQL, see if it works. Evaluator-Optimizer combines LLM evaluation (logic errors) with execution validation (structural errors) — both must pass to stop.

**Guardrail** — A safety check screening input/output for policy violations. Implemented as a **sectioning** parallel task: a safety screener evaluates the *user's intent* concurrently with SQL generation and can block a risky query.

## H

**Hallucination (schema hallucination)** — The model inventing something unreal — here, a table/column that doesn't exist (writing `dept` when it's `department`). The most common Text2SQL failure. Prompt chaining fights it by grounding Stage 3 only on schema names Stage 2 confirmed.

## I

**Iterative refinement** — Improving an output over repeated Generate → Feedback → Refine cycles. The **Self-Refine** paper (Madaan et al., 2023) is canonical: the *same* LLM generates, critiques, and refines (no reward model), improving outputs 5–40% in 2–3 iterations. Underlies the self-improvement layer and Evaluator-Optimizer.

## L

**LLM (Large Language Model)** — The text-in, text-out model (GPT-4o-mini here) powering every step. "An LLM call" = one request to the model.

**LLM-as-judge** — Using one LLM to evaluate another's output. The critique function, evaluator, and voting critics are all LLM-as-judge. Its weakness (anchoring bias when judging *itself*) is why Evaluator-Optimizer separates the roles.

## M

**Memory-based improvement** — Persisting past mistakes across runs and reusing them. The notebook stores `(failed_query, error, corrected_query)` triples in an **error memory** and reuses relevant ones. Turns one-off mistakes into reusable lessons. See **episodic memory**, **Reflexion**.

## O

**Optimizer** — The generator half of Evaluator-Optimizer (Workflow 4) — the call that *produces* (and re-produces) the SQL, optimizing across retries toward passing the evaluator. Not a gradient-descent optimizer; no weights change.

**Orchestrator** — The "manager" LLM in Workflow 5. Analyzes a complex task, **decides at runtime** how to split it, delegates to workers, then synthesizes. Its judgment makes or breaks the pattern.

**Orchestrator-Worker** — Workflow 5: an orchestrator LLM **dynamically** decomposes a task, dispatches subtasks to specialized worker LLMs, and synthesizes their outputs. Key twins: unlike **Routing** (one classification → one handler) it dispatches to *many* workers; unlike **Parallelization** (fixed workers) it *chooses* which workers to invoke per input.

## P

**Parallelization** — Workflow 3: run multiple LLM calls concurrently and aggregate their outputs. Two sub-patterns: **sectioning** (independent *different* subtasks) and **voting** (the *same* task run several times for consensus). Prerequisite: subtasks must be independent — if B needs A's output, that's a chain, not a fan-out.

**Persona (system prompt)** — The identity/behavior instructions given to the agent up front (role, rules, tone). The Text2SQL persona tells the model it's a careful SQL analyst that grounds queries on real schema. A recurring agentic component.

**Planner** — Turns a user question into an ordered plan of steps before execution ("1. list tables, 2. get schema, 3. write SQL, 4. run it"). The notebook's `generate_plan()` does this; the ReAct loop executes it.

**Prompt chaining** — Workflow 1: decompose a task into a **fixed sequence** of small single-responsibility LLM calls, each feeding the next. Controllable because intermediate outputs are inspectable and you can insert **validation gates**. Twin: it has *no backflow* — a bad step corrupts everything downstream (**error propagation**), which Evaluator-Optimizer fixes with a loop.

## R

**ReAct (Reason + Act)** — An agent loop alternating reasoning with tool actions: think → call a tool → observe → think again → … until done. The notebook's `run_agent()` is a ReAct loop using the message list as short-term memory — its one truly *agentic* (model-controlled) piece.

**Reflexion (Shinn et al., 2023)** — A self-improvement method where an agent writes a natural-language self-critique after a failure and stores it in a memory buffer that persists across trials. The notebook cites its idea of verbal feedback as a **"semantic gradient"** — a concrete verbal direction for improvement instead of a numeric reward. Sibling of **Self-Refine**; Reflexion emphasizes *persistent memory across attempts*.

**Routing** — Workflow 2: classify an incoming input, then dispatch it via a **route dispatch table** (`{category: handler}`) to a specialized handler (a different prompt, tool chain, or model). It's the **control plane** — decides *where* work goes, not *how*, and does **not** iterate or reflect. Twin to watch: Routing picks *one* path; Orchestrator-Worker delegates to *many* workers.

## S

**Sectioning (parallelization sub-pattern)** — Split a task into independent subtasks with **different** responsibilities that run concurrently (e.g., generate a response *while* a separate call screens it for safety). Twin to watch: sectioning = *different jobs in parallel*; **voting** = *same job repeated in parallel*.

**Self-correction** — The agent detects a *specific* defect (syntax error, wrong column) and applies a surgical fix. Narrower than **self-reflection** (open-ended reasoning about *why* it's wrong).

**Self-improvement (inference-time)** — Improving output *at run time* by evaluating and rewriting it — **no weight updates, no human intervention**. Umbrella for Part 1: self-reflection, self-correction, iterative refinement, memory, feedback-conditioned prompting. Contrast with training-time learning.

**Self-reflection** — The agent writes a natural-language critique of its *own* output before acting, articulating *why* it might be wrong ("I used `salary` but the column is `annual_salary`"). Broader and more open-ended than self-correction. Rooted in **Reflexion**.

**Self-Refine (Madaan et al., 2023)** — The canonical **iterative refinement** loop: one LLM plays generator, critic, and refiner across 2–3 passes, +5–40% with no reward model. Contrast: Reflexion emphasizes cross-trial memory.

**Semantic gradient** — Reflexion's metaphor: verbal feedback ("your GROUP BY is missing") points the next attempt in a concrete direction, like a training gradient — but it's *words*, not a numeric reward. Richer than a scalar pass/fail.

**Single-shot prompting** — One prompt, one answer, no iteration/tools/verification. The baseline the notebook improves on: a single SQL call is right ~70% of the time; agentic workflows push toward ~95% at 2–4× the calls.

**Structured output (JSON)** — Forcing the LLM to return machine-parseable JSON (e.g., `{syntax_ok, schema_ok, ...}`) instead of prose, so downstream code acts deterministically. Essential for validation gates and evaluator PASS/FAIL.

**Synthesizer LLM** — A dedicated LLM call that *merges* parallel outputs into one result. Used both as a parallelization aggregation strategy and as the orchestrator's final synthesis step in Workflow 5.

## T

**Task decomposition** — Breaking a big task into smaller focused subtasks. Behind Prompt Chaining (fixed decomposition) and Orchestrator-Worker (runtime decomposition). Reduces per-step cognitive load.

**Task dispatch (routing stage 2)** — After classification, send the input to the correct handler. Stage 1 (classify) + Stage 2 (dispatch) = the whole of Routing.

**Text2SQL** — Turning a natural-language question ("average salary per department") into executable SQL. The running example for every pattern. Common failure modes: schema hallucination, wrong joins, missing GROUP BY, ambiguous filters.

**Tool** — A Python function the agent calls to fetch data or act (`list_tables()`, `get_table_schema()`, `execute_sql()`). Exposed via a **tool schema**; invoked via a **dispatch map**.

## V

**Validation gate** — A programmatic or LLM-based check inserted *between* steps of a prompt chain to catch errors before they propagate downstream. E.g., checking a translated marketing copy for tone consistency before formatting it. The countermeasure to **error propagation**.

**Voting (parallelization sub-pattern)** — Run the **same** task multiple times (varied prompts or specialized critics), then aggregate by **majority rule** — an issue is confirmed only if enough agree, filtering any single critic's noise. The notebook uses three SQL critics (schema, aggregation, join). Twin: voting = *same job repeated*; **sectioning** = *different jobs in parallel*.

## W

**Worker** — A specialized LLM call, dispatched by the orchestrator, handling one focused subtask ("identify tables/joins," "plan the aggregation," "handle the subquery"). Workers in **parallelization** are fixed; in **Orchestrator-Worker** they're chosen at runtime.

**Workflow** — See **Agent (vs Workflow)**. A predefined, developer-controlled code path orchestrating LLMs and tools. All five numbered patterns are workflows.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Advanced_Agents_Reading_Brief.md)
