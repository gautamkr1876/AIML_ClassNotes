<a id="top"></a>
# Advanced Agent Concepts — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Advanced_Agents_Reading_Brief.md`](./Advanced_Agents_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `Copy of Advanced Agent Concepts.ipynb` in this folder.
> **Prereq:** the 5 agent components + ReAct loop from `../1. Agents: Foundations & Planning/`. This notebook rebuilds that Text2SQL agent, then layers the patterns below on top.

---

## A

**ACI (Agent-Computer Interface)** — Anthropic's term for the contract between the LLM and its tools (names, descriptions, parameter schemas). The notebook's guidance: invest as much in the ACI as you would in a human UI — the LLM's tool-use accuracy depends entirely on how well tools are documented.

**Agentic workflow** — A system where the LLM plans, acts, observes, and adapts toward a goal through a *structured* process (defined stages, control flow, stopping conditions) — between rigid single-shot prompting and fully open-ended autonomy. The notebook teaches the **5 canonical patterns** below.

**Aggregation (parallelization)** — How concurrent outputs are combined: majority vote, threshold logic, or a synthesizer LLM. The "gather" in scatter-gather.

## E

**Evaluator-Optimizer** — **Workflow 4.** A two-role iterative loop: a **generator (optimizer)** produces output, a separate **evaluator (critic)** scores it against explicit criteria, and structured feedback flows back into the generator for bounded retries. Splitting the roles avoids the **anchoring bias** of a model critiquing its own work. First pattern with a *true feedback loop*.

**Execution validation** — Running the generated SQL against the real database to get **ground-truth** signal (does it actually execute?). Deterministic, no LLM. Paired with LLM evaluation in Workflow 4: the LLM catches *logical* errors, execution catches *structural* ones; both must pass to stop.

## I

**Iterative refinement (Self-Refine)** — Madaan et al. (2023): the loop Generate → Feedback → Refine → repeat, with the *same* LLM playing all three roles. No separate reward model. Reported 5–40% output improvement. Underlies the notebook's self-improvement layer.

## L

**`llm_call()`** — The notebook's thin single-call wrapper (system + user prompt → text, gpt-4o-mini, temp 0). Every workflow pattern is built from repeated `llm_call`s.

## O

**Orchestrator-Worker** — **Workflow 5.** A central **orchestrator** LLM analyzes a complex task, **dynamically decomposes** it into subtasks *at runtime*, delegates each to a specialized **worker** LLM, then **synthesizes** the workers' outputs. Key distinction: the decomposition itself is LLM-decided, not hardcoded — a simple query invokes few workers, a complex one invokes many.

## P

**Parallelization** — **Workflow 3.** Distribute independent LLM work across concurrent calls, then aggregate (scatter-gather). Prerequisite: subtasks must be **independent** (if B needs A's output, that's a chain, not parallel). Two sub-patterns: **sectioning** and **voting**. Implemented with `ThreadPoolExecutor`.

**Prompt chaining** — **Workflow 1.** Decompose a task into a *fixed* sequence of single-responsibility LLM calls, each feeding the next. Improves controllability (inspectable intermediate outputs, validation gates) but the sequence is fixed at design time — a workflow, not an agent. Key risk: **error propagation** (a bad step N corrupts all downstream steps); countered by intermediate validation gates.

## R

**Reflexion** — Shinn et al. (2023): an agent stores a natural-language critique of its own output in episodic memory as a **"semantic gradient"** — a verbal direction for improvement, richer than a scalar reward. Basis for **self-reflection** in the notebook's self-improvement layer.

**Routing** — **Workflow 2.** Classify an incoming input, then dispatch it to a specialized downstream handler (a different prompt/tool chain/model). The "control plane" — decides *where* work goes, not *how* it's done. Two stages: **classification** + **dispatch**. Types: task-based, complexity-based, tool-based, model-based.

## S

**Scatter-gather** — The distributed-computing pattern behind parallelization: split a task, process pieces concurrently, consolidate results.

**Sectioning** — A **parallelization** sub-pattern: break a task into independent subtasks with *different* responsibilities run concurrently (e.g., generate a response *while* a separate call screens it for safety). Wall-clock time = `max(subtask times)`, not the sum.

**Self-correction** — The agent detects a *specific* defect (syntax error, wrong column) and applies a targeted fix. Narrower than reflection — surgical repair, not open-ended reasoning.

**Self-improvement** — The agent evaluates its own output against structural/semantic/execution signals and rewrites — *at inference time*, no weight updates, no human. Umbrella over self-reflection, self-correction, and iterative refinement.

**Self-reflection** — The agent generates a natural-language critique of its *own* output before acting on it (Reflexion-style). It articulates *why* something may be wrong ("I used `salary` but the column is `annual_salary`"), not just pass/fail.

**Semantic gradient** — A verbal, natural-language critique used as the "direction for improvement," contrasted with a numeric reward signal. From Reflexion.

**Synthesizer** — An LLM (or rule) that combines multiple parallel/worker outputs into one final result — e.g., merging worker sections into a report, or fusing critic votes.

## T

**Text2SQL (carried example)** — The running task across the notebook: NL question → SQL → result. Each workflow pattern is demonstrated generically *and* applied to Text2SQL (e.g., parallel SQL critics, evaluator-optimizer SQL loop).

## V

**Voting** — A **parallelization** sub-pattern: run the *same* task multiple times (varied prompts/temperatures/perspectives) and aggregate by majority or threshold. The notebook's example: 3 independent SQL critics (schema, aggregation, join), each focused on one dimension; an issue is confirmed only if ≥2 agree (`vote_threshold=2`).

## W

**Workflow vs Agent** — **Workflows** orchestrate LLMs/tools on predefined code paths (developer decides sequence); **Agents** let the LLM control tool order and stopping. Most production systems are **hybrids**: deterministic scaffolding with pockets of model-driven decisions. The 5 patterns are workflow-side building blocks.

**The 5 workflow patterns (Anthropic, "Building Effective Agents")** — (1) Prompt Chaining, (2) Routing, (3) Parallelization, (4) Evaluator-Optimizer, (5) Orchestrator-Worker. The notebook's spine; each is composable with the others.

[🔝 Back to top](#top)
