<a id="top"></a>
# Agents: Foundations & Planning — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebooks. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Agents_Foundations_Reading_Brief.md`](./Agents_Foundations_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebooks:** `L1_ Agents_ Foundations & Planning.ipynb` (main) + `L1-text2sql-liveclass.ipynb` (the live-coded build) in this folder.

---

## A

**Agent (AI agent)** — A software system that uses an LLM as its **reasoning engine** to pursue a goal autonomously, operating in a loop: perceive → reason → act (via tools) → observe → repeat until done. The defining trait: the **LLM decides the control flow at runtime**, instead of a developer hardcoding every step. Quoted definition (Harrison Chase, LangChain): *"a system that uses an LLM to decide the control flow of an application."*

**Agentic workflow** — A multi-step process where the **model** decides what happens next (dynamic tool selection, iterative reasoning, goal-oriented looping), as opposed to a deterministic workflow where the **developer** hardcodes the path. Defining traits: model-driven decisions, dynamic tool selection, reason→act→observe loops, goal orientation.

**Agent loop (orchestration loop)** — The `while`/`for` loop that *runs* the agent: send messages to the LLM → if it returns `tool_calls`, execute them and append results → loop → when the LLM returns plain text, stop and return it. The notebook's `run_agent()`. This loop is the concrete implementation of **ReAct**.

## B

**BIRD** — A large text-to-SQL benchmark dataset (`birdsql/bird_mini_dev`). The notebook loads it into MySQL so the Text2SQL agent has a realistic database to query.

## C

**Chain-of-Thought (CoT)** — A prompting strategy where the LLM reasons step-by-step *before* answering. In an agent it operates *within* each ReAct cycle: the model thinks "I need the table first, then its columns, then the query" before acting. (Covered in depth in Module 6, Lecture 6.)

**Control flow** — The order in which steps execute. The single most important idea in this lecture: in traditional software the *developer* fixes the control flow; in an agent the *LLM* decides it at runtime.

## D

**Deterministic workflow** — A process where every branch is hardcoded at design time (`if X do Y else Z`) and runs the same path every time — rule engines, ETL pipelines, the classic flowchart. **Not** an agent: no autonomous planning, no dynamic tool selection, no reasoning loop.

## F

**Function calling (tool calling)** — The LLM capability that powers tools. The model does **not** run the function; given tool **schemas**, it outputs a structured **JSON** object naming the function and its arguments. *Your app* executes the function and feeds the result back. Only models fine-tuned for it (gpt-4o, gpt-4o-mini, Gemini 2.0, Claude) can do this reliably — base LLMs cannot.

**Function calling cycle (6 steps)** — (1) app sends prompt + tool definitions → (2) LLM decides: answer or call a tool → (3) if tool, LLM emits JSON args → (4) app executes the real function → (5) result sent back to LLM → (6) LLM produces final natural-language answer. Repeats for multi-tool tasks — this *is* the ReAct loop.

## G

**gpt-4o-mini** — The small, cheap OpenAI model used throughout, at **temperature=0.0** (SQL demands precision, not creativity).

## K

**Knowledge (four sources)** — Everything an agent can reason over: **foundational training** (static), **fine-tuning** (static), **external tools** (dynamic/live — e.g., discovering real table names), and **memory** (dynamic/session). The first two are frozen at training; the last two are what make an agent adaptive.

## L

**LLM as Brain / Planner** — Component 2. The LLM is the agent's reasoning engine: given a goal + persona, it **dynamically plans** which tools to call, in what order, and how to interpret results — revising the plan as it learns. Contrast with a static pipeline that always runs the same steps.

## M

**MCP (Model Context Protocol)** — An open standard from Anthropic for connecting LLMs to external tools/data. It collapses the **N×M** integration explosion (every model × every data source needs a custom connector) into **N+M** (each side implements MCP once). Pre-built servers exist for Drive, Slack, GitHub, Postgres, etc.

**Memory** — Component 5. **Short-term** = the conversation `messages` list within a session (lets the agent remember it already listed the tables). **Long-term** = persisted across sessions (preferences, cached schemas) living outside the context window. Managing the finite context window deliberately is **context engineering**. (Detailed in the Advanced Agents lecture.)

## O

**Orchestrator** — The plain-Python code driving the agent loop: it parses the LLM's requested tool call, runs the real function, feeds back the observation, repeats. The LLM "thinks," the orchestrator "does."

## P

**Persona** — Component 1. The agent's identity, delivered as the **system prompt**: its role, tone, boundaries (what it *can* and *cannot* do), and output format. The notebook's Text2SQL persona = "Senior SQL Developer," can run SELECTs, **cannot** DELETE/DROP/UPDATE/INSERT or guess table names. Simplest component, arguably highest-impact.

**Planning** — Decomposing a high-level goal into ordered steps *before* acting. Can be **explicit** (LLM writes the plan) or **implicit** (encoded in orchestration code). The notebook's `generate_plan()` makes the LLM output a numbered, tool-aware plan.

**Prompting strategy** — Component 3. *How* information is assembled into each LLM call (system prompt + query + prior tool outputs + memory). The two relevant paradigms: **CoT** (reason then answer) and **ReAct** (interleave reason + tool calls). For the Text2SQL agent, **ReAct is dominant**, with CoT operating inside each cycle.

## R

**ReAct (Reasoning + Acting)** — The core agent pattern: the LLM alternates **Thought** (reason about next step) → **Action** (call a tool) → **Observation** (tool result) → repeat, until it can answer. The agent loop is its implementation. (Introduced in Module 6, Lecture 6; here it drives the full Text2SQL agent.)

## S

**SOP (Standard Operating Procedure)** — Step 2 of LangChain's build framework: a human-readable, step-by-step description of how a person would do the task. Writing it surfaces the decisions, tools, and data the agent will need.

**Six-step build framework** (LangChain) — (1) define the job via 5–10 concrete examples → (2) write the SOP → (3) build a narrow MVP on the single most critical reasoning task → (4) connect real data + orchestration → (5) iterative testing (manual → automated w/ tracing) → (6) deploy as the start of a refinement loop. Core principle: prove the reasoning works *before* adding complexity.

**System prompt** — The first message sent to the LLM every turn; in practice this *is* the persona. Sets role, rules, and output format.

## T

**Text2SQL agent** — The lecture's running example: an agent that turns natural-language questions into SQL, runs them, and summarizes results. Built from all 5 components over MySQL/BIRD with 3 tools. Real-world versions cited: Salesforce **Horizon**, Uber **Finch**.

**Tool** — Component 4. A function the LLM can invoke to *act* on the world (DB query, API call, code exec). The notebook's three: `list_tables`, `get_table_schema`, `execute_sql`. A tool is a **deterministic** function called by a **non-deterministic** agent; its **description** is the interface the LLM reads to decide when to use it.

**Tool schema** — The JSON description of a tool (name, description, parameter types) that the LLM "sees." The `description` fields are critical — vague ones confuse the model. Registered in OpenAI's `tools=[...]` format; paired with a `tool_functions` dict mapping names → real Python functions.

**`tool_choice="auto"`** — The OpenAI setting that lets the LLM decide whether to call a tool or answer directly.

## W

**Workflow vs Agent (Anthropic distinction)** — **Workflows** = LLMs/tools on *predefined* code paths (predictable, for well-defined tasks). **Agents** = the *LLM* controls its own process and tool use (flexible, for open-ended tasks). Guidance: start simple; only reach for an agent when the task is open-ended and the number of steps can't be predicted. Agents trade higher latency/cost/error-surface for flexibility.

[🔝 Back to top](#top)
