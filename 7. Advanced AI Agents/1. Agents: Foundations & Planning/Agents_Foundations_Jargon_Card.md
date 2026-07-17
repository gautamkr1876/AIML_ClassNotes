<a id="top"></a>
# AI Agents — Foundations & Planning Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Agents_Foundations_Reading_Brief.md`](./Agents_Foundations_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `L1_ Agents_ Foundations & Planning (1).ipynb` in this folder.

---

## A

**Agent (AI Agent)** — A software system that uses a large language model (LLM) as its reasoning engine to pursue a goal on its own, step by step. Unlike a chatbot that answers once, an agent runs in a **loop**: think → act through tools → observe → decide what's next. The notebook's anchor definition (Harrison Chase, LangChain): *"a system that uses an LLM to decide the control flow of an application."* The whole notebook builds one — a Text2SQL agent.

**Agentic workflow** — A workflow where the **model** (not the developer) decides what happens next at each step: it plans, picks tools on the fly, observes results, and adapts. Contrast the **deterministic workflow**, where every branch is hardcoded. The slogan: developer controls the path (deterministic) vs model controls the path (agentic).

**Autonomy** — The agent's ability to take multiple steps toward a goal with little human intervention. It lets an agent handle open-ended tasks (unknown number of steps), but it's also why agents cost more and can compound errors — hence guardrails.

## B

**Base LLM (base model)** — An LLM trained *only* to predict the next token. It generates fluent text but has **no built-in way to output structured function calls**, so it can't reliably use tools. Matters because only tool-fine-tuned models (GPT-4o, Claude, Gemini 2.0) can drive an agent — a base model ignores the tools.

**BIRD (benchmark)** — A well-known text-to-SQL benchmark dataset. The notebook loads it into a MySQL database so the agent has real tables to query. "Fine-tuned on BIRD or Spider" = "specialised at turning English into SQL."

## C

**Chain-of-Thought (CoT)** — A prompting strategy that nudges the LLM to reason step-by-step *before* answering. Plain English: "show your working." Here it makes the model think *"find the table, then its columns, then write the query"* instead of jumping to SQL. CoT is pure reasoning — no tools. Its action-taking cousin is **ReAct**.

**Context window** — The finite amount of text an LLM can "see" at once (system prompt + conversation + tool outputs). The notebook's analogy: a **whiteboard** — once full, old writing gets erased and details are lost. Memory systems exist to manage this scarce space.

**Context engineering** — Deliberately deciding what to keep inside the context window, what to store outside it, and what to pull back in — so the model spends its limited attention on high-signal information. The "why" behind short-term vs long-term memory.

**Control flow** — The order in which steps of a program run (which branch, which function, when to stop). The single most important idea in the whole notebook: in an agent, **the LLM decides the control flow at runtime**, instead of a developer wiring it in advance.

## D

**Deterministic** — Always produces the same output for the same input, following a fixed path. Traditional software, rule engines, and ETL pipelines are deterministic. Here **tools are deterministic** (a `list_tables()` call always behaves the same) — that reliability is what an unpredictable agent leans on.

**Deterministic workflow** — A workflow where every branch is hardcoded (`if X do Y else do Z`). The `Start → Decision → Process → End` flowchart is the classic example — explicitly **NOT** agentic: no autonomous planning, no runtime tool selection, no reasoning loop.

**Docstring** — The description text attached to a function. The docstring/`description` field is the **interface the LLM reads to decide when to call a tool**. Write it *for an LLM*, not a human — a vague "does stuff with data" confuses the model.

## E

**ETL pipeline** — Extract, Transform, Load: a fixed, repeatable data-processing pipeline. The notebook uses it as the canonical example of a **static / deterministic** workflow — the opposite of an agent, because it always runs the same steps regardless of context.

## F

**Fine-tuning** — Extra training that specialises a pretrained model on a narrower task (e.g., text-to-SQL on BIRD/Spider) — a generalist turned specialist. Like foundational training it's **static** (frozen at training time). One of the four sources of agent knowledge; specifically what makes a model better at SQL.

**Foundational training (pretraining)** — The original large-scale training that gives an LLM its broad "base intelligence": language, SQL syntax, general reasoning. **Static/frozen** — knowledge source #1 of four. Contrast **external tools**, which give *live* knowledge it couldn't have learned.

**Function calling (tool calling)** — The capability that lets an LLM use external tools in a structured way. Crucial subtlety the notebook repeats: **the LLM does NOT run the function.** It reads the tool schemas and outputs a **structured JSON object** (function name + arguments); *your application* executes the real function and feeds the result back. Only tool-fine-tuned models can do this reliably.

## G

**Guardrails** — Safety limits placed around an agent: sandboxed testing, human-in-the-loop checkpoints, clear stopping conditions, and rules about what it may not do. The notebook's persona guardrails forbid DELETE/DROP/UPDATE/INSERT and forbid guessing table names — because agent autonomy without limits invites compounding errors.

## H

**Hallucination** — When an LLM confidently produces something false or invented (e.g., a table or column name that doesn't exist). The notebook's rule *"NEVER guess or invent table or column names"* and *"ONLY use names confirmed by the schema tool"* is a direct defence against SQL hallucination.

**Human-in-the-loop** — A design where a person reviews or approves the agent's actions at key points. Cited as a guardrail (e.g., OpenAI's contract agent keeps human review in the loop) and as one reason agents can be deployed safely.

## L

**LangChain** — A popular framework for building LLM apps and agents. The notebook draws two things from it: the working definition of an agent, and a **six-step build framework** (define the job by examples → write an SOP → narrow MVP → connect real data → test iteratively → deploy as a refinement loop).

**Latency** — How long the agent takes to respond. Agents trade **higher latency and cost** for better task performance because they make several LLM calls in a loop instead of one. The notebook's "simplicity first" rule exists because of this tradeoff.

**LLM (Large Language Model)** — An AI model trained on massive text that predicts language and reasons over it. In an agent it's the **brain / planner**: it decides which tool to call, in what order, and when to stop. The notebook uses `gpt-4o-mini` throughout.

**Long-term memory** — Memory that persists **across sessions** — user preferences, past query patterns, cached schemas — stored *outside* the context window (in a database, vector store, or file) and pulled in when relevant. Contrast with **short-term memory**, which lives inside the context window and vanishes when the session ends.

## M

**MCP (Model Context Protocol)** — An open standard from Anthropic for connecting AI systems to external tools/data uniformly. Problem it solves: M models × N data sources normally needs **M×N** custom integrations; MCP turns that into **M+N** (each side integrates once). Covered as an addendum — "plug tools in without rewriting your tool-calling logic."

**Memory** — What lets an agent carry context across steps instead of restarting each time. Split into **short-term** (this session) and **long-term** (across sessions). Here, memory is the growing `messages` list that remembers earlier tool results — so the agent doesn't re-`list_tables` every question. (Deep dive in Advanced Agents.)

**MVP (Minimum Viable Product)** — In LangChain's framework, the deliberately **narrow** first version: nail the single most critical reasoning task (e.g., intent detection) on hand-fed inputs before building the full agent. Prove the reasoning works *first*.

**Multi-agent system** — Several agents splitting the work, often a supervisor directing worker sub-agents. Notebook examples: Uber's Finch (supervisor + SQL-writer), Anthropic's research system (lead + parallel sub-agents, which beat a single agent by 90%+).

## N

**N×M complexity (integration explosion)** — The scaling problem MCP solves: every model needing a custom connector to every data source. 10 models × 10 sources = 100 integrations. MCP collapses this to N+M. (Written M×N or N×M interchangeably.)

**Non-deterministic** — Given the same input, the output can vary: the agent might call a tool, answer from memory, ask a question, or hallucinate. LLMs are non-deterministic; **tools are deterministic**. The notebook frames a tool as the *contract* between the non-deterministic agent and the deterministic outside world.

## O

**Orchestration loop (agent loop)** — The `while` loop that actually runs the agent: send messages to the LLM → if it requests a tool, execute it and append the result → repeat → stop when the LLM answers with no tool call. This is the notebook's `run_agent()` function — the piece that wires persona, planner, tools, and memory into a working agent.

## P

**Persona** — The agent's identity, told to the LLM before any user query — in practice, the **system prompt**. It sets role, tone, boundaries (can/can't), and output format. The simplest component but arguably the most impactful: it turns a generic chatbot into a focused SQL specialist. Built here as a Python dict (role + instructions + rules + output_format).

**Planner (LLM as planner)** — The role the LLM plays when it produces a step-by-step plan to answer a question and revises it based on results. Contrast with a traditional program where *you* write the plan ("first X, then Y"); in an agent the **LLM writes the plan** and adapts it.

**Prompting strategy** — The blueprint for **how** information is organised and delivered to the LLM at each turn (system prompt + user query + prior tool outputs + retrieved examples). The notebook's point: response quality depends less on the model and more on how well these layers are assembled. The two strategies covered are **CoT** and **ReAct**.

## R

**ReAct (Reasoning + Acting)** — The dominant agent prompting strategy: interleave **thinking** and **doing** in a loop — *think* → *act* (call a tool) → *observe* → *think* again. It's Chain-of-Thought extended with tool calls. Here ReAct drives the whole Text2SQL loop (list → schema → write → execute); CoT operates *within* each cycle.

**Retrieval-augmented prompting / RAG** — Pulling in relevant external data (documents, examples, schemas) and adding it to the prompt so the model answers from real context, not just memory. Mentioned in the OpenAI contract-agent use case and as one way to enhance a single LLM call before reaching for a full agent.

## S

**Sandbox** — An isolated, safe environment for testing an agent so its actions can't damage real systems. Listed as an essential guardrail given agent autonomy.

**Schema (database schema / tool schema)** — Two related meanings. **Database schema:** the structure of a table (its column names/types) — the agent's `get_table_schema` tool fetches this. **Tool schema:** the JSON description of a tool (name + description + parameters) that the LLM reads to know a tool exists and how to call it. Both are "the shape of the thing."

**Short-term memory** — The conversation history within a **single session** (system prompt, user turns, tool calls, tool results). It lives **inside** the context window and is literally the `messages` list that grows each ReAct cycle. Contrast **long-term memory** (across sessions, stored externally).

**SOP (Standard Operating Procedure)** — A plain-English, step-by-step description of how a *human* would do the task. In LangChain's framework you write the SOP early because it surfaces the decisions, tools, and data sources the agent will need — before you write any agent code.

**SQLAlchemy** — A Python library that gives a uniform way to talk to databases via an "engine." The notebook creates a `db_engine` for the BIRD MySQL database so tools can connect to it.

**System prompt** — The very first message sent to the LLM in every conversation, setting behaviour before any user input. In this notebook the **persona IS the system prompt** — role, instructions, rules, and output format assembled into one string by `build_system_prompt()`.

## T

**Temperature** — A dial (0.0 to ~2.0) controlling LLM randomness. Low = precise and repeatable; high = creative and varied. The notebook fixes `temperature=0.0` deliberately, because SQL generation demands precision, not creativity — a persona-level design choice.

**Text2SQL (text-to-SQL)** — Turning a plain-English question ("How many customers pay in EUR?") into a SQL query, running it, and returning the answer. The running example for the entire notebook, and a real product pattern (Salesforce Horizon, Uber Finch).

**Tool** — A function the agent can call to *act* on the outside world (search, run code, hit an API, query a database). Without tools an LLM is "a brain in a jar." In this notebook the three tools are `list_tables`, `get_table_schema`, and `execute_sql`. Tools are **deterministic**; the agent that calls them is not.

**Tool lookup map** — A plain Python dictionary mapping tool names (as they appear in the LLM's JSON) to the actual Python functions. When the LLM says "call `execute_sql`," the app uses this map to find and run the real function. In the notebook it's `tool_functions = {...}`.

**tool_choice="auto"** — The API setting that lets the **LLM decide** whether to call a tool or just answer in text. "Auto" is what makes the behaviour agentic — you're handing the decision to the model rather than forcing a call.

## W

**Workflow** — Anthropic's term for a system where LLMs and tools follow **predefined code paths**. Good for well-defined, predictable tasks (predictable, cheaper). Contrast with **agents**, where the LLM dynamically controls its own process — better for open-ended tasks but costlier. The notebook's guidance: use the simplest thing that works; only go agentic when it measurably helps.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Agents_Foundations_Reading_Brief.md)
