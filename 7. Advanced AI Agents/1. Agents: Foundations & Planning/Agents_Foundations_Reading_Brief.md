<a id="top"></a>
# AI Agents — Foundations & Planning — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word will already make sense — you'll be confirming, not learning blind.
>
> **Side reference:** keep [`Agents_Foundations_Jargon_Card.md`](./Agents_Foundations_Jargon_Card.md) open in another tab. When an unknown word appears, look it up there.
> **The notebook:** `L1_ Agents_ Foundations & Planning (1).ipynb` in this folder.

---

## 🎯 30-second TL;DR

**An AI agent is just an LLM that decides its own control flow.** In a normal program, *you* write "first do X, then Y, then stop." In an agent, the **LLM** decides — at runtime — which tool to call next, in what order, and when to stop. Anchor definition (Harrison Chase, LangChain): *"a system that uses an LLM to decide the control flow of an application."*

The notebook proves this by **building a working Text2SQL agent**: you ask a plain-English question and the agent, on its own, discovers the tables, reads the schema, writes SQL, runs it, and answers — looping through five components:

**Persona → LLM/Planner → Prompting Strategy (CoT/ReAct) → Tools (function calling) → Memory.**

The biggest mental shift: **the LLM never runs the SQL itself.** It outputs *structured JSON* ("call `execute_sql` with this query"); *your code* runs it and feeds the result back. That request-execute-observe cycle, repeated, is the whole agent.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **What an AI agent is** — LLM-decides-control-flow; three industry definitions (LangChain, IBM, Google); five core characteristics.
2. **Real-world use cases** — OpenAI contracts, Salesforce Horizon (Text2SQL in Slack), Uber Finch, LinkedIn Hiring Assistant, Anthropic multi-agent research.
3. **When (and when NOT) to use agents** — simplicity first; workflow-vs-agent tradeoff.
4. **Deterministic vs agentic workflows** — the flowchart that is NOT an agent; developer-path vs model-path.
5. **How to build an agent** — LangChain's six-step framework (examples → SOP → MVP → real data → test → deploy).
6. **Database setup** — MySQL on Colab, load the BIRD benchmark, SQLAlchemy engine.
7. **Component 1 — Persona** — system prompt as identity; built as a Python dict.
8. **Component 2 — LLM as Planner** — four sources of agent knowledge; LLM generating a plan.
9. **Component 3 — Prompting Strategy** — CoT vs ReAct; why ReAct dominates.
10. **Component 4 — Tools** — function calling; the six-step cycle; three tools + JSON schemas; demo call. Addendum: MCP.
11. **Component 5 — Memory** — short-term vs long-term; the `messages` list.
12. **Putting it together** — the `run_agent()` loop on two real questions.

---

## 🧠 The big idea — who holds the steering wheel

**One analogy: printed directions vs a taxi driver.**

- A **deterministic workflow** is a **printed set of directions**: "at the 2nd light turn left, then go 400m, then turn right." It works only for the exact trip it was written for; hit a road closure and it's stuck. This is traditional software, rule engines, ETL pipelines, the `Start → Decision → Process → End` flowchart. **The developer decided the path in advance.**

- An **agent** is a **taxi driver**. You give a *destination* ("average salary in Engineering"), not turn-by-turn directions. The driver figures out the route as they go, reroutes around closures, checks a map when unsure, and knows when they've arrived. **The LLM decides the path at runtime.**

Everything hangs off this distinction. The five components are just the driver's equipment: **persona** = training and rulebook, **LLM** = brain, **prompting strategy (ReAct)** = think-then-do habit, **tools** = hands (map, meter, GPS), **memory** = short-term recall of "I already checked that street."

The honest catch: a driver costs more than a printed map (latency, cost, more ways to go wrong). So **use the map when it works** — hire the driver only for open-ended trips you can't write directions for.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook: mental model + meaning + example + why it matters.

### 1. Agent vs Workflow (the control-flow distinction)

> **🪜 Mental model:** printed directions (workflow) vs taxi driver (agent) — who decides the route, you or the model?

**What it is.** A **workflow** runs LLMs and tools along a **predefined code path** — the developer wired the branches. An **agent** lets the **LLM dynamically decide** what to do next at each step. The litmus test: *deterministic = developer controls the path; agentic = model controls the path.*

**Tiny example.** The `Start → Decision → Process → End` flowchart is a workflow: every branch is fixed. The Text2SQL agent is agentic: it decides whether to list tables, read a schema, or run a query *based on what it has learned*.

**Why it matters in this notebook.** An agent adds three things a deterministic system lacks: **autonomous planning** (decompose a new goal), **dynamic tool selection** (choose tools at runtime), and an **iterative reasoning loop** (observe → reflect → adapt). Lack all three and it's not an agent. The practical rule: **start with the simplest solution** (often one good LLM call), go agentic only when the extra latency/cost is justified.

### 2. Persona (Component 1)

> **🪜 Mental model:** the agent's job description handed over on day one — role, rules, and how to format its answers — before it hears a single question.

**What it is.** A **persona** tells the LLM *who it is* before any user query — in practice it **is the system prompt**, the first message in every conversation. Without it, an LLM answers a database question with a Wikipedia essay; with it, it stays a focused SQL specialist.

**Tiny example.** Built as a Python dictionary with four pillars:
- **Role:** "Senior SQL Developer and Data Analyst"
- **Instructions:** the workflow — list tables → get schema → write SELECT → execute
- **Rules (boundaries):** only confirmed table/column names; never guess; **never DELETE/DROP/UPDATE/INSERT**; state assumptions if unclear; retry on failure
- **Output format:** always show the SQL, the result, and a plain-English summary

`build_system_prompt(persona)` flattens that dict into the string sent to the model.

**Why it matters in this notebook.** The simplest component but arguably the most impactful. Note `temperature=0.0` — SQL needs precision, not creativity. And telling the agent what it **cannot** do (no destructive SQL, no guessing) is a guardrail baked into the prompt.

### 3. LLM as Planner + the four sources of knowledge (Component 2)

> **🪜 Mental model:** a specialist consultant — trained broadly, specialised by experience, but still looks things up live and remembers what they just found.

**What it is.** The LLM is the agent's **brain**: given a question, it produces a **plan** ("list tables → get schema of `customers` → write query") and revises it as results come in. In a normal program *you* write the plan; in an agent the **LLM writes it**.

**The four sources of agent knowledge** (the notebook's table):

| Source | Static or Dynamic | Text2SQL example |
|---|---|---|
| **Foundational training** | Static (frozen) | Knows SQL syntax, JOINs, aggregation |
| **Fine-tuning** | Static (frozen) | Better accuracy on text-to-SQL benchmarks (BIRD/Spider) |
| **External tools** | **Dynamic (live)** | Discovers actual table/column names at runtime |
| **Memory** | **Dynamic (session)** | Remembers `customers` has a `Currency` column from an earlier step |

**Why it matters in this notebook.** Foundational training and fine-tuning are **frozen** — the model can't know your specific tables from them. The agent's real power is the **dynamic** half: tools give live knowledge, memory accumulates context. The notebook demos this by asking the LLM to plan three questions of rising difficulty (count → EUR/CZK ratio → ambiguous "show me everything," which tests the persona's boundaries).

### 4. Prompting Strategy — Chain-of-Thought vs ReAct (Component 3)

> **🪜 Mental model:** CoT = "show your working" on paper; ReAct = "show your working, but you're also allowed to get up and check a reference between steps."

**What it is.** A **prompting strategy** is *how* you assemble each message to the LLM (system prompt + query + prior tool outputs + examples). Two matter for agents: **Chain-of-Thought (CoT)** — reason step-by-step internally before answering, **no tools**; and **ReAct (Reasoning + Acting)** — interleave thinking and tool calls in a loop (*think → act → observe → think again*).

**Tiny example (the notebook's comparison):**

| Aspect | Chain-of-Thought | ReAct |
|---|---|---|
| What it does | Reasons through steps, then answers | Alternates reasoning and tool execution |
| Tool usage | None — pure reasoning | Core — tools called mid-thought |
| In our agent | Plans the SQL logic ("I need a CASE WHEN for a ratio") | Drives the full loop: list → schema → write → execute |

**Why it matters in this notebook.** **ReAct dominates** here because the agent *must* touch the database at each step; CoT operates *inside* each ReAct cycle. Key takeaway: the prompting strategy isn't one prompt, it's **the architecture of every LLM call**.

### 5. Tools & Function Calling (Component 4)

> **🪜 Mental model:** the LLM is a manager who writes a work order ("run this SQL"); it never does the work — an employee (your code) does, then reports back.

**What it is.** **Tools** are deterministic functions that let the agent *act* (query a DB, hit an API, run code); **function calling** is the mechanism. The crucial subtlety, repeated three times: **the LLM does NOT execute the function.** It reads the tool schemas, decides one fits, and outputs **structured JSON** (`{"name": "execute_sql", "arguments": {"query": "..."}}`); your app runs the real function and returns the result.

**The six-step cycle:** (1) app sends the prompt **plus tool definitions**; (2) LLM decides: answer directly or use a tool; (3) if a tool, LLM emits **JSON arguments** (not text); (4) app **executes the actual function**; (5) result goes back to the LLM; (6) LLM produces a final answer (or loops again).

**Tiny example (the three tools):** `list_tables()` → `get_table_schema(table_name)` → `execute_sql(query)`. Each is registered with a JSON schema whose `description` the LLM reads to decide when to call it. A `tool_functions` dict maps the JSON name back to the real Python function.

**Why it matters in this notebook.** Two hard rules: (a) **descriptions are the interface** — write them for an LLM, not a human ("Run a SQL SELECT query and return results" beats "does stuff with data"); (b) **not all LLMs can call tools** — only models fine-tuned for it (GPT-4o, Gemini 2.0, Claude); a base model ignores them. The **MCP addendum** previews standardising this (M×N → M+N).

### 6. Memory + the Orchestration Loop (Component 5 + finale)

> **🪜 Mental model:** the context window is a whiteboard — finite; once it fills, old notes get wiped. Memory decides what to keep on the board and what to file away.

**What it is.** **Memory** lets the agent carry context across steps instead of restarting. **Short-term** = the conversation so far (system prompt, user turns, tool calls, results) — lives **inside** the context window, i.e., the **`messages` list that grows each cycle**. **Long-term** = persists **across sessions** (preferences, cached schemas), stored **outside** the window and pulled in when relevant.

**Tiny example (the without/with diagram):** *Without* memory, question 2 ("now count CZK customers") re-runs `list_tables` and `get_schema` redundantly. *With* memory, it skips straight to `execute_sql`.

**Why it matters in this notebook.** Memory turns a stateless function-caller into a coherent assistant. The **orchestration loop** (`run_agent()`) makes it concrete: seed `messages` with system prompt + question, then loop — call the LLM → if it requests tools, run them and **append results to `messages`** → repeat → **stop when the LLM answers with no tool call.** That growing list *is* the short-term memory. (Deep memory comes in Advanced Agents.)

---

## 🔥 The headline takeaway — at a glance

The notebook's payoff isn't an accuracy number — it's a **working agent** and a set of design rules. Anatomy of what gets built:

| Component | What it is here | Remember |
|---|---|---|
| **Persona** (C1) | System prompt: Senior SQL Developer, `temperature=0.0`, no destructive SQL | Persona IS the system prompt; constraints matter as much as capabilities |
| **LLM / Planner** (C2) | `gpt-4o-mini` generating a step-by-step plan | Only 2 of 4 knowledge sources are dynamic (tools + memory) |
| **Prompting** (C3) | ReAct loop, CoT inside each cycle | ReAct dominates — the agent must hit the DB every step |
| **Tools** (C4) | `list_tables` → `get_table_schema` → `execute_sql` + JSON schemas | LLM emits JSON; **your code** runs the function |
| **Memory** (C5) | The growing `messages` list | Short-term = in the window; long-term = outside it |
| **Loop** (finale) | `run_agent()` while-loop | Stop = LLM replies with **no tool call** |

**Real-world proof points cited (real numbers):**
- **Anthropic multi-agent research** beat single-agent Claude Opus by **>90%** on internal benchmarks.
- **LinkedIn Hiring Assistant** saved recruiters **~4 hrs/role**, cut profile reviews **62%**.
- **OpenAI contract agent** scaled from hundreds to **1,000+ contracts/month** without proportional headcount.
- **MCP** turns **10 models × 10 sources = 100** integrations into just **20** (N+M).

**The two live demo questions** the finished agent answers end-to-end: *"average salary in the Engineering department?"* and *"how many employees are in each city?"*

---

## 🧮 Formulas to memorise

**This lecture is almost entirely conceptual — there is only one "formula," and it's about scaling integrations.**

### MCP integration count

```
Traditional:  integrations = M × N
With MCP:      integrations = M + N
```

**In words:** without MCP, custom connectors = **number of models (M) × number of data sources (N)** — every model needs its own connector to every source. With MCP it drops to **models + sources**, because each side integrates with the protocol just **once**.

**Worked example:** 10 models × 10 sources → traditional = `100` integrations; with MCP = `20`. The gap widens fast as the numbers grow — the whole selling point of a standard protocol.

Everything else is a *pattern to internalise*: the **ReAct loop** (think → act → observe → repeat), the **function-calling cycle** (see primer 5), and the **stop condition** (halt when the LLM responds with **no tool call**).

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **0–4** | Colab badge, title, API key setup | **Skim** — 1 min. (A real key is hardcoded — an anti-pattern; use `getpass`.) |
| **5–16** | What an agent is; 3 definitions; 5 real use cases | **Read** — ~6 min. Use cases make "agent" concrete. |
| **17–30** | When (not) to use agents; deterministic vs agentic workflows | **FOCUS** — ~6 min. The conceptual core (developer-path vs model-path). |
| **31–34** | LangChain's six-step build framework | **Read** — ~3 min. "Prove the reasoning first." |
| **35–60** | MySQL install, load BIRD, SQLAlchemy engine | **Skim/reference** — ~4 min. Setup plumbing; get the *why*, not every command. |
| **61–71** | C1 — Persona (dict → system prompt → test) | **FOCUS** — ~5 min. Read the persona dict carefully. |
| **72–82** | C2 — LLM as Planner; four knowledge sources; plan generation | **Read** — ~5 min. The knowledge-sources table is exam-worthy. |
| **83–85** | C3 — Prompting Strategy (CoT vs ReAct) | **FOCUS** — ~4 min. Short but load-bearing. |
| **86–112** | C4 — Tools & function calling; 3 tools + schemas; demo; MCP addendum | **FOCUS — the heart** — ~10 min. The 6-step cycle and "LLM never executes." |
| **113–116** | C5 — Memory (short vs long, without/with diagram) | **Read** — ~4 min. Memory = the `messages` list. |
| **118–124** | Putting it together — `run_agent()` loop + 2 live demos | **FOCUS — the payoff** — ~6 min. Watch the loop and stop condition. |

**Notebook read time:** ~55 min. Plus this brief's ~22 min = **~75 min** total.

---

## ✅ Walk-away checklist

After reading the notebook, you should be able to say in your own words:

- [ ] **What separates an agent from a workflow** — the LLM (not the developer) decides control flow at runtime; agents add autonomous planning + dynamic tool selection + an iterative reasoning loop.
- [ ] **When NOT to use an agent** — when a single LLM call or fixed workflow works; agents cost more latency, money, error surface.
- [ ] **What the five components do** — Persona (system prompt), LLM (planner/brain), Prompting Strategy (CoT/ReAct), Tools (function calling), Memory (context across steps).
- [ ] **How function calling works** — the LLM emits JSON naming a tool + args; **your code executes it**; only tool-fine-tuned models can do this.
- [ ] **CoT vs ReAct** — CoT reasons then answers (no tools); ReAct interleaves reasoning with tool calls in a loop.
- [ ] **Short-term vs long-term memory** — short-term = context window (`messages` list); long-term = outside it (DB/store).
- [ ] **How the loop terminates** — it stops when the LLM returns a response with **no tool call**.

If any feel shaky after the notebook, come back to the relevant primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. **Conceptual:** In one sentence, what is the defining difference between a deterministic workflow and an agentic workflow?
2. **Conceptual:** Why does the notebook say the persona is "just a string" yet "arguably the most impactful" component?
3. **Mechanism:** When the LLM "calls a tool," what does it actually produce, and who runs the real function?
4. **Mechanism/formula:** You have 6 models and 4 data sources. How many integrations without MCP, and how many with MCP?
5. **Synthesis:** In the `run_agent()` loop, what tells the agent it's finished and should stop looping?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **In a deterministic workflow the developer hardcodes the path in advance; in an agentic workflow the LLM decides it dynamically at runtime.** (Taxi-driver vs printed-directions — plus the agent adds autonomous planning, runtime tool selection, and an observe-reflect-adapt loop.)

2. **Because the persona IS the system prompt** — a plain string sent first in every conversation. Impactful because it sets role, boundaries (no DELETE/DROP, no guessing names), workflow, and output format, turning a generic chatbot into a focused SQL specialist. `temperature=0.0` belongs here too.

3. **A structured JSON object** naming the function and arguments (`{"name": "execute_sql", "arguments": {"query": "..."}}`) — it does **not** run anything. **Your application** executes the real Python function (via the `tool_functions` map) and sends the result back. Only tool-fine-tuned models can do this.

4. **Without MCP: `6 × 4 = 24` integrations. With MCP: `6 + 4 = 10` integrations.** MCP collapses M×N custom connectors into M+N because each side integrates with the protocol once.

5. **The loop stops when the LLM returns a response with no `tool_calls`** — i.e., it's satisfied and produces a final natural-language answer instead of requesting another tool. Until then, each tool result is appended to the `messages` list (short-term memory) and the loop repeats.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Agents_Foundations_Jargon_Card.md)
