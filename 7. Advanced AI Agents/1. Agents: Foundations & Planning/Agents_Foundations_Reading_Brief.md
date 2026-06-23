<a id="top"></a>
# Agents: Foundations & Planning — Reading Brief

> **Read this ONCE, end to end, before opening the notebooks.** Target time: ~24 minutes. By the time you reach the notebooks, every word in them will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Agents_Foundations_Jargon_Card.md`](./Agents_Foundations_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebooks:** `L1_ Agents_ Foundations & Planning.ipynb` (the main, fully-explained build, ~125 cells) + `L1-text2sql-liveclass.ipynb` (the live-coded version of the same thing, ~58 cells).
> **Prereqs:** ReAct, CoT, and tools were introduced in Module 6, Lecture 6 (Prompt Engineering: Introduction).

---

## 🎯 30-second TL;DR

A chatbot answers once. An **agent** runs in a *loop* — it reasons, calls tools, looks at the results, and decides what to do next, over and over, until the goal is met. The one sentence that defines the whole lecture (Harrison Chase, LangChain):

> **"An AI agent is a system that uses an LLM to decide the control flow of an application."**

In normal software *you* write the control flow. In an agent the *LLM* decides it at runtime. The lecture builds intuition (what agents are, when to use them, deterministic vs agentic) and then constructs a **Text2SQL agent** — turn an English question into SQL, run it, summarize the answer — out of **five reusable components**:

> **Persona → LLM/Planner → Prompting Strategy → Tools → Memory** → wired together by a **ReAct agent loop**.

---

## 🗺️ Agenda — what the notebooks teach, in order

1. **What is an AI agent?** — LLM as reasoning engine; the loop; industry definitions.
2. **Real-world use cases** — OpenAI contracts, Salesforce Horizon, Uber Finch, LinkedIn Hiring Assistant, Anthropic multi-agent research.
3. **When (and when not) to use agents** — simplicity first; workflows vs agents.
4. **Deterministic vs agentic workflows** — developer-controlled path vs model-controlled path.
5. **How to build an agent** — LangChain's 6-step framework.
6. **Database setup** — MySQL + BIRD benchmark (the agent's playground).
7. **Component 1: Persona** — the system prompt that defines identity + boundaries.
8. **Component 2: LLM as Planner** — the brain that plans which tools to call.
9. **Component 3: Prompting Strategy** — CoT vs ReAct in an agent.
10. **Component 4: Tools** — function calling; the three SQL tools; schemas. (+ MCP addendum.)
11. **Component 5: Memory** — short-term vs long-term.
12. **End-to-end** — the ReAct agent loop ties all five together.

---

## 🧠 The big idea — the LLM controls the control flow

Every program has **control flow**: the order steps run in. For 70 years, a *human* wrote that order — `if this, do that`. An agent flips it: the **LLM decides, at runtime, what to do next**, based on the goal and what it has observed so far.

**The transferable analogy: a GPS with live traffic vs. printed turn-by-turn directions.** Printed directions (a **deterministic workflow**) are fixed the moment they're printed — same route every time, even into a closed road. A GPS with live traffic (an **agent**) re-decides at each junction based on what it observes, reroutes when blocked, and only stops when you arrive. The printed directions aren't "dumber" — for a known, unchanging route they're faster and more predictable. That's the lecture's core judgment call: **use a workflow (fixed path) for well-defined tasks; use an agent (model-decided path) only when the task is open-ended and you can't predict the steps in advance.** Agents buy flexibility at the cost of latency, money, and more ways to go wrong.

The Text2SQL agent makes this concrete. You ask "What's the average salary in Engineering?" Nobody hardcoded "first list tables, then read the `employees` schema, then write this exact SQL." The LLM *figures that out* — calling `list_tables`, then `get_table_schema`, then `execute_sql` — adapting if a query errors. That runtime decision-making is what makes it an agent.

---

## 📖 Core concept primers

The five agent components are the heart of the lecture. Each primer has a **mental model**, plain-English meaning, the notebook's concrete code, and why it matters.

### 1. Persona (Component 1) — who the agent is

> **🪜 Mental model:** an employee's job description pinned above their desk — role, rules, and "do not do" list, read before every task.

The **persona** is the agent's identity, delivered as the **system prompt** sent first on every turn. It sets **role** (Senior SQL Developer), **tone** (precise, no filler), **boundaries** (CAN: list tables, read schemas, run SELECTs; CANNOT: DELETE/DROP/UPDATE/INSERT, guess names, answer non-DB questions), and **output format** (show the SQL, the result, a plain-English summary). The notebook stores it as a dict and `build_system_prompt()` renders it to a string. **Why it matters here:** it's the simplest component (just a string) but the highest-leverage — telling the agent what it *cannot* do (no destructive SQL, no guessing columns) is as important as what it can. Set `temperature=0.0` for precision.

### 2. LLM as Planner (Component 2) — the brain

> **🪜 Mental model:** a senior analyst who sketches the steps before touching the keyboard — and crosses one out if it stops making sense.

The LLM is the agent's reasoning engine. Given a question + persona, it **plans dynamically**: which tools, in what order, how to read the results — revising as it goes. The notebook's `generate_plan()` asks the LLM to output a numbered, tool-aware plan ("1. list_tables, 2. get_schema(employees), 3. execute_sql(...)"). It also frames the **four sources of agent knowledge**: foundational training + fine-tuning (both *static*, frozen) vs external tools + memory (both *dynamic*, live). **Why it matters here:** the plan is what separates an agent from a fixed pipeline — a pipeline always runs the same steps; the planner decides them per question, and an *ambiguous* question (e.g., "show me everything") exposes whether the persona's boundaries hold.

### 3. Prompting Strategy (Component 3) — CoT vs ReAct

> **🪜 Mental model:** CoT is thinking before you speak; ReAct is think-a-bit, do-a-bit, look, repeat.

The prompting strategy is *how* each LLM call is assembled (system prompt + query + prior tool results + memory). Two paradigms: **Chain-of-Thought** (reason step-by-step, no tools) and **ReAct** (interleave reasoning with tool calls in a loop). **Why it matters here:** for Text2SQL, **ReAct is the dominant strategy** — the agent *must* touch the database at each step (list → schema → query → execute) — while CoT operates *inside* each ReAct cycle (the model reasons "I need a CASE WHEN for this ratio" before acting). The strategy isn't one prompt; it's the architecture of every call.

### 4. Tools (Component 4) — giving the agent hands

> **🪜 Mental model:** a contract between two worlds — a reliable, deterministic function that an unpredictable, creative agent decides *when* to call.

Tools let the agent **act** beyond generating text. The mechanism is **function calling**: given tool **schemas** (name, description, parameters as JSON), the LLM outputs a **JSON object** naming a function and its arguments — it does **not** run the function; *your app* does, then feeds the result back. The notebook builds three tools — `list_tables`, `get_table_schema`, `execute_sql` — registers them in OpenAI's `tools=[...]` format with `tool_choice="auto"`, and maps names → real functions in a `tool_functions` dict. **Why it matters here:** three critical facts the notebook hammers: (a) the **description is the interface** — write it for the LLM, not a human; (b) the LLM **never executes** the tool — it just requests it; (c) **not all LLMs can call tools** — only ones fine-tuned for it (gpt-4o-mini, Gemini 2.0, Claude). The **MCP** addendum generalizes this: a standard that turns N×M custom integrations into N+M.

### 5. Memory (Component 5) — remembering across steps

> **🪜 Mental model:** a whiteboard with finite space — short-term scribbles accumulate during the meeting; anything you'll need next week goes in a filing cabinet (long-term).

**Short-term memory** is the `messages` list that grows through the loop — system prompt, user question, each tool call, each tool result — so the LLM remembers it already discovered the `employees` table and doesn't re-list. **Long-term memory** persists across sessions (preferences, cached schemas) outside the finite **context window**; deciding what to keep, store, and retrieve is **context engineering**. **Why it matters here:** memory is what carries context across ReAct iterations — without it every loop step starts from zero. (The lecture flags that memory is covered in depth in the Advanced Agents notebook.)

### 6. Putting it together — the ReAct agent loop

> **🪜 Mental model:** the conveyor belt that connects all five components — it keeps cycling the LLM until it stops asking for tools.

`run_agent()` wires everything: seed `messages` with the persona (system) + user question (**memory**); loop — call the LLM with the **tool schemas** (**planning + strategy**); if the reply has `tool_calls`, execute each via `tool_functions` (**tools**), append results to `messages`, and loop; when the reply is plain text, that's the final answer. **Why it matters here:** this ~20-line loop *is* the agent — it's the runtime realization of ReAct and the function-calling cycle, and it's exactly the architecture every production agent (Horizon, Finch) elaborates on.

---

## 🔥 The five components — at a glance

| # | Component | One-liner | In the Text2SQL agent |
|---|---|---|---|
| 1 | **Persona** | Identity via system prompt | "Senior SQL Developer," SELECT-only, no guessing |
| 2 | **LLM / Planner** | The reasoning brain | `generate_plan()`, gpt-4o-mini @ temp 0 |
| 3 | **Prompting Strategy** | How each call is assembled | ReAct loop, CoT inside each cycle |
| 4 | **Tools** | Hands to act on the world | `list_tables`, `get_table_schema`, `execute_sql` |
| 5 | **Memory** | Context across steps | `messages` list (short-term) |
| → | **Agent loop** | Ties it all together | `run_agent()` = ReAct in code |

---

## 🧮 The one "formula" — the function-calling cycle

Not math, but the load-bearing mechanism. Memorize the 6 steps:

```
1. app → LLM:   prompt + tool definitions
2. LLM:         decide — answer directly, or call a tool?
3. LLM → app:   structured JSON {name, arguments}   (NOT a text answer)
4. app:         execute the real Python function
5. app → LLM:   send the function's result back
6. LLM → app:   final natural-language answer   (or loop to step 2 for another tool)
```

**Word-by-word translation:** "The LLM is a decision-maker that *requests* actions in JSON; your application is the hands that *perform* them and report back; repeat until the LLM stops requesting tools." **Worked reading:** ask "How many customers pay in EUR?" → step 3 returns `list_tables()` (no answer yet) → app runs it, returns table names → LLM requests `get_table_schema("customers")` → … → finally returns text with the SQL + count. That repetition of steps 2–5 *is* the ReAct loop.

---

## 🗺️ Notebook reading map

**Main notebook** (`L1_ Agents_ Foundations & Planning.ipynb`):

| Cells | What it teaches | How to read |
|---|---|---|
| 0–2 | Setup, API key (`getpass`) | **Skim.** |
| 4–15 | What is an agent; industry definitions; 5 real-world case studies | **Focus.** The "why." |
| 16–29 | When to use agents; deterministic vs agentic workflows | **Focus.** The core judgment call. |
| 30–33 | LangChain's 6-step build framework | **Read.** |
| 34–59 | MySQL + BIRD database setup | **Skim/run.** Plumbing. |
| 60–70 | **Component 1: Persona** (dict → system prompt → test) | **Focus.** |
| 71–81 | **Component 2: LLM as Planner**; 4 knowledge sources | **Focus.** |
| 82–84 | **Component 3: Prompting Strategy** (CoT vs ReAct) | **Focus.** |
| 85–111 | **Component 4: Tools** (3 tools, schemas, function-calling demo) + MCP addendum | **Focus + slow down.** |
| 112–116 | **Component 5: Memory** (short vs long term) | **Read.** |
| 117–123 | **End-to-end** ReAct agent loop (`run_agent`) | **Focus.** The payoff. |

**Live-class notebook** (`L1-text2sql-liveclass.ipynb`): the same build, live-coded and terser (setup → persona → planner → 3 tools → agent loop, ending with `run_agent("Who stole my PS5?")` to test out-of-scope handling, and an improved `run_agent` with `max_steps` + trace). **Read it second**, as a fast recap of the main notebook.

---

## ✅ Walk-away checklist

After the notebooks, you should be able to say, in your own words:

- [ ] Why "the LLM decides the control flow" is the defining property of an agent.
- [ ] When to choose a deterministic workflow vs an agent (and the cost of agents).
- [ ] The five components and what each contributes.
- [ ] Why the **persona's boundaries** matter as much as its capabilities.
- [ ] How **function calling** works — and that the LLM never executes the tool itself.
- [ ] Why only certain LLMs can use tools.
- [ ] How the **ReAct agent loop** ties all five components into a working agent.

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. Complete the definition: "An AI agent is a system that uses an LLM to decide the ______ of an application."
2. You're automating a fixed, well-understood nightly report with the same steps every time. Workflow or agent — and why?
3. In the function-calling cycle, the LLM outputs `{"name": "execute_sql", "arguments": {...}}`. What happens next, and who runs the SQL?
4. The Text2SQL persona says the agent "must NEVER run DELETE, DROP, UPDATE, or INSERT." Which component is this, and why is stating it important?
5. During one `run_agent` call the LLM calls `list_tables`, then `get_table_schema`, then `execute_sql`, then returns text. Which prompting strategy is this, and what makes the loop finally stop?

<details>
<summary>Answers</summary>

1. **Control flow.** (The LLM decides the order of steps at runtime, rather than the developer hardcoding it.)
2. A **workflow** (deterministic). The task is well-defined with predictable steps, so a fixed code path is more predictable, faster, and cheaper. Agents are for *open-ended* tasks where the number/order of steps can't be predicted — using one here just adds latency, cost, and error surface.
3. Your **application** receives the JSON, **executes the real `execute_sql` Python function** itself, then sends the result back to the LLM for the next decision. The **LLM never runs the SQL** — it only *requests* the call.
4. **Persona (Component 1)**, delivered via the system prompt. Stating boundaries matters because what the agent *cannot* do (destructive SQL, guessing columns) is as important as what it can — it's the guardrail that keeps an autonomous, non-deterministic system safe.
5. **ReAct (Reasoning + Acting)** — the agent interleaves reasoning with tool calls. The loop stops when the LLM's reply contains **no `tool_calls`** (i.e., it returns plain text), which the orchestrator treats as the final answer.

</details>

[🔝 Back to top](#top)
