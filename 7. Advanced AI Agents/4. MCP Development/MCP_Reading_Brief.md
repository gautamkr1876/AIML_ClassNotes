<a id="top"></a>
# MCP Development — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target: ~22 min. By the time you reach the notebook, every word will already make sense — you'll be confirming what you know, not learning blind.
>
> **Side reference:** keep [`MCP_Jargon_Card.md`](./MCP_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `MCP_Server_Client_Tutorial (1).ipynb` in this folder — a *teaching walkthrough* (not a runnable Colab) that builds a Text2SQL agent piece by piece.

---

## 🎯 30-second TL;DR

The notebook builds, from scratch, a system where **you ask a database a question in plain English and an LLM answers it by writing and running SQL for you** — without ever touching the database directly.

The three moving parts:

1. **MCP Server** — exposes three database tools (`list_tables`, `get_table_schema`, `execute_sql`) over the **Model Context Protocol (MCP)**.
2. **MCP Client** — spawns that server as a subprocess, talks to it over **stdio** pipes, and discovers its tools at runtime.
3. **Agent Loop** — connects an OpenAI LLM to the client, running a **ReAct** (Reason + Act) cycle.

**The headline trace:** for *"How many customers pay in EUR?"* the agent makes **4 LLM calls** and **3 tool calls** (discover → inspect → query), arriving at *"There are 4 customers who pay in EUR"* — for about **$0.002**.

The single biggest idea: **the server is LLM-agnostic**. Define tools *once*; any client (OpenAI, Claude, Gemini) can use them. That's the whole reason MCP exists.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **What we're building** — the three parts (server, client, agent) as two processes talking over pipes.
2. **The three tools** — `list_tables`, `get_table_schema`, `execute_sql`, called in order: discover → inspect → query.
3. **Sample database** — a SQLite file, three tables (`employees`, `customers`, `orders`, 28 rows), with a foreign key that forces JOINs.
4. **Tool functions as plain Python** — no MCP yet; testable functions with connection handling and safety checks.
5. **Safety inside tools** — regex validation vs SQL injection; keyword blocking so only `SELECT` runs; errors returned as strings for self-correction.
6. **Functions → MCP server** — one `FastMCP` instance, `@mcp.tool()`, and the JSON Schema it auto-generates.
7. **Bonus primitives** — resources (read-only endpoints) and a prompt template (server-owned persona).
8. **The stdio entry point** — `mcp.run(transport="stdio")`.
9. **Building the MCP client** — spawn subprocess, connect handshake, `async with` cleanup, tool wrappers.
10. **The agent loop** — system prompt, MCP→OpenAI schema conversion, and the ReAct loop.
11. **End-to-end trace** — one question through every layer, and the resulting `messages` list.

---

## 🧠 The big idea — MCP is a universal power strip for LLM tools

Before MCP, every time you wanted an LLM to *do* something (query a database, hit an API, read a file) you wrote custom glue — and rewrote it for each model. Like every appliance needing its own unique wall socket.

**Analogy — the universal power strip.** MCP is a standard power strip. The **server** is the strip: labelled sockets (tools), each with a spec sheet (the JSON Schema) saying what plugs in. The **client** is any device that knows the standard socket. Because the socket is standard, you plug in an OpenAI agent today and a Claude one tomorrow — **the strip never changes**. Build the database tools once; every model uses them.

Everything follows from this separation:
- The **server** knows the database, nothing about the LLM.
- The **client** knows the transport (stdio pipes), nothing about the database or LLM.
- The **agent** knows the LLM, nothing about how tools are implemented.

Each layer minds its own business, communicating only through contracts (JSON Schema for tools, JSON-RPC on the wire). That's what makes the system swappable, testable, and safe.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook — each with a **mental model**, a plain-English *what*, a tiny example, and *why it matters here*.

### 1. MCP (Model Context Protocol)

> **🪜 Mental model:** a *universal power strip* — expose tools once via a standard socket; any model can plug in.

**What it is.** MCP (Model Context Protocol) is an open standard for connecting LLMs to external tools, data, and reusable prompts through one uniform interface. "M-C-P" spells out its job: a **P**rotocol for giving a **M**odel its **C**ontext (which tools exist, how to call them). Instead of custom glue per model, you build an MCP *server* that advertises capabilities, and any MCP-aware *client* can consume them.

**Why it matters in this notebook.** The payoff line: *"The server is LLM-agnostic. Swap OpenAI for Claude or Gemini — the server doesn't change."* A single converter adapts the server's tool schemas to OpenAI's format; write another for Anthropic and the server is untouched. And the client hardcodes *nothing* — it learns the tools at runtime, so adding a fourth needs no client change.

### 2. Tools vs Resources vs Prompts (the three server primitives)

> **🪜 Mental model:** tools = buttons the *LLM* presses; resources = files the *client* reads; prompts = form letters the server hands out.

**What they are.** An MCP server can expose three kinds of things:
- **Tools** — actions the LLM chooses to call (can have side effects). The notebook's `list_tables`, `get_table_schema`, `execute_sql`.
- **Resources** — read-only data endpoints, like GET requests; the client reads them proactively to pre-load context. Bonus: `db://tables`, `db://schema/{table_name}`.
- **Prompts** — reusable templates the server defines and the client requests, so the *server* can own the persona. Bonus: `text2sql_analyst`.

**Why it matters in this notebook.** Tools are the main event — what the LLM invokes during the ReAct loop. Resources and prompts are shown as "bonus" to reveal MCP's full surface, but the exam-relevant distinction is **who triggers it**: the *LLM* decides to call a tool; the *client* decides to read a resource.

**Disambiguation table** (from the notebook):

| | Tools | Resources |
|--|-------|-----------|
| Who triggers | The LLM decides | The client reads proactively |
| Side effects | Can have them | Read-only by convention |
| Use case | Dynamic actions | Pre-loading context, caching |

### 3. Building a tool: plain function → `@mcp.tool()` decorator

> **🪜 Mental model:** the decorator is a *label maker* — you write a normal function; `@mcp.tool()` slaps on the machine-readable label the LLM reads.

**What it is.** In the `mcp` SDK, you write an ordinary Python function, then add `@mcp.tool(name=..., description=...)` above it. The **decorator** (a `@`-line that wraps a function to add behavior) does three things automatically: registers the function in the tool registry, **auto-generates a JSON Schema** from its type hints and `Field` descriptions, and sets up a handler so calls route to it.

**Why it matters in this notebook.** The notebook writes all three tools as plain, testable functions (Section 3), *then* wraps them with the decorator (Section 4) — proving the MCP layer is thin. The whole server is ~150 lines: "most of it is tool logic — the MCP wiring is just decorators and one `mcp.run()` call."

**Tiny concrete example.** For `get_table_schema`, the type hint `table_name: str = Field(description="The exact name of the table to inspect")` auto-generates a JSON Schema declaring a required string parameter `table_name` with that description. The LLM never sees your Python — only this schema. **The schema is the contract**, and the `description` is critical: it's what the LLM reads to decide *when* to use the tool.

### 4. The MCP Client: spawn, connect, discover, call, clean up

> **🪜 Mental model:** the client is a *phone operator* — it dials the server (spawns it), holds the line (session), asks who's available (discover), places calls (call_tool), and hangs up cleanly (cleanup).

**What it is.** The `MCPClient` class does five jobs: (1) **spawn** the server as a subprocess, (2) **connect** over stdio + do the MCP handshake, (3) **discover** tools, (4) **call** tools for the agent, (5) **clean up** by killing the subprocess. It's wrapped in `async with` so cleanup is automatic even on crash.

**Why it matters in this notebook.** This is where "two processes talking over pipes" becomes real. `connect()` has four steps: `StdioServerParameters` (how to spawn), `stdio_client()` (spawn + get read/write streams), `ClientSession` (protocol handler), `initialize()` (handshake). After that, the client can ask "what tools do you have?" — everything else is thin wrappers.

**Tiny concrete example — the whole usage:**

```python
async with MCPClient(command="python", args=["src/mcp_server.py"]) as client:
    tools = await client.list_tools()               # discovers 3 tools
    result = await client.call_tool("list_tables", {})
# subprocess killed automatically here
```

The `AsyncExitStack` tracks the subprocess, streams, and session, closing them **in reverse order** — session first, then subprocess. No leaks.

### 5. The ReAct agent loop

> **🪜 Mental model:** a *detective interrogation* — ask a question, follow leads (tools), gather clues (results), and keep going until you can state the conclusion.

**What it is.** **ReAct** = **Rea**son + **Act**. The LLM alternates between *thinking* about the next step and *acting* by calling a tool, using each result to inform the next thought. The code is a `for` loop (capped at `MAX_ITERATIONS = 15`): call the LLM with the tool schemas → if it returned a tool request, run it and append the result to `messages`, loop → if it returned plain text, that's the final answer, stop.

**Why it matters in this notebook.** The key insight: *"The agent doesn't know in advance how many tool calls it'll make. The LLM decides at each step — that's what makes this agentic instead of a fixed pipeline."* A simple count uses 3 tools; a JOIN inspects two schemas first. `tool_choice="auto"` + `temperature=0.0` = adaptive yet deterministic.

**Tiny concrete trace** ("How many customers pay in EUR?"):

```
Iter 1: LLM → list_tables({})          → ["customers","employees","orders"]
Iter 2: LLM → get_table_schema(customers) → [id,name,country,currency,balance]
Iter 3: LLM → execute_sql("SELECT COUNT(*) ... WHERE currency='EUR'") → 4
Iter 4: LLM → final text: "There are 4 customers who pay in EUR."
```

### 6. Layered safety (defense-in-depth)

> **🪜 Mental model:** *airport security* — the boarding pass check, the metal detector, and the gate scan each catch what the others miss.

**What it is.** **Defense-in-depth** means stacking safety checks so no single failure is catastrophic. Because tool inputs come from a non-deterministic LLM that can hallucinate or be manipulated, tools are the **trust boundary** between agent and real system — they validate everything.

**Why it matters in this notebook.** Three layers:
1. **System prompt** tells the LLM the rules ("only SELECT, never invent names").
2. **`get_table_schema`** validates the table name with the regex `^[A-Za-z_][A-Za-z0-9_]*$` — so `"employees; DROP TABLE employees"` is rejected before the DB is touched (SQL injection blocked).
3. **`execute_sql`** rejects queries containing `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/REPLACE`, and catches SQL errors, returning them as *strings* so the agent can **read the error and retry** ("column is `currency`, not `currency_code`").

---

## 🔥 The headline example — at a glance

The running example is *"How many customers pay in EUR?"* — the full accounting:

| Aspect | Value |
|---|---|
| **Question** | "How many customers pay in EUR?" (plain English) |
| **Generated SQL** | `SELECT COUNT(*) AS eur_count FROM customers WHERE currency = 'EUR'` |
| **Answer** | 4 customers |
| **LLM calls** | 4 (discover → inspect → query → answer) |
| **MCP tool calls** | 3 (`list_tables`, `get_table_schema`, `execute_sql`) |
| **Transport** | stdio pipes — no HTTP, no ports, no network |
| **Cost** | ~$0.002 total |
| **Model** | `gpt-4o-mini`, `temperature=0.0`, `tool_choice="auto"` |

**The sample database** (3 tables, 28 rows, one foreign key):

| Table | Rows | Key columns | Answers questions like |
|---|---|---|---|
| `employees` | 8 | department, salary, city | "average salary by department?" |
| `customers` | 10 | country, currency, balance | "how many pay in EUR?" (→ 4) |
| `orders` | 10 | customer_id (FK), product, amount | "orders from German customers?" (needs JOIN) |

**Headline takeaway** (from the Key Takeaways): *"MCP is just plumbing. You write functions, decorate them, and everything works. The server is LLM-agnostic — swap OpenAI for Claude and the server doesn't change."*

---

## 🧮 Key rules & message-flow (build-based, not formula-based)

This is a **build tutorial**, so there are no math formulas. Memorise these **flows and rules** — they're what interviewers probe.

### Rule 1 — The tool-call order the agent follows

```
list_tables → get_table_schema → execute_sql → (final text)
 discover   →     inspect       →   query     →   answer
```

**In words:** discover the tables first (never guess a name), inspect the relevant schema next, only then write and run SQL, finally summarise. The system prompt enforces this order so the LLM can't skip a step and hallucinate.

### Rule 2 — The client connect sequence (4 steps)

```
StdioServerParameters → stdio_client() → ClientSession → initialize()
  (how to spawn)         (spawn+streams)   (protocol)     (handshake)
```

**In words:** describe how to launch the server, spawn it and grab the pipe streams, wrap them in a protocol session, then handshake to agree on a version. Only after `initialize()` can you discover or call tools.

### Rule 3 — The ReAct loop condition

```
if LLM reply has tool_calls:  run them, append results, loop
else (plain text):            it's the final answer, break
```

**In words:** the presence or absence of a tool-call request is the *only* signal deciding whether to loop or stop. That single branch is the entire "agentic" mechanism.

### Rule 4 — What `messages` holds (short-term memory)

```
[ system(persona), user(question),
  assistant(tool_call), tool(result), ..., assistant(final text) ]
```

**In words:** persona, question, each tool request, each result, and the final answer are appended in order. This list is resent every LLM call and *is* the agent's memory — on a follow-up question the LLM sees it already knows the schema and skips redundant calls.

---

## 🗺️ Notebook reading map — where to spend your attention

The notebook is ~104 cells across 7 sections — a teaching walkthrough, so most cells are short markdown + a code snippet.

| Cells | What it teaches | How to read |
|---|---|---|
| **0–6** | The three parts, the three tools, project layout | **Skim** — ~4 min. Just absorb the architecture diagram and the discover→inspect→query idea. |
| **7–19** | Building the SQLite sample database (3 tables, foreign key) | **Skim** — ~4 min. You only need to know the tables and that `orders` FK forces JOINs. |
| **20–41** | The three tool functions as plain Python + their safety checks | **FOCUS** — ~10 min. The regex, the keyword block, and "return errors as strings" are prime interview material. |
| **42–62** | Wrapping functions as an MCP server; `@mcp.tool()`, JSON Schema, resources, prompts, stdio entry point | **FOCUS — this is the MCP core** — ~12 min. Understand the decorator → schema → contract chain. |
| **63–83** | The MCP client: spawn, connect (4 steps), `async with` cleanup, tool wrappers | **Read carefully** — ~10 min. The connect sequence and AsyncExitStack cleanup are the client's whole story. |
| **84–100** | The agent loop: system prompt, MCP→OpenAI schema conversion, the ReAct loop | **FOCUS — this is the payoff** — ~12 min. Trace the loop by hand once. |
| **101–104** | End-to-end data flow trace + `messages` snapshot + Key Takeaways | **Read carefully** — ~6 min. The 5 Key Takeaways are the exam summary. |

**Total target read time for the notebook:** ~55 min. Add this brief's ~22 min for **~75 min** of real *understanding*, versus reading the raw notebook cold.

---

## ✅ Walk-away checklist

After reading the notebook, you should be able to say in your own words:

- [ ] **What MCP is and why it exists** — a standard interface so tools are defined once and any LLM can use them; the server is LLM-agnostic.
- [ ] **The three server primitives and who triggers each** — tools (LLM), resources (client), prompts (server-owned templates).
- [ ] **How a plain function becomes a tool** — the `@mcp.tool()` decorator auto-generates a JSON Schema; the schema, not the code, is what the LLM sees.
- [ ] **What the client does end to end** — spawn subprocess → stdio handshake → discover → call → auto-cleanup via AsyncExitStack.
- [ ] **The ReAct loop in one sentence** — call LLM; if it asks for a tool, run it and feed the result back; if it gives text, that's the answer.
- [ ] **The three safety layers** — system-prompt rules, regex table-name validation, SELECT-only keyword blocking; plus why returning errors as strings enables self-correction.
- [ ] **The EUR trace** — 4 LLM calls, 3 tool calls, discover→inspect→query→answer, ~$0.002.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. What does "the server is LLM-agnostic" mean, and which single piece of code has to change when you switch from OpenAI to Claude?
2. In MCP, what's the difference between a **tool** and a **resource** — specifically, *who* triggers each?
3. The `execute_sql` tool catches SQL errors and returns them as a *string* instead of crashing. Why is that a deliberate design choice, not laziness?
4. Trace the tool calls the agent makes for *"How many customers pay in EUR?"* — how many LLM calls and how many tool calls, and in what order?
5. What is the `@mcp.tool()` decorator's job, and what does the LLM actually see when deciding whether to call a tool?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **The MCP server exposes tools through a standard protocol, with zero knowledge of which LLM is calling them.** Switching providers changes only the **agent's schema-conversion function** (`_mcp_tools_to_openai_schema`) and the LLM SDK call — you'd write an Anthropic version instead. The server (`mcp_server.py`) never changes.
2. **A tool is triggered by the LLM** (it decides to call it; can have side effects). **A resource is triggered by the client** (it proactively *reads* it, like a GET request, to pre-load context; read-only by convention). Mnemonic: tools = buttons the LLM presses; resources = files the client reads.
3. **Because a failed query is information, not a dead end.** In the ReAct loop the agent *reads* the error string (e.g., "no such column: currency_code"), reasons "the column must be `currency`", and retries. If the tool crashed instead, the loop would break and self-correction would be impossible — a core piece of agentic behavior.
4. **4 LLM calls, 3 tool calls.** (1) LLM → `list_tables({})` → three tables. (2) LLM → `get_table_schema({"table_name":"customers"})` → columns (finds `currency`). (3) LLM → `execute_sql("SELECT COUNT(*) ... WHERE currency='EUR'")` → 4. (4) LLM returns final text "There are 4 customers who pay in EUR." Order: discover → inspect → query → answer.
5. **It turns a plain function into a registered MCP tool:** adds it to the registry, **auto-generates a JSON Schema** from its type hints and `Field(description=...)`, and wires up the call handler. The LLM never sees your Python — only the generated schema (name, description, inputSchema). The `description` is what it reads to decide *when* the tool applies, so clarity is critical.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./MCP_Jargon_Card.md)
