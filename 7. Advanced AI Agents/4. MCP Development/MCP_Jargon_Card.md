<a id="top"></a>
# MCP Development Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`MCP_Reading_Brief.md`](./MCP_Reading_Brief.md) FIRST. That file gives you the punchline and the mental model; this card is just the vocabulary.
>
> **The notebook:** `MCP_Server_Client_Tutorial (1).ipynb` — a hands-on build of a Text2SQL agent that turns plain-English questions into SQL, using the Model Context Protocol.

---

## A

**Agent** — A program that uses a Large Language Model (LLM) to *decide what to do*, then does it by calling tools. In this notebook the agent takes a plain-English question, lets the LLM pick which database tools to run, runs them, and returns a human-readable answer. It's the glue between the LLM's reasoning and the MCP client's tool access.

**Agent Loop** — The repeating cycle at the heart of the agent: call the LLM → check if it wants a tool → run the tool → feed the result back → repeat until the LLM gives a final text answer. The notebook's loop lives in `run_agent_loop()` and caps out at `MAX_ITERATIONS = 15` so it can never spin forever.

**AsyncExitStack** — A Python helper (from `contextlib`) that tracks every async resource you open — the server subprocess, the transport streams, the session — and closes them all in reverse order when done. The client uses it so that even if code crashes mid-conversation, the server subprocess still gets killed and nothing leaks.

**async / await / async with** — Python keywords for *asynchronous* code — code that can pause while waiting (e.g., for a subprocess to reply) and let other work happen meanwhile. MCP's client talks to the server over pipes (lots of waiting), so nearly every client method is `async` and called with `await`. `async with MCPClient(...) as client:` is the async version of a `with` block — spawns the server on entry, shuts it down on exit.

## C

**call_tool** — The client method that invokes a named tool on the server (e.g., `call_tool("list_tables", {})`). It sends the request over stdio, the server runs the matching function, and the result comes back. In the agent loop, every time the LLM asks for a tool, the code forwards it through `call_tool`.

**ClientSession** — The MCP object that speaks the protocol on the client side. Created over the read/write streams, it handles the handshake, tool discovery, and tool calls. It's the client's "phone line" to the server — you call `session.list_tools()` or `session.call_tool()` and it handles the JSON-RPC wire format for you.

**Context (LLM context)** — Everything you send the LLM in one call: the system prompt, the conversation history, and the tool schemas. The point of MCP is to feed the model the *right* context (which tools exist, what their schemas are) so it can reason correctly — "M-C-P" is **Model Context Protocol**, a standard way to give a model its context.

## D

**Defense-in-depth** — A security mindset: don't rely on one safety check, stack several so a failure in one is caught by another. The notebook uses it three ways — system-prompt rules, `execute_sql` keyword blocking, and `get_table_schema` regex validation. No single layer is trusted to be perfect.

**Decorator** — A Python feature (the `@something` line above a function) that wraps a function to add behavior without changing its body. Here `@mcp.tool()` registers a plain function as an MCP tool, auto-generates its JSON Schema, and wires up the call handler — all from one line.

## E

**execute_sql** — The third and most powerful tool: it runs an actual SQL `SELECT` and returns the rows as a formatted text table. It has two safety layers: it rejects any query containing write keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.), and it catches SQL errors and returns them as a *string* instead of crashing — so the agent can read the error and retry.

## F

**FastMCP** — The high-level server class from the `mcp` Python SDK. If you've used Flask or FastAPI it feels familiar: create one server object, then decorate functions with `@mcp.tool()`, `@mcp.resource()`, or `@mcp.prompt()`. `mcp = FastMCP("Text2SQL-MCP")` creates the whole server in one line; everything else is decorated functions.

**Foreign key** — A column in one table that points to the primary key of another, linking the two. The `orders` table has a foreign key `customer_id` referencing `customers.id`. This is *why* the agent sometimes writes a JOIN — "show orders from German customers" must connect the two tables through that key.

**Function calling (tool calling)** — The LLM feature where the model, instead of answering in prose, returns a structured request to run a named function with specific arguments (e.g., "call `execute_sql` with this query"). Your code runs it and feeds the result back. It's what lets an LLM *act*, not just talk — the foundation the whole MCP + agent stack sits on.

## G

**get_table_schema** — The second tool: given a table name, it returns each column's name, type, whether it's nullable, and whether it's a primary key. It uses SQLite's `PRAGMA table_info()`. Crucially it validates the table name with a regex first, because the name comes from the (untrusted, non-deterministic) LLM.

## H

**Handshake (MCP initialize)** — The opening exchange where client and server agree on a protocol version before any real work — the `await self._session.initialize()` call inside `connect()`. Until it finishes, the client can't ask "what tools do you have?"

**Hallucination** — When an LLM confidently makes something up — here, a table or column name that doesn't exist. The notebook fights this: `list_tables` and `get_table_schema` exist *so the LLM doesn't have to guess*, and the system prompt orders it to "never invent table or column names."

## I

**inputSchema (JSON Schema)** — The machine-readable description of a tool's parameters — arguments, types, which are required. MCP auto-generates it from your function's type hints and `Field(description=...)`, and the LLM reads it to know *how* to call the tool. For `list_tables` it's empty (no args); for `get_table_schema` it declares a required string `table_name`.

## J

**JSON-RPC** — The message format MCP uses over the wire: a lightweight standard for "call this remote function with these arguments, here's the result." You never write it by hand — `ClientSession` and `FastMCP` handle it. It's the language the two processes speak through the stdio pipe.

## L

**LLM (Large Language Model)** — The AI text model (here OpenAI's `gpt-4o-mini`) that does the reasoning: reads the question, decides which tool to call next, writes the SQL, composes the answer. It's the *brain* but has no direct database access — it can only *ask* the MCP server to act, through tools.

**list_tables** — The first tool the agent always calls: returns every table name by querying SQLite's internal `sqlite_master` table. It's the "discovery" step — without it the agent would guess table names (see **Hallucination**). Returns `['customers', 'employees', 'orders']`.

## M

**MAX_ITERATIONS** — A hard cap (15 in the notebook) on how many times the inner loop can call the LLM for a single question. A safety valve: if the LLM gets stuck in a loop of failing tool calls, this stops it running forever (and burning money).

**MCP (Model Context Protocol)** — An open standard for connecting LLMs to external tools, data, and prompts through a uniform interface. Instead of custom glue for every model + tool, you expose tools *once* via an MCP server, and any MCP-aware client (OpenAI, Claude, Gemini…) can use them. The notebook builds one MCP server and one MCP client from scratch.

**MCP Client** — The half that spawns the server, connects over a transport, discovers its tools, calls them for the agent, and cleans up. In the notebook it's the `MCPClient` class. It knows *nothing* about the LLM or the database — its only job is MCP transport.

**MCP Server** — The half that *exposes* capabilities: advertises tools (with schemas), handles incoming calls, talks over a transport. Here it's `mcp_server.py`, built with `FastMCP`. It knows the database but nothing about the LLM — that separation is the whole point.

**messages (message history)** — The running conversation list sent to the LLM every call. Each entry has a `role` — `system` (persona), `user` (question), `assistant` (reply or tool request), `tool` (a result). It's the agent's **short-term memory**; on a follow-up question the LLM sees it already knows the schema and skips redundant tool calls.

## O

**OpenAI SDK** — The Python library (`from openai import OpenAI`) used to talk to OpenAI's chat models. The agent calls `openai_client.chat.completions.create(...)` with the messages and tool schemas; the model replies with text or a tool-call request. MCP is model-agnostic — swap this for Anthropic's or Google's SDK and the *server* code wouldn't change.

## P

**Persona (system prompt)** — The instructions given as the first `system` message, defining who the LLM is and how to behave. The notebook's persona is "You are a Senior SQL Developer and Data Analyst" plus a 4-step workflow and hard rules (only SELECT, never invent names). It's called the single most impactful piece of prompt engineering in the system.

**Prompt (MCP prompt)** — In MCP, a *prompt* is a reusable template the server defines and the client can request — letting the server, not the client, own the persona. The notebook adds `text2sql_analyst_prompt` as a bonus. Distinguish from the everyday sense of "prompt" (text you send an LLM).

**Pydantic Field** — `Field(description=...)` attaches a human-readable description to a function parameter. MCP folds it into the tool's JSON Schema so the LLM sees what each argument means. Used so `table_name` and `query` come with clear guidance.

## R

**ReAct pattern** — Short for **Reason + Act**: the LLM alternates between *thinking* about what to do and *acting* by calling a tool, using each result to inform the next thought. The notebook's loop is a ReAct loop — the LLM isn't following a fixed pipeline; it decides at each step whether to discover, inspect, query, or answer. That adaptivity is what makes it "agentic."

**Resource (MCP resource)** — A read-only data endpoint the server exposes — like a GET request in a web API. The client *proactively reads* a resource to pre-load context, versus a tool the *LLM* chooses to call. The notebook exposes `db://tables` and `db://schema/{table_name}` as bonus resources. Twin: **tool vs resource** — the LLM triggers tools; the client reads resources; tools can act, resources are read-only by convention.

## S

**Serialize / Deserialize** — Serialize = turn an in-memory object into text/bytes for transport; deserialize = turn it back. When the LLM sends tool arguments as a JSON string like `{"table_name": "customers"}`, MCP deserializes it into a Python dict before calling your function; results are serialized to travel back over the stdio pipe — all handled for you, so you never touch JSON by hand.

**SQL injection** — An attack where malicious text is smuggled into a query to run unintended commands (e.g., a table name of `employees; DROP TABLE employees`). The notebook blocks it with a regex allowing only valid identifier characters, and by rejecting write keywords in `execute_sql`. Tools are the trust boundary between the non-deterministic LLM and the database, so they validate everything.

**SQLite** — A tiny, file-based database built into Python (no server to install). The notebook uses it for `data/sample.db` — three tables (`employees`, `customers`, `orders`), 28 rows. Chosen because it's zero-setup and perfect for a teaching demo.

**stdio (standard input/output)** — The transport used here: the server reads JSON-RPC from *stdin* and writes replies to *stdout*; the client pipes data through both. No HTTP, no ports, no network — two processes talking through pipes. The client *spawns* the server as a subprocess and communicates over these pipes.

**System prompt** — See **Persona** — the first `system`-role message that sets the LLM's identity, workflow, and rules.

## T

**temperature** — A dial (0.0 to ~2.0) controlling LLM randomness: lower = more deterministic. The agent uses `temperature=0.0` because SQL generation should be consistent and correct, not creative.

**Text2SQL** — Turning a plain-English question into a runnable SQL query. The whole agent is a Text2SQL system: "How many customers pay in EUR?" → `SELECT COUNT(*) FROM customers WHERE currency = 'EUR'` → "There are 4 customers who pay in EUR."

**Tool (MCP tool)** — A function the server exposes that the LLM can choose to call, with a name, description, and parameter schema. The notebook exposes three: `list_tables`, `get_table_schema`, `execute_sql`. Twin: **tool vs resource** — a *tool* is an action the LLM invokes (can have side effects); a *resource* is read-only data the client pulls.

**Tool discovery** — The client asking the server "what tools do you have?" (via `list_tools`) and getting back names + schemas without knowing them in advance. MCP's superpower: the client doesn't hardcode tools, it learns them at runtime, so adding a server tool needs no client change.

**tool_choice="auto"** — An OpenAI setting telling the model it *may* call a tool or answer directly — its choice. "auto" is what makes the agent adaptive: sometimes it calls three tools, sometimes zero. The alternative forces a specific tool every time.

**Transport** — The channel over which client and server exchange MCP messages. This notebook uses **stdio**; other MCP transports include HTTP/SSE (Server-Sent Events) for networked servers. The server picks it with `mcp.run(transport="stdio")`. A `try/finally` block in each tool (closing the DB connection even on error) is the tool-level cousin of this cleanup discipline.

---

[🔝 Back to top](#top) · [→ Reading Brief](./MCP_Reading_Brief.md)
