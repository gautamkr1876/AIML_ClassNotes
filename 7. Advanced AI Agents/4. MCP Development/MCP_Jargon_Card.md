<a id="top"></a>
# MCP Development — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebooks. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`MCP_Reading_Brief.md`](./MCP_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebooks:** `MCP_Theory.ipynb` (the concepts) + `MCP_Server_Client_Tutorial.ipynb` (a from-scratch Text2SQL MCP build) in this folder. Slides: `MCP.pdf` in `../3.Agent: Frameworks and Protocols/`.
> **Prereq:** function calling / tools from `../1. Agents: Foundations & Planning/` (MCP was previewed there as an addendum).

---

## C

**Client (MCP Client)** — A protocol-level component, created by a **Host**, that manages **one** connection to **one** Server. It discovers the server's capabilities (`list_tools`, `list_prompts`), forwards tool calls, and reads resources — but never interprets or modifies schemas; it's a pass-through pipe. One Host can run many Clients (one per Server).

**ClientSession** — The MCP SDK object that *is* the live connection to a Server. The tutorial wraps it in a class with async context management (`__aenter__`/`__aexit__`) so the connection is set up on entry and cleaned up on exit.

**`create_message()`** — The Server-side call used for **sampling**: the Server sends a message list and asks the Client to run its LLM and return the generated text. No API key needed on the Server.

## E

**Elicitation** — A Client feature that lets a Server **request specific information from the user** mid-interaction (e.g., "which file did you mean?"). One of several capabilities Clients can offer Servers.

## F

**FastMCP** — The high-level Python SDK class for building a Server in a few lines: `mcp = FastMCP("name")`, then register capabilities with `@mcp.tool()`, `@mcp.resource(...)`, `@mcp.prompt(...)` decorators. Decorator-based, Flask/FastAPI-like.

## H

**Host** — The top-level, user-facing AI application (an IDE with AI, a chat app, an agent platform). It's the **process container and trust boundary**: it spawns Clients, mediates between the LLM and Clients, enforces security/consent, and aggregates capabilities from multiple Servers. No Server ever touches the LLM directly — everything goes through the Host.

## J

**JSON-RPC 2.0** — The message format all MCP communication uses. Every exchange is a structured JSON **request/result** pair or a **notification**. You rarely write it by hand — the SDK handles it.

## M

**M×N problem** — The integration-scaling pain MCP solves: with **M** AI models/apps and **N** tools/services, direct integration needs **M×N** bespoke connectors (a connector per pair), which grows multiplicatively. MCP turns this into **M+N** (each model implements one MCP client, each service one MCP server). Same idea as the RAG-era N×M point, now formalized.

**MCP (Model Context Protocol)** — An open standard from Anthropic that puts a **protocol-level abstraction** between AI models and the external tools/data they use, so integrations are standardized, discoverable, and vendor-neutral. Think "USB-C for AI tools."

**MIME type** — A hint a **resource** declares (`application/json`, `text/plain`) telling the Client how to deserialize the response — e.g., `application/json` → `json.loads(resource.text)`.

## N

**Notification** — A **fire-and-forget** MCP message that expects no response (progress updates, log messages, tool-change events, the `initialized` handshake signal). Contrast with a request (which expects a result).

## P

**Primitive (server primitive)** — One of the three capability types a Server exposes, distinguished by **who controls invocation**: **Tools** (model-controlled), **Resources** (application-controlled), **Prompts** (user-controlled).

**Prompt (MCP primitive)** — A **user-controlled**, pre-written instruction template the Server author ships (triggered by slash commands, buttons, menu picks — e.g., Claude's chat-starter buttons). `@mcp.prompt(...)` returns a ready-to-send **messages array**. Lets the Server author, who knows the domain, do the prompt engineering once for every Client.

## R

**Request** — An MCP message that **expects a result** (`call_tool_request` → `call_tool_result`, `list_tools_request`, `initialize_request`). Half of a request/result pair.

**Resource (MCP primitive)** — **Application-controlled**, **read-only** data the app fetches proactively (not the LLM). Addressed by a URI (`docs://documents`), like a REST GET endpoint. Used to populate a UI file-picker, inject context into a prompt, provide autocomplete. Static URI (fixed) or templated (`documents/{doc_id}` with parameters). Contrast with a **Tool** (LLM-invoked, can act).

**Result** — The response that completes a **request** (`call_tool_result`, `list_tools_result`).

**Roots** — A **Client-declared** mechanism granting a Server access to specific files/directories (`--roots /home/user/videos`), defining the filesystem boundary the Server may operate within. ⚠️ The SDK does **not** auto-enforce this — every filesystem-touching tool must manually check `is_path_allowed(...)`.

## S

**Sampling** — A Server feature that **inverts LLM access**: instead of the Server holding its own API key, it *asks the Client* to run the LLM (via `create_message()`), and the Client (already authenticated) generates and returns the text. Removes the key/cost/rate-limit liability from public Servers.

**Server (MCP Server)** — A standalone process exposing a system's capabilities (a DB, an API, a filesystem) as MCP **Tools / Resources / Prompts**. This is where the real work happens; the Client never executes operations itself. Built with `FastMCP` + decorators.

**SSE (Server-Sent Events)** — Persistent one-way streaming connections used by the **Streamable HTTP** transport to let the Server push messages to the Client (which plain HTTP can't do). Two channels: a long-lived GET (server-initiated messages) and a short-lived POST-response (results/logs for one call).

**`stateless_http`** — A Streamable-HTTP flag (default `false`). Set `true` for horizontal scaling behind a load balancer — but it disables session IDs and **breaks** sampling, progress, and logging (Server→Client messages). A deployment footgun.

**stdio transport** — The simplest transport: the Client spawns the Server as a **child process on the same machine** and they exchange JSON over stdin/stdout. No network, no ports. Full bidirectional messaging works. Limitation: same-machine only.

**Streamable HTTP transport** — The transport for **remote** Servers (accessible at a URL over the network). Because HTTP is one-directional, it uses **SSE** to fake Server→Client pushes. Feature-rich by default, but flags (`stateless_http`, `json_response`) silently disable bidirectional features.

## T

**Text2SQL MCP (tutorial)** — The `MCP_Server_Client_Tutorial.ipynb` build: an MCP Server exposing three DB tools (`list_tables`, `get_table_schema`, `execute_sql`) over SQLite, an MCP Client that spawns it over stdio, and an agent loop wiring an OpenAI LLM to the client. The Lecture-1 agent, re-expressed through MCP.

**Tool (MCP primitive)** — **Model-controlled** executable functions the LLM decides to invoke (API calls, file writes, SQL). Defined with `@mcp.tool()` + typed params; the SDK **auto-generates the JSON Schema** from the signature (no hand-authoring). The `description` is the interface the LLM reads to decide when to call it.

**Transport** — The mechanism carrying JSON-RPC messages between Client and Server. Two options: **stdio** (local, child process) and **Streamable HTTP** (remote, over the network via SSE).

[🔝 Back to top](#top)
