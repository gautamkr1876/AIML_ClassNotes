<a id="top"></a>
# MCP Theory — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`MCP_Theory_Reading_Brief.md`](./MCP_Theory_Reading_Brief.md) **first**. That file gives you the big picture and the storyline; this card is just the vocabulary. This is the *theory* companion to lecture 4's hands-on MCP build.

---

## A

**Abstraction layer** — A middle layer between two things so neither needs the other's internals. MCP is that layer between AI models and the tools/data they use: the model talks to the layer, the layer talks to the backend, and either side can change freely. This is the single idea that turns M×N into M+N.

**Adapter (connector)** — Custom "glue code" letting one specific model talk to one specific service (its own auth, data formatting, error handling). Without MCP you write one per (model, service) pair — exactly what explodes into M×N. MCP replaces them all with one shared protocol.

## B

**Backend** — The system doing the real work behind a server: a database, filesystem, external API like GitHub, or code sandbox. In MCP the **model never touches the backend directly** — the Server sits in front and translates safe, scoped requests, so backend internals can change without breaking any model.

**Bipartite graph** — Two groups of dots (M models, N services) with lines between them. "Fully connected" = every model wired to every service = M×N lines. MCP collapses this web into a tidy hub-and-spoke shape.

## C

**Capability** — Any single thing a Server can do or provide, advertised so a model can find it: a tool, a resource, or a prompt. Servers "expose capabilities"; clients "discover capabilities." The whole point of the protocol is making capabilities introspectable — discoverable at runtime rather than hard-coded.

**Capability discovery** — The runtime process where a Client asks a Server "what can you do?" and gets a machine-readable list of tools, resources, and prompts. Lets an agent pick tools *dynamically* mid-task instead of the developer wiring them in advance. In code: `tools/list`, `list_prompts`, etc.

**Capability handshake (negotiation)** — At connection time, Client and Server each declare which features they support (tool calls, resource reads, sampling, notifications). Both sides must respect those declarations for the whole session. Think of it as agreeing on the rules of the game before playing.

**Client (MCP Client)** — A protocol adapter living *inside* the Host, bound to exactly **one** Server (1:1). It sends JSON-RPC messages, relays discovery, and forwards tool calls — but never executes tools or holds app logic. **Don't confuse with Host** (the whole app) or **Server** (the capability provider). Five Servers → five Clients.

## E

**Elicitation** — A client feature letting a **Server pause and ask the user for missing information** mid-task, instead of failing or demanding everything up front. Example: a travel Server asks "window or aisle?" during booking. Flows Server → Client → user, keeping the user in control.

**Extensibility** — The property that you can add new abilities without touching existing code. In MCP, adding a tool = deploying a new Server; the Host and model don't change at all. This is one of the four reasons the notebook says M+N "matters for production."

## H

**Host (MCP Host)** — The top-level, user-facing AI app (an AI-powered IDE, chat agent, orchestration platform). The **process container and coordinator**: spawns Clients, talks to the model, enforces the security/trust boundary, and merges many Servers' capabilities into one menu. **Don't confuse with Client** — the Host is the whole app; a Client is one connection it owns.

**Hub-and-spoke** — Everything connects through a shared center, not to each other. MCP is the hub; models and services are spokes. This shape is *why* integrations grow as M+N (add one spoke) instead of M×N (wire every pair).

## J

**JSON-RPC 2.0** — A lightweight standard for "call a function on another program and get a result back," where every message is JSON (a text format of keys and values). MCP uses it as the wire format for all Client↔Server talk. It defines the three message shapes below: requests, results, notifications.

## L

**LLM (Large Language Model)** — The AI model (e.g., Claude) that reads text and generates text, and here also *decides which tools to call*. In MCP the LLM lives behind the Host and never reaches a Server directly — the Host mediates every request and response (the "Mediated Access Pattern").

**Logging notification** — A Server → Client "fire-and-forget" message carrying a debug/status line during tool execution (via `context.info(...)`). Purely for real-time feedback. It **breaks** if you deploy over HTTP with `stateless_http=true` or `json_response=true`.

## M

**Mediated Access Pattern** — The architectural rule that the LLM never reaches external systems directly; every request and response passes through the Host's control plane. This is the security backbone of MCP — it's where user consent and access control get enforced.

**MCP (Model Context Protocol)** — An open Anthropic standard defining a single, model- and service-agnostic way for AI apps to discover and use external tools and data. Headline win: it turns the M×N integration explosion into M+N. The whole notebook either explains that win or the machinery (Host/Client/Server, primitives, transports) delivering it.

**MIME type** — A short label (e.g., `application/json`, `text/plain`) attached to a resource that tells the Client how to read the returned bytes — parse as JSON vs. treat as raw text. It's a *hint*, not enforced; Clients rely on it to deserialize correctly.

**M×N problem** — The core pain MCP solves: with M models, N services, and no standard, you need M×N custom connectors, each separately built, tested, versioned, maintained. Growth is *multiplicative* — one new model adds N connectors, one new service adds M — making direct integration "fundamentally unscalable."

**M+N solution** — With MCP, each model implements the Client interface **once** and each service implements the Server interface **once**, so integrations drop from M×N to M+N — *additive* (linear), not multiplicative. The single most important number in the notebook.

## N

**Notification** — One of the three JSON-RPC message types: a **fire-and-forget** message that expects no reply (progress updates, logs, "initialized" signals, tool-list-changed events). Contrast with a **request** (expects a result). Notifications are the Server-to-Client messages that HTTP struggles to deliver.

## P

**Primitive** — One of the exactly **three** kinds of capability an MCP Server can expose: **Tools, Resources, Prompts**. The clean way to remember them is *who controls invocation*: Tools = model-controlled, Resources = application-controlled, Prompts = user-controlled. This trio is the heart of the "MCP Servers" section.

**Progress notification** — A Server → Client message reporting "step X of Y done" during a long tool run (via `context.report_progress(current, total)`). Keeps long operations from looking frozen. Like logging, it **breaks** under the two HTTP flags.

**Prompt (primitive)** — **User-controlled**: a pre-written, tested instruction template the *user* triggers explicitly (a slash command like `/format`, a button). The Server author bakes in the prompt engineering once; every Client gets it free. Returns a ready-to-send **messages array**. **Don't confuse with a Tool** (model decides) or **Resource** (app decides) — a Prompt is a human clicking a shortcut.

**Pydantic `Field()`** — A Python helper used inside tool definitions to attach a description and validation rules to each parameter. The SDK reads these to auto-generate the JSON tool schema, so the developer never hand-writes schemas.

## R

**Request** — A JSON-RPC message that **expects a result back** (e.g., `call_tool_request`, `initialize_request`, `list_tools_request`). Always half of a request/result pair. Contrast with a **notification** (no reply expected).

**Resource (primitive)** — **Application-controlled**, **read-only** data the *app code* fetches proactively — to populate a file picker, inject context, or show metadata. The model does **not** invoke resources; the Host does. Addressed by URI. **Don't confuse with a Tool**: a Tool performs a model-chosen action; a Resource hands data to the app around the model.

**Result** — A JSON-RPC message that is the **answer to a request** (e.g., `call_tool_result`). It completes the request/result pair. Flows Server → Client.

**Roots** — A **Client-declared** mechanism telling a Server which files/directories it may touch — a filesystem sandbox, usually passed as `--roots` at startup. Critically, the **SDK does not auto-enforce** roots; each tool must manually call `is_path_allowed()`, or the boundary is only advisory. Solves the "user won't type a full file path" problem by letting the model navigate granted directories.

## S

**Sampling** — A Server asking the **Client to run an LLM call on its behalf** (via `create_message()`), instead of holding its own API keys and billing. The Client already has an authenticated model connection, so it generates and returns the text. Keeps keys, cost, and permissions Client-side — essential for safe public Servers, and enables "agentic" Servers that need reasoning mid-tool.

**Sampling callback** — The Client-side function that fires when a Server sends a sampling request; it calls the Client's own LLM and returns the completion through the MCP channel. The Server side just calls `create_message()` and gets text back.

**Server (MCP Server)** — A standalone process exposing capabilities (tools, resources, prompts) to any MCP-compliant Client and translating them into safe backend operations. Single-responsibility (one domain per Server: GitHub, Postgres). **Don't confuse with Host or Client** — the Server is where the *work* happens; it never touches the LLM or other Servers.

**SSE (Server-Sent Events)** — A web technique that holds a connection open so a server can *push* a stream of messages to a client (impossible over plain HTTP, which is request-then-response). Streamable HTTP uses SSE to deliver the Server → Client messages MCP needs (notifications, sampling).

**Stateless (`stateless_http`)** — An HTTP-transport flag (default `false`); set `true` to allow horizontal scaling behind a load balancer. The trade-off: it drops session IDs and the push channel, so **sampling, progress, and logging all break**. A classic deploy-time surprise.

**Stdio transport** — The simplest transport: the Client launches the Server as a **child process on the same machine**; they exchange JSON over standard input/output — no network, no ports. Fully bidirectional, so every feature works. **Limitation:** same-machine only. **Contrast with Streamable HTTP** (remote, but feature-fragile).

**Streamable HTTP transport** — The transport for **remote** Servers reachable at a URL. Because HTTP is one-directional (client asks, server answers), it bolts on SSE to push Server → Client messages. Reaches further than stdio, but features silently vanish if you flip `stateless_http` or `json_response` to `true`. **Contrast with stdio** (local, always full-featured).

## T

**Tool (primitive)** — **Model-controlled**: an executable function the LLM inspects and *autonomously decides* to call mid-conversation (an API POST, file write, code execution). Defined with `@mcp.tool()` and typed params; the SDK auto-builds the JSON schema. **Don't confuse with a Resource** (read-only, app-triggered) or **Prompt** (template, user-triggered). The only primitive the model itself invokes.

**Transport** — The mechanism physically carrying JSON-RPC messages between Client and Server. Two exist: **stdio** (local child process) and **Streamable HTTP** (remote + SSE). Picking the wrong one, or wrong HTTP flags, is the notebook's headline "deployment pitfall."

**Trust boundary** — The line where security is enforced. In MCP it's the **Host**: it decides which Servers connect, what the model sees, and whether the user must consent before a tool runs. Everything crossing that line is mediated, never direct.

## U

**URI (Uniform Resource Identifier)** — The address string that identifies a resource, exactly like a web route. MCP resources come as **static URIs** (`docs://documents`, always the same endpoint) or **templated URIs** (`documents/{doc_id}`, with a placeholder that maps to a function argument). One templated resource can serve any item by ID.

## V

**Vendor neutrality** — Because MCP is an open standard, a tool exposed once works with *any* compliant client — Anthropic, OpenAI, Google DeepMind, or open-source. No lock-in to one model provider's tool-calling format. One of the four production benefits the notebook lists (with extensibility, maintainability, agent composition).

---

[🔝 Back to top](#top) · [→ Reading Brief](./MCP_Theory_Reading_Brief.md)
