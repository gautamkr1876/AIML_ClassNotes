<a id="top"></a>
# MCP Theory — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every term in it will already make sense — you'll be confirming what you know, not learning blind.
>
> **Side reference:** keep [`MCP_Theory_Jargon_Card.md`](./MCP_Theory_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `MCP_Theory (1).ipynb` in this folder. This is the *theory / slide-deck* companion to lecture 4's hands-on MCP build — it's almost all markdown, with small illustrative code snippets rather than a runnable project.

---

## 🎯 30-second TL;DR

**MCP (Model Context Protocol)** is an open standard from Anthropic that gives every AI app **one shared way** to plug into external tools and data — databases, GitHub, filesystems, Slack, anything.

The single number that captures its value:

- **Without MCP:** wiring M models to N services needs **M × N** custom connectors — grows *quadratically*.
- **With MCP:** each model implements the protocol once, each service once → **M + N** connectors — grows *linearly*.

The notebook's framing: MCP is a **protocol-level abstraction between AI models and the services they consume**, and the M×N → M+N shift "is not an incremental improvement — it is a change in the asymptotic scaling behavior of integration complexity." Everything else (Host/Client/Server, the 3 primitives, transports, sampling, roots) is the *machinery* that makes M+N real.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **The integration scaling problem** — why connecting AI models to tools gets unmanageable without a standard.
2. **The M×N problem, formally** — M models × N services = M×N bespoke connectors; quadratic maintenance pain.
3. **MCP's M+N solution** — a shared abstraction layer (built on JSON-RPC 2.0) turns the web into hub-and-spoke.
4. **Why it matters for production** — extensibility, vendor neutrality, maintainability, agent composition, long-term leverage.
5. **A concrete example (GitHub)** — MCP *shifts who does the work* from every app developer to one Server author.
6. **The three-component architecture** — **Host** (coordinator), **Client** (protocol bridge), **Server** (capability provider).
7. **The three Server primitives** — **Tools** (model-controlled), **Resources** (app-controlled), **Prompts** (user-controlled).
8. **Client features & code flow** — elicitation, roots, sampling; tool discovery and tool execution in code.
9. **JSON message types** — requests, results, notifications (the JSON-RPC taxonomy).
10. **Transports** — **stdio** (local child process) vs **Streamable HTTP** (remote, via SSE) and what breaks when.
11. **Core features** — **sampling** (Server asks Client to run the LLM), log/progress notifications, and **roots** (filesystem sandbox).

---

## 🧠 The big idea — MCP is "USB-C for AI tools"

Before USB-C, every device had its own plug — one cable for your phone, another for your camera, another for your laptop. Connecting *any* device to *any* accessory meant a drawer full of specific cables: the **M×N** mess. USB-C replaced them with **one standard port** — build the device once, build the accessory once, and everything plugs into everything. That's **M+N**.

**MCP is USB-C for AI tools.** The "devices" are AI apps (models/hosts); the "accessories" are tools and data sources (GitHub, a database, a filesystem). Instead of a custom adapter per (app, tool) pair, each app speaks MCP once (as a **Client**) and each tool once (as a **Server**) — and now any app can use any tool.

Keep this analogy the whole notebook. Every piece maps to it:
- The **Server** is the accessory's USB-C port (exposes what it can do); the **Client** is the device's port (speaks the standard); the **Host** is the device itself (decides what to do with what's plugged in).
- The **primitives** (Tools/Resources/Prompts) are the *kinds of thing* that flow across the cable; the **transport** (stdio vs HTTP) is the physical cable — same protocol, different wire.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook. Each has a **mental model**, plain-English *what*, tiny example, and *why it matters here*.

### 1. The M×N problem → M+N solution

> **🪜 Mental model:** a tangled cable drawer (every device needs its own cable for every accessory) vs. one universal USB-C port.

**What it is.** Let **M** = number of AI models (or host apps) and **N** = number of external services (tools, DBs, APIs). With no shared standard, every (model, service) pair needs its own hand-built **connector** — custom auth, formatting, error handling. That's **M × N**, a fully-connected web. MCP introduces a shared **abstraction layer** (a middle layer both sides talk to instead of each other), so each model implements the Client interface **once** and each service the Server interface **once**: **M + N**.

**Tiny example.** 5 models, 20 services. Direct integration: **5 × 20 = 100** connectors. With MCP: **5 + 20 = 25** implementations. Add a 6th model: direct approach needs **+20** new connectors; MCP needs **+1** (just its Client).

**Why it matters in this notebook.** The entire thesis. It's a change in *scaling behavior* (quadratic → linear), not a tweak: "schema drift in a single API can cascade into M separate breakage events" without MCP, versus updating **one** Server with MCP.

### 2. Host vs Client vs Server (the three-component architecture)

> **🪜 Mental model:** a restaurant — the **Host** seats you and runs the floor, the **Client** is one waiter assigned to one kitchen, the **Server** is a kitchen that actually cooks.

**What it is.** MCP splits every session into three roles with strict separation of concerns:
- **Host** — the user-facing app (an AI IDE, chat agent). The coordinator and **trust boundary**: spawns Clients, talks to the LLM, enforces consent/security, merges all Servers' capabilities into one menu.
- **Client** — a protocol adapter *inside* the Host, bound **1:1** to a single Server. Sends/receives JSON-RPC, relays discovery, forwards tool calls. Infrastructure, **not** app logic — never executes tools.
- **Server** — a standalone process exposing capabilities and translating them into safe backend operations (a DB query, file read). One domain per Server (GitHub, Postgres).

**Tiny example.** A Host needing GitHub + Postgres + a filesystem instantiates **three Clients** (one per Server). The LLM asks to "create an issue" → Host routes it to the GitHub Client → Client sends a JSON-RPC `tools/call` to the GitHub Server → Server hits the API and returns the result up the chain.

**Why it matters in this notebook.** This is the **Mediated Access Pattern**: the LLM *never* reaches a Server directly. That mediation is where security lives, and it's why "adding a capability requires only deploying a new Server, with no modifications to the Host or existing Servers."

### 3. The three Server primitives — Tools / Resources / Prompts

> **🪜 Mental model:** sort them by *who pulls the trigger* — model (Tools), app (Resources), user (Prompts).

**What it is.** A Server exposes capabilities through exactly **three** primitive types:
- **Tools** — **model-controlled** executable functions. The LLM reads the schemas and *autonomously decides* when to call one (write a file, POST to an API). Defined with `@mcp.tool()`; the SDK auto-generates the JSON schema from the typed signature.
- **Resources** — **application-controlled**, **read-only** data the *app code* fetches proactively (a document list for a file picker). Addressed by **URI**: static (`docs://documents`) or templated (`documents/{doc_id}`). The model does **not** invoke these.
- **Prompts** — **user-controlled** templates the user triggers explicitly (a slash command like `/format`, a button). The Server author bakes in the prompt engineering once; the Server returns a ready-to-send **messages array**.

**Tiny example.** A document Server: `read_doc_contents` (Tool — model reads a doc mid-chat) · `docs://documents` (Resource — app lists docs to build a menu *before* the chat) · `/format` (Prompt — user clicks to rewrite a doc in markdown).

**Why it matters in this notebook.** "Who controls invocation" is *the* exam question. Model decides → Tool. App fetches data around the model → Resource. Human clicks a shortcut → Prompt. Memorize that mapping.

### 4. Transports — stdio vs Streamable HTTP

> **🪜 Mental model:** stdio is an intercom between two people in the same room; HTTP is a phone line to someone across the country (and the callback feature only works on certain plans).

**What it is.** The **transport** is the physical channel carrying JSON-RPC messages.
- **stdio** — the Client launches the Server as a **child process on the same machine**; they exchange JSON over standard input/output. No network, no ports. **Fully bidirectional**, so *every* feature works.
- **Streamable HTTP** — for **remote** Servers at a URL. But plain HTTP is one-way (client asks, server answers), and MCP needs the Server to *push* back (notifications, sampling). So HTTP bolts on **SSE (Server-Sent Events)** — a held-open connection that lets the Server stream to the Client.

**Tiny example.** `FastMCP(...)` run locally = stdio, everything works. Deploy the same Server behind a load balancer with `stateless_http=true` = the push channel disappears, so **sampling, progress, and logging silently break** — but tool calls still work.

**Why it matters in this notebook.** The headline "deployment pitfall": an app perfect in local stdio dev can silently lose features over HTTP with the wrong flags. The rule: **develop against the same transport you deploy with.**

### 5. Sampling — the Server borrows the Client's LLM

> **🪜 Mental model:** the Server doesn't own a phone; it hands its note to the Client and says "you already have Claude on the line — could you ask for me?"

**What it is.** Some Servers need a bit of LLM reasoning inside their tool logic (summarize data, classify an input). **Sampling** lets the Server *request an LLM completion through the Client* rather than calling a model itself. The Server calls `create_message(messages=[...])`; the Client's registered **sampling callback** runs the Client's own authenticated model and returns the text through the MCP channel.

**Tiny example.** A travel Server has 20 flights and wants the best picked. Instead of holding its own API key, it sends the list via sampling; the Client's model returns "pick flight #7."

**Why it matters in this notebook.** A security-and-cost win: the Server never holds API keys or eats billing/rate-limit risk — critical for **public** Servers, where otherwise "anyone connecting could trigger LLM calls on the Server's account." (Sampling rides the push channel, so it too breaks under the HTTP flags.)

### 6. Roots — a Client-declared filesystem sandbox

> **🪜 Mental model:** a hotel keycard that only opens *your* floors — the Server can roam the granted directories but is locked out of everything else (if the code actually checks the card).

**What it is.** **Roots** are files/directories the *user grants a Server up front* (typically `--roots /path/a /path/b` at startup), defining the Server's filesystem boundary. It solves the "user won't type the full path to `bikin.mp4`" problem: with `list_roots` + `read_directory` tools, the LLM can *navigate* the granted trees. **But the SDK does not auto-enforce roots** — every filesystem tool must manually call `is_path_allowed()`, or the boundary is only advisory.

**Tiny example.** User: "Convert bikin.mp4 to gif." Server started with `--roots /home/user/videos`. The LLM calls `list_roots` → `read_directory(...)` → finds the file → `convert_video(path)`. Each tool first calls `is_path_allowed(path)`, raising "Access denied" if the path escapes the roots.

**Why it matters in this notebook.** The concrete security lesson: advertising a boundary ≠ enforcing it. A Server that *claims* roots as its limit can still be told to read files outside them unless you add the check.

---

## 🔥 The headline takeaways — at a glance

### A) The M×N → M+N win

| Scenario | Connectors needed | Add 1 model | Add 1 service | Growth |
|---|---|---|---|---|
| **Direct integration (no MCP)** | **M × N** | +N connectors | +M connectors | Quadratic |
| **With MCP** | **M + N** | +1 (its Client) | +1 (its Server) | Linear |

*Example: M=5, N=20 → 100 connectors without MCP vs 25 with MCP.*

### B) The three Server primitives (memorize this table)

| Primitive | Controlled by | What it does | Triggered when | Example |
|---|---|---|---|---|
| **Tools** | The **model** (LLM) | Executable actions | Model decides, mid-inference | `read_doc_contents`, API POST, file write |
| **Resources** | The **application** | Read-only data access | App fetches proactively | `docs://documents` list for a file picker |
| **Prompts** | The **user** | Pre-built instruction templates | User clicks / types a command | `/format` to rewrite a doc in markdown |

### C) What breaks under each transport

| Feature | stdio | HTTP (defaults) | HTTP `stateless_http=true` | HTTP `json_response=true` |
|---|---|---|---|---|
| Tool execution | ✅ | ✅ | ✅ | ✅ |
| Progress notifications | ✅ | ✅ (via SSE) | ❌ | ❌ |
| Logging messages | ✅ | ✅ (via SSE) | ❌ | ❌ |
| Sampling requests | ✅ | ✅ (via SSE) | ❌ | ❌ |

**Takeaway:** tool calls always survive; the *push-based* features (progress, logging, sampling) are the fragile ones.

---

## 🧮 "Formulas" to memorise (mostly conceptual)

MCP theory has **no math to derive** — but there's one counting rule and a handful of hard rules that behave like formulas. Memorize these.

### 1. The integration-count rule (the one real formula)

```
Without MCP:  total connectors = M × N     (quadratic)
With MCP:     total connectors = M + N     (linear)
```

**In words:** without a standard, the number of integrations equals the number of models **times** the number of services; with MCP it equals the number of models **plus** the number of services.
**Worked example:** M = 5 models, N = 20 services → without MCP `5 × 20 = 100`; with MCP `5 + 20 = 25`. Adding one more model changes `+N = +20` (without) vs `+1` (with).

### 2. Key rules (conceptual, no math)

- **1 Client ↔ 1 Server.** A Client binds to exactly one Server. Five Servers → five Clients.
- **Exactly 3 primitives.** Tools, Resources, Prompts — no more, no fewer. Sort by who invokes: model / app / user.
- **3 message types.** Request (expects a result) · Result (answers a request) · Notification (fire-and-forget, no reply).
- **The LLM never touches a Server directly.** Everything routes through the Host (Mediated Access Pattern = the trust boundary).
- **Push features need a push channel.** Progress, logging, and sampling require Server→Client messaging → they work on stdio and default HTTP, break under `stateless_http` or `json_response`.
- **Roots aren't auto-enforced.** You must call `is_path_allowed()` in every filesystem tool.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **0–3** | Colab badge, sources, intro: MCP as an "architectural primitive" | **Skim** — note the thesis: protocol layer between models and services. |
| **4–10** | M×N problem, formal statement, M+N solution, why it matters (extensibility / vendor neutrality / maintainability / agents) | **FOCUS — the thesis.** Nail the counting + four benefits. |
| **12** | GitHub example — "MCP shifts *who* does the work" table | **Read** — concrete grounding for M×N. |
| **14–16** | Host / Client / Server architecture + the 3-primitive overview | **FOCUS** — memorize roles and the who-controls-it mapping. |
| **18–33** | Code flows: Server init, Tools, Client init, discovery/execution, Resources, Prompts | **Read prose, skim code** — snippets only; the real build is lecture 4. |
| **35–46** | JSON message types (request/result/notification) + stdio transport | **Read** — the 3 message types + stdio's bidirectional strength. |
| **47–53** | Streamable HTTP, SSE, the two flags, the "what breaks" table, deployment pitfall | **FOCUS** — the transport table is a favorite exam target. |
| **54–66** | Core features: sampling, log/progress notifications, roots (+ `is_path_allowed`) | **Read carefully** — sampling and roots each carry a security lesson. |

**Total notebook read time:** ~50 min. Add this brief's ~22 min → ~72 min, vs. reading the dense notebook cold (~100+ min).

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why direct integration doesn't scale** — M models × N services = M×N bespoke connectors; quadratic maintenance.
- [ ] **What MCP changes** — a shared abstraction layer (JSON-RPC 2.0) drops it to M+N; add a model or service with one integration.
- [ ] **The three components and their roles** — Host (coordinator + trust boundary), Client (1:1 protocol bridge), Server (capability provider). And that the LLM never touches a Server directly.
- [ ] **The three primitives by who controls them** — Tools (model), Resources (app), Prompts (user) — and give an example of each.
- [ ] **stdio vs Streamable HTTP** — local child-process (always full-featured) vs remote over network+SSE (features break under `stateless_http` / `json_response`).
- [ ] **What sampling and roots are for** — sampling lets a Server borrow the Client's LLM (no keys on the Server); roots sandbox a Server's filesystem access (but you must enforce it manually).

If any feel shaky, return to the matching primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Your stack has **4 models** and **10 services**. How many integrations without MCP? With MCP? How many does adding one more service cost in each case?
2. A user clicks a `/summarize` button in a chat UI and a pre-written instruction gets sent to the model. Which of the three primitives is this — and who "controls" it?
3. You built and tested an MCP app locally with **stdio** and progress bars worked great. You deploy it over Streamable HTTP with `stateless_http=true`. What silently stops working, and why?
4. In one sentence: what problem does **sampling** solve, and why is it safer than the alternative for a public Server?
5. A Server was started with `--roots /home/user/videos`. A tool is asked to read `/etc/passwd`. Does MCP automatically block this? What must the developer have done?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **Without MCP: 4 × 10 = 40** connectors. **With MCP: 4 + 10 = 14.** Adding one service costs **+M = +4** without MCP (one per model) but **+1** with MCP (just that Server). Quadratic vs linear.

2. It's a **Prompt** — the third primitive, **user-controlled** (triggered by an explicit human action: slash command, button). Contrast: Tool = model-controlled, Resource = app-controlled. The Server author wrote it once; the user just clicks.

3. **Progress notifications, logging, and sampling all silently break.** These are Server→Client *push* messages needing the SSE channel. `stateless_http=true` disables session IDs and the GET SSE connection (to allow load-balanced scaling), so the Server can't push intermediate messages — the Client gets only the final result. **Tool execution still works.** This is the "deployment pitfall": develop against the transport you deploy with.

4. **Sampling lets a Server request an LLM completion *through the Client*** (via `create_message()`) instead of calling a model itself. Safer for a public Server: the Server never holds API keys, never eats billing/rate-limit risk, and strangers can't run LLM calls on its account — the Client, already authenticated and permission-controlled, does the generating.

5. **No — MCP does not auto-block it.** Roots are only advisory unless code checks them. The developer must have written an **`is_path_allowed()`** check in every filesystem tool (e.g. `os.path.commonpath([root, requested_path]) == root`) that raises "Access denied" for paths outside the granted roots. Without it, the Server can be told to read files outside its advertised boundary.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./MCP_Theory_Jargon_Card.md)
