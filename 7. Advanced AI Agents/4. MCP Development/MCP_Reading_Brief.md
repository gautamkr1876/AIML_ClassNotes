<a id="top"></a>
# MCP Development — Reading Brief

> **Read this ONCE, end to end, before opening the notebooks.** Target time: ~22 minutes. By the time you reach the notebooks, every word in them will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`MCP_Jargon_Card.md`](./MCP_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebooks:** `MCP_Theory.ipynb` (concepts, ~66 cells) + `MCP_Server_Client_Tutorial.ipynb` (a from-scratch Text2SQL MCP build, ~104 cells). Slides in `../3.Agent: Frameworks and Protocols/MCP.pdf`.
> **Prereq:** function calling / tools from Module 7 Lecture 1 (MCP appeared there as a one-cell addendum; this is the full treatment).

---

## 🎯 30-second TL;DR

In Lecture 1 you gave an agent tools by hand-writing a JSON schema + a Python function for every capability. That doesn't scale: **M** models × **N** services = **M×N** bespoke connectors to build and maintain. **MCP (Model Context Protocol)** — Anthropic's open standard — fixes this by putting a *protocol* between models and services, collapsing **M×N → M+N**:

> Each service implements **one MCP Server**; each app implements **one MCP Client**; they all speak the same JSON-RPC language.

The theory notebook explains the architecture (**Host → Client → Server**, three **primitives**, two **transports**, and features like sampling and roots). The tutorial notebook makes it concrete by rebuilding the **Text2SQL agent** as an MCP Server + Client. The mental one-liner everyone uses: **MCP is USB-C for AI tools** — one standard plug instead of a custom cable per device.

---

## 🗺️ Agenda — what the notebooks teach, in order

**Theory notebook:**
1. The integration scaling problem (M×N) and the M+N fix.
2. Why it matters for production (extensibility, vendor neutrality, maintainability).
3. Architecture: **Host, Client, Server** + JSON-RPC.
4. The three **server primitives**: Tools, Resources, Prompts.
5. Server & Client code-flow walkthroughs for each primitive.
6. **Transports**: stdio (local) vs Streamable HTTP (remote, via SSE).
7. Message types: requests, results, notifications.
8. Core features: **sampling**, log/progress notifications, **roots**.

**Tutorial notebook:**
9. Build a SQLite DB (employees, customers, orders).
10. Write 3 plain tool functions (with SQL-injection + SELECT-only guards).
11. Wrap them in a `FastMCP` server (`@mcp.tool`), add a resource + a prompt.
12. Build the Client (spawns server over stdio, discovers tools).
13. The agent loop: OpenAI LLM ↔ MCP Client.

---

## 🧠 The big idea — a protocol, not a pile of adapters

Every integration you wrote by hand in Lecture 1 was a private handshake: *this* app knows how to talk to *this* tool, in a way nothing else can reuse. Add a second model and a third service and you're maintaining a tangle of one-off connectors — the count grows as **M×N**.

**The transferable analogy: USB-C.** Before USB-C, every device had its own charger; *P* devices × *Q* accessories meant a drawer full of incompatible cables. USB-C defined one standard port, so any device works with any accessory — the drawer collapses to a few universal cables. MCP does this for AI: define one protocol, and any MCP-compliant **Client** (Claude, an IDE, your app) can use any MCP **Server** (GitHub, Postgres, your database), regardless of vendor. You go from **M×N** custom connectors to **M+N** — each side implements the standard *once*.

That single shift buys the three things the theory notebook stresses: **extensibility** (add a tool by deploying a Server, zero changes to the model), **vendor neutrality** (a Server works with any compliant Client), and **maintainability** (one Server owns one interface contract, instead of M adapters drifting out of sync). The rest of MCP is just the concrete shape of that protocol.

---

## 📖 Core concept primers

Five primers cover the heart of MCP. Each has a **mental model**, plain-English meaning, notebook specifics, and why it matters.

### 1. The three-component architecture: Host → Client → Server

> **🪜 Mental model:** an operating system. The **Host** is the OS, each **Client** is a device driver, each **Server** is a peripheral. Apps talk to the OS; the OS mediates the hardware.

**Host** = the user-facing app (IDE, chat, agent platform) — the process container and the **security/trust boundary**. It spawns one **Client** per connection, each managing exactly one **Server**. The **Server** exposes a real system's capabilities. The rule that keeps it secure: **no Server ever talks to the LLM directly** — every capability is mediated through the Host, which controls what's connected and whether the user must consent before a tool runs. **Why it matters here:** it explains the separation of concerns in the tutorial — the server file knows the database, the client file knows the transport, the agent file knows the LLM, and none peeks into the others.

### 2. The three primitives: Tools, Resources, Prompts

> **🪜 Mental model:** *who holds the remote?* Tool = the **model** presses the button; Resource = the **app** presses it; Prompt = the **user** presses it.

A Server exposes capabilities as exactly three primitive types, distinguished by **who controls invocation**:

| Primitive | Controlled by | What it is | Example |
|---|---|---|---|
| **Tool** | the **model** (LLM) | executable action | run SQL, write a file, call an API |
| **Resource** | the **app** | read-only data (URI-addressed) | list documents for a file-picker |
| **Prompt** | the **user** | pre-written template | a slash command / chat-starter button |

**Why it matters here:** this is the most-tested MCP distinction. A **Tool** is invoked *reactively by the model during* a conversation; a **Resource** is fetched *proactively by the app*, often *before* the LLM is even involved; a **Prompt** fires on an explicit *user* action. The tutorial exposes all three: three tools, a `db://tables` resource, and a `text2sql_analyst` prompt.

### 3. Tools in MCP — schemas for free

> **🪜 Mental model:** you write a normal typed Python function; the SDK writes the LLM-facing paperwork.

In Lecture 1 you hand-wrote each tool's JSON schema. With MCP you just add `@mcp.tool()` above a typed function and the SDK **auto-generates the JSON Schema** from the signature (using `Field(description=...)` for each parameter). The Client and LLM only ever see that schema — never your Python. **Why it matters here:** it removes the most tedious, error-prone part of tool-building, and it enforces the Lecture-1 lesson that *the `description` is the interface* — the tutorial's `execute_sql` description literally says "ONLY SELECT" so the model doesn't waste turns generating a blocked `DELETE`. The tools still validate inputs (regex on table names, SELECT-only keyword blocking) because **tools are the boundary between a non-deterministic agent and a real system**.

### 4. Transports: stdio vs Streamable HTTP

> **🪜 Mental model:** stdio = talking to a program you launched in your own terminal; HTTP = calling a service across the internet.

MCP messages (JSON-RPC) travel over a **transport**. **stdio**: the Client spawns the Server as a **child process on the same machine** and they pipe JSON over stdin/stdout — no network, no ports, and *full bidirectional* messaging works out of the box. **Streamable HTTP**: for **remote** Servers at a URL; since HTTP can't natively push Server→Client, it uses **SSE** streams to fake it. **Why it matters here:** the tutorial uses **stdio** (`mcp.run(transport="stdio")`) — simplest, local, everything works. The theory notebook's warning is the takeaway: deploying the same app over HTTP with `stateless_http=true` or `json_response=true` **silently breaks** sampling, progress, and logging — features that worked in local testing. Know which transport disables what.

### 5. Sampling & roots — the "gotcha" features

> **🪜 Mental model:** *sampling* = the Server borrows the Client's phone to make a call (no phone of its own); *roots* = the Client hands the Server a key to specific rooms only.

**Sampling** inverts LLM access: a Server that needs an LLM (to summarize, classify) doesn't hold an API key — it calls `create_message()` to ask the *Client* (already authenticated) to run the model and return text. This removes the key/cost/abuse liability from public Servers. **Roots** let the Client declare which files/directories a Server may touch (`--roots /home/user/videos`). **Why it matters here:** both carry a beginner trap the notebook flags — **roots are not auto-enforced**; every filesystem tool must manually call `is_path_allowed()`, or the boundary is fiction. Sampling and roots are the features that separate "toy MCP server" from "safe production MCP server."

---

## 🔥 The M×N → M+N payoff — at a glance

| | Without MCP (direct) | With MCP |
|---|---|---|
| **Connectors** | M × N (one per model-service pair) | M + N (one per side) |
| **Add a new model** | +N connectors | +1 (implement the Client) |
| **Add a new service** | +M connectors | +1 (deploy a Server) |
| **Tool schema** | hand-authored per tool | auto-generated from function signature |
| **Vendor lock-in** | per-vendor tool conventions | any compliant Client ↔ any Server |
| **Maintenance** | M adapters drift per service change | 1 Server updates per change |

---

## 🗺️ Notebook reading map

**Theory** (`MCP_Theory.ipynb`):

| Cells | Teaches | How to read |
|---|---|---|
| 1–9 | M×N problem, M+N fix, why it matters | **Focus.** The "why." |
| 11–13 | GitHub example; Host/Client/Server architecture | **Focus.** |
| 15–33 | The 3 primitives (Tools/Resources/Prompts), server+client flow | **Focus + slow down.** Core. |
| 34–52 | Message types; stdio vs Streamable HTTP; the flags that break things | **Focus.** Note the "what breaks when" table. |
| 53–65 | Sampling, log/progress, roots | **Read.** Note roots aren't auto-enforced. |

**Tutorial** (`MCP_Server_Client_Tutorial.ipynb`):

| Cells | Teaches | How to read |
|---|---|---|
| 0–5 | What we're building; project layout | **Read.** The 3-file separation. |
| 6–18 | Build SQLite DB (employees/customers/orders) | **Skim.** Data setup. |
| 19–40 | 3 tool functions + guards (SQL-injection regex, SELECT-only) | **Focus.** The safety boundary. |
| 41–60 | `FastMCP` server: register tools, add resource + prompt, stdio entry point | **Focus.** How `@mcp.tool` replaces hand-written schemas. |
| 61+ | MCP Client + agent loop (LLM ↔ client) | **Focus.** Ties back to the Lecture-1 ReAct loop. |

---

## ✅ Walk-away checklist

After the notebooks, you should be able to say, in your own words:

- [ ] The M×N problem and how MCP makes it M+N.
- [ ] The roles of **Host**, **Client**, **Server** — and why the Server never touches the LLM.
- [ ] The three primitives and *who controls* each (model / app / user).
- [ ] Why `@mcp.tool()` means you no longer hand-write JSON schemas.
- [ ] stdio vs Streamable HTTP, and which HTTP flags silently break features.
- [ ] What **sampling** solves and how **roots** scope filesystem access (and that roots aren't auto-enforced).

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. An org has 4 AI apps and 6 backend services. How many connectors under direct integration vs under MCP?
2. A server capability lets the LLM *decide during a conversation* to run a SQL query. Which primitive is that — Tool, Resource, or Prompt?
3. Your MCP app works perfectly in local testing (stdio) but loses all progress/logging messages after you deploy it remotely. Name a likely cause.
4. With MCP, you add `@mcp.tool()` to a typed Python function. What does the SDK do that you had to do by hand in Lecture 1?
5. A public MCP Server needs to summarize data with an LLM but you don't want it holding an API key. Which MCP feature solves this, and how?

<details>
<summary>Answers</summary>

1. Direct: **4 × 6 = 24** connectors. MCP: **4 + 6 = 10** (each app implements one Client, each service one Server).
2. A **Tool** — tools are model-controlled and invoked reactively by the LLM during inference. (A Resource is app-controlled read-only data; a Prompt is a user-triggered template.)
3. You likely deployed over **Streamable HTTP with `stateless_http=true`** (or `json_response=true`), which disables Server→Client messages — breaking sampling, progress, and logging that worked under stdio's full bidirectional channel.
4. It **auto-generates the JSON Schema** (name, description, parameter types) from the function signature. In Lecture 1 you wrote that schema by hand for every tool.
5. **Sampling.** The Server calls `create_message()` to ask the **Client** (which is already authenticated to an LLM) to generate the text and return it — so the Server needs no API key, and carries no token cost or abuse liability.

</details>

[🔝 Back to top](#top)
