<a id="top"></a>
# ReAct, Reflection & Memory — Reading Brief

> **Read once, end to end, before the notebook.** ~11 min.
>
> **Side reference:** [`ReAct_Reflection_Memory_Jargon_Card.md`](./ReAct_Reflection_Memory_Jargon_Card.md).
>
> **Notebook:** `Classroom_pynb_file_REACT___REFLECTION_and_MEMORY.ipynb` — 45 cells. One key needed (`GROQ_API_KEY` in Colab Secrets). Model throughout: `openai/gpt-oss-120b` on Groq, `temperature=0`.

---

## 🎯 30-second TL;DR

**Three patterns that turn a chatbot into an agent: ReAct lets it act, Reflection lets it self-correct, Memory lets it persist.**

**ReAct** is the only one fully built here and it's the star: three tools over a fake e-commerce database, and four queries of rising difficulty showing the agent decide — unprompted — which tools to call and in what order. The best moment is Query 4: asked to reorder everything from order 1004, the agent looks it up, discovers the order was **cancelled**, checks stock anyway, and reports both facts. Nobody wrote that branch.

**Reflection** is a real, runnable *generate → judge → revise* loop, and its most instructive outcome is an anticlimax: the judge replied `LGTM.` on the very first iteration, so no revision ever ran. **Memory** is taught as **pseudocode only** — three memory types (short-term, episodic, long-term) and the architecture that combines them, deliberately not executed.

---

## 🗺️ Agenda — what the notebook teaches, in order

| # | Cells | Topic | The one idea |
|--:|------|-------|--------------|
| 1 | 0–6 | Setup: Groq + `gpt-oss-120b`, `temperature=0` | deterministic traces |
| 2 | 7–11 | ReAct theory + comparison table | ReAct vs CoT vs Plan-and-Solve |
| 3 | 12–15 | Three tools over hardcoded dicts | the docstring *is* the tool spec |
| 4 | 16–18 | `create_react_agent` + trace printer | one line builds the agent |
| 5 | 21–28 | Four queries, rising difficulty | 1 tool → 3 tools → 4 tools → an edge case |
| 6 | 29–33 | Reflection theory + two architectures | same model vs stronger judge |
| 7 | 34–38 | Judge-Revise loop, run and verified | `LGTM` exits early |
| 8 | 40–44 | Memory: three types (pseudocode) | short-term / episodic / long-term |

---

## 🧠 The big idea — debugging a live system

The notebook's own comparison table gives the analogy that carries all three patterns, and it's worth memorising as a set:

- **Chain-of-Thought** = solving a maths problem on paper. You think, you write the answer. No tools, no feedback, no way to correct mid-reasoning.
- **Plan-and-Solve** = following a recipe. You write all the steps first, then execute them in order. You *can* re-plan, but it's expensive, so in practice the plan stays rigid.
- **ReAct** = **debugging a live production system.** You form a hypothesis, run one command, *look at what came back*, and let that reshape what you do next.

That third one is the whole lesson. The reason ReAct handles Query 4's cancelled order gracefully is not that it planned for cancellations — it's that it **observed** `Status: Cancelled` and reasoned onward from there. Adaptability comes from observing after every action, not from planning better.

Reflection adds a second loop *around* the answer (is this output good enough?), and memory adds a channel *across* runs (what did I learn last time?).

---

## 📖 Core concept primers

### 1. The ReAct loop — Thought, Action, Observation

> **🪜 Mental model:** a detective. Form a hunch, check one fact, let the fact change the hunch.

Three steps repeating until the agent can answer:

| Step | What happens | Example from the notebook |
|---|---|---|
| **Thought** | reason about what's known and needed | "The customer is asking about order 1001. Let me look it up." |
| **Action** | call a tool | `lookup_order("1001")` |
| **Observation** | receive the result | "Order 1001: Shipped, 2 items, arriving June 15" |

In LangGraph these map onto message types: an `AIMessage` carrying reasoning text and/or a `tool_calls` list, then a `ToolMessage` carrying the observation. The `run_agent` helper streams with `stream_mode="updates"` and prints each as it arrives, which is why you can watch the loop rather than just its output.

**Where it earns its keep here:** Query 3 ("which of my items are in stock, which are hard to replace, what's my total warranty?") requires *four* tool calls whose targets are only knowable after the first one returns — the agent can't know to look up "Noise Cancelling Headphones" until `lookup_order("1003")` tells it that's in the order.

### 2. Tools — the docstring is the interface

> **🪜 Mental model:** a job advert. The model only applies for tools whose advert describes its problem.

```python
@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its order ID. Returns order details including
    customer name, items ordered, order status, dates, and total amount.
    Use this when you need to find information about a specific order."""
```

The `@tool` decorator turns a plain function into something the LLM can call, and **the docstring becomes the description the model reads**. Note the structure of all three docstrings: *what it does* → *what it returns* → *when to use it*. That last sentence is the one beginners skip and the one that most affects tool selection.

Second design choice worth stealing: all three tools **return helpful errors**, not exceptions. `lookup_order("9999")` returns `"Error: No order found with ID '9999'. Valid order IDs are: ['1001', ...]"` — text the agent can read and recover from. An exception would just crash the loop.

Third: the "database" is three hardcoded Python dictionaries. Zero network calls, zero flakiness, fully reproducible in a classroom.

### 3. Reflection — generate, judge, revise

> **🪜 Mental model:** submitting code for review, then pushing a fix.

Three one-line functions, all wrapping the same model with different system messages:

```python
generate_code(spec)                      # "You are an expert Python developer…"
judge_code(code)                         # "You are a senior reviewer. Reply LGTM if perfect."
revise_code(spec, previous_code, feedback)
```

`reflect_and_revise` loops up to `max_iterations=3`: generate, judge, **exit early if `"LGTM"` appears in the critique**, otherwise revise and repeat. It returns a full history (every iteration's code and critique) plus `approved: True/False`.

The notebook distinguishes **two architectures**: (1) one model playing both roles via different system prompts — simple, but the model can be blind to its own systematic errors; (2) a separate, often stronger model as judge — more robust, since a different model brings a different perspective. This lab implements architecture 1.

**What actually happened:** the spec asked for `merge_sorted_lists` with edge cases and O(n+m) complexity. The first generation was a textbook two-pointer merge; the judge replied `LGTM.` immediately; the loop exited at iteration 1 with `approved: True`. Cell 38 then **extracts the code from the markdown fences with a regex, `exec`s it, and runs four real test cases** — which all pass. That verification step is the part most reflection demos skip.

### 4. Memory — three types, one prompt

> **🪜 Mental model:** what's on your desk (short-term), what you remember about this colleague (episodic), and the company wiki (long-term).

| Type | Stores | Duration | Implementation |
|---|---|---|---|
| **Short-term** | current conversation context | one session | the prompt / context window |
| **Episodic** | history across sessions | medium | `conversation_history`, last 10 turns |
| **Long-term** | accumulated knowledge | persistent | vector store (Chroma, FAISS, Pinecone) |

The `MemoryAugmentedAgent` pseudocode shows them combining in one `respond()` call: retrieve the top-3 relevant long-term memories by embedding similarity, take the last 10 episodic turns, assemble both into a labelled prompt, generate, then write back to *both* stores.

The subtlety worth carrying away: **retrieval turns long-term memory into short-term memory.** The vector store is huge; the context window is not; the retrieval step is what decides which slice of the past the model actually gets to see this turn.

---

## 🔥 The headline experiment — at a glance

The four ReAct queries, and what each proves:

| Query | Tool calls | What it demonstrates |
|---|---|---|
| 1 — status of order 1002? | 1 (`lookup_order`) | the simple case: one lookup, one answer |
| 2 — price + warranty for each item in 1002? | 3 (lookup, then 2 × product details) | **dependent** calls — targets unknown until call 1 returns |
| 3 — stock, replaceability, total warranty for 1003? | 4 | multi-tool reasoning across three data sources |
| 4 — reorder everything from 1004? | 3 | **edge case:** discovers `Status: Cancelled`, checks stock anyway, reports both |

Query 4 is the one to dwell on. The agent was never told what to do about cancelled orders. It found the fact, kept going, and surfaced it — which is exactly the "adjusts after every observation" row in the comparison table, happening live.

---

## 🧮 Shapes to memorise

**1. The ReAct loop**
```
Thought → Action → Observation → (repeat) → Answer
```
*In words:* reason about what you need, take one step to get it, read what came back, and let that decide the next step — until you can answer.

**2. Building a ReAct agent**
```python
agent = create_react_agent(model=llm, tools=[t1, t2, t3], prompt="You are a …")
```
*In words:* hand it a model, a list of `@tool`-decorated functions, and a system prompt; LangGraph builds the whole loop. (Deprecation note: in LangGraph v1.0 this moved to `from langchain.agents import create_agent` — the notebook's output shows the warning.)

**3. The reflection loop**
```
generate → judge → (LGTM? stop) → revise → judge → … up to max_iterations
```
*In words:* draft it, review it, stop if the reviewer approves, otherwise rewrite using the review — with a hard cap so a never-satisfied judge can't loop forever.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–6 | Agenda, install, Groq key, two identical LLM setups | **Skim.** Cells 3/4 and 5/6 are duplicates |
| 7–11 | ReAct theory, ASCII loop diagram, comparison table | **Focus on cell 11's table** — it's the most interview-quotable thing here |
| 12–15 | Three dicts + three `@tool` functions | **Focus on the docstrings**, skim the dictionary data |
| 16–18 | `create_react_agent` + the `run_agent` trace printer | **Read normally.** Note `stream_mode="updates"` |
| 19–20 | *(empty)* | Skip |
| 21–28 | Queries 1–4 | **Read every trace.** Count the tool calls; spend longest on Query 4 |
| 29–33 | Reflection theory, two architectures | **Focus.** Know when a separate judge model is worth it |
| 34–38 | Judge-Revise loop + `exec` verification | **Read normally.** Note it approved on iteration 1 |
| 39 | *(empty)* | Skip |
| 40–44 | Memory: table, retrieval pseudocode, combined agent | **Read as architecture**, not as code — it never runs |

---

## ⚠️ Gotchas

1. **Reflection never actually revised anything.** The judge said `LGTM.` at iteration 1, so the revise path was never exercised. You saw the loop's *structure*, not its payoff — if you want to see a revision, hand it a deliberately sloppier spec.
2. **`"LGTM" in critique` is brittle control flow.** A critique containing "this is not LGTM yet" would exit as approved. Real systems use structured output (a JSON verdict field), not substring matching.
3. **Memory is pseudocode.** Both cells say so explicitly. There is no working vector store, no embedding model, no persistence in this notebook.
4. **Cell 33's markdown says "We are using the raw Gemini SDK here."** It isn't — the code uses `ChatGroq` with `openai/gpt-oss-120b`. Leftover text from an earlier version; trust the code, not the prose.
5. **Trace step numbers look wrong.** Tool calls print as `[Step 0 - Tool Call #1]` because `step_count` only increments on reasoning text with content, and the model often emits a tool call with empty content. Cosmetic, but confusing on first read.
6. **`create_react_agent` is deprecated** in LangGraph v1.0 (moved to `langchain.agents.create_agent`). It still runs; you'll see the warning.
7. **`exec()` on model-generated code is unsafe** outside a throwaway sandbox. Fine for a Colab demo, never in production.
8. **Cells 3/4 and 5/6 are exact duplicates.** Not a bug, just re-run cells left in place.

---

## ✅ Walk-away checklist

- [ ] The three ReAct steps, and why observing after *every* action is what buys error recovery.
- [ ] How ReAct differs from Chain-of-Thought and Plan-and-Solve, with one analogy each.
- [ ] Why a tool's docstring is the most important line in a tool definition.
- [ ] Why tools should return error *strings* rather than raise exceptions.
- [ ] The reflection loop, and the trade-off between a same-model judge and a separate stronger judge.
- [ ] The three memory types, and why retrieval is what turns long-term memory into short-term memory.

---

## 🎯 5-question self-check

1. Query 2 asks for price and warranty of every item in order 1002, and the agent makes three tool calls. Why couldn't it have made all three simultaneously from the start?
2. You add a fourth tool, `check_shipping_status`, and the agent never calls it. What's the first thing to inspect?
3. The reflection loop returned `approved: True, total_iterations: 1`. What does that tell you about the *code*, and what does it not tell you about the *loop*?
4. Your reflection loop runs against a judge that never says LGTM. What stops it running forever, and what is `approved` at the end?
5. An agent has 10,000 stored long-term memories. Why can't it just put them all in the prompt, and what does the pseudocode do instead?

<details>
<summary><strong>Answers</strong></summary>

1. Because the later calls **depend on the earlier one's observation**. Until `lookup_order("1002")` returns, the agent doesn't know the order contains a USB-C Hub and a Mechanical Keyboard — so it can't know which products to look up. That dependency is exactly why ReAct interleaves acting with reasoning instead of planning everything up front. (Where calls *are* independent — as in Query 4's two `check_inventory` calls — the model does emit them together.)

2. **The docstring.** It becomes the tool's description, and it's the only text the model reads when deciding whether the tool fits. Check that it states what the tool does, what it returns, and explicitly *when to use it* — and that it doesn't overlap confusingly with an existing tool.

3. It tells you the generated `merge_sorted_lists` was good enough that the judge approved it first try (and cell 38's four `exec`'d test cases confirm it works). It tells you **nothing about whether the revise path works** — that branch never executed. A loop you've only seen take the happy path is an untested loop.

4. `max_iterations=3` caps it. After three judge rounds the function returns with `approved: False` and `final_code` set to the last revision — so the caller can tell the difference between "the judge signed off" and "we ran out of attempts", which is the information that actually matters downstream.

5. They wouldn't fit in the context window, and even if they did, irrelevant memories dilute the prompt and degrade the answer. The pseudocode instead embeds the incoming query, runs a similarity search over the vector store, takes `top_k=3`, and additionally discards anything scoring below `0.5` — so only a small, relevant slice of long-term memory is promoted into this turn's short-term memory.

</details>

[🔝 Back to top](#top)
