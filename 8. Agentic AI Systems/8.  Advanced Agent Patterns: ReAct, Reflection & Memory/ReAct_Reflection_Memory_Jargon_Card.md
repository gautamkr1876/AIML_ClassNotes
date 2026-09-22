<a id="top"></a>
# ReAct, Reflection & Memory — Jargon Card

> **How to use.** Skim once (~5 min) before the notebook, then keep open as a side reference while reading.
>
> **Companion:** read [`ReAct_Reflection_Memory_Reading_Brief.md`](./ReAct_Reflection_Memory_Reading_Brief.md) first.
>
> **Notebook:** `Classroom_pynb_file_REACT___REFLECTION_and_MEMORY.ipynb` — 45 cells covering three agent patterns. Needs a Groq API key (stored in Colab Secrets as `GROQ_API_KEY`).

---

## A

**Action** — The middle step of the ReAct loop: the agent calls a tool or takes an external step, e.g. `lookup_order("1001")`. In LangGraph this surfaces as a `tool_calls` entry on an `AIMessage`.

**Agent** — Here specifically a **ReAct agent**: an LLM in a loop that decides whether to call a tool, reads what came back, and repeats until it can answer. Built in one line with `create_react_agent(model=llm, tools=tools, prompt=...)`.

**`agent.stream(..., stream_mode="updates")`** — Runs the agent and yields each intermediate step as it happens instead of only the final answer. This is how the notebook's `run_agent` helper prints the full visible trace.

**AIMessage / ToolMessage** — The two message classes the trace distinguishes. An `AIMessage` is the model speaking (reasoning text, and/or a `tool_calls` list); a `ToolMessage` is a tool's result coming back — the **Observation**.

## C

**Chain-of-Thought (CoT)** — "Think step by step, then answer." No tools, no external data, no mid-course correction. Good for maths and closed-book logic; helpless when the answer lives in a database.

**Context window** — The maximum amount of text a model can consider at once. It is the physical limit on short-term memory: everything the agent "knows right now" has to fit inside it.

**Cosine similarity** — A score for how close two vectors point in the same direction. Used to rank stored memories against a query embedding. The pseudocode keeps only results scoring above `0.5`.

**Critic / Judge** — The reviewing role in reflection. In this notebook it's the *same* model under a different system message: `"You are a senior reviewer. Reply LGTM if perfect."`

## E

**Embedding** — Text converted into a list of numbers such that similar meanings land near each other. The bridge that makes long-term memory searchable: store the embedding, later search by embedding.

**Episodic memory** — Memory of *past conversations*, across sessions. Analogy from the notebook: your memory of previous chats with a colleague. Implemented in the pseudocode as `self.conversation_history`, sliced to the last 10 turns.

## G

**Generator** — The producing role in reflection: writes the first draft, then the revision. Same model, different system message (`"You are an expert Python developer. Return only Python code."`).

**Groq** — An inference provider known for very fast token generation. The notebook uses `ChatGroq(model="openai/gpt-oss-120b", temperature=0)` — an open-weights model served by Groq, *not* an OpenAI API call.

## L

**LGTM ("Looks Good To Me")** — The approval token. The reflection loop checks `if "LGTM" in critique.strip().upper()` and exits early when the judge approves. A string check as a control-flow signal is fragile — see the Brief's gotchas.

**Long-term memory** — Accumulated knowledge in a vector store, persisting indefinitely. Analogy: a company knowledge base.

## M

**`max_iterations`** — The hard cap on reflection cycles (`3` here). Without it, a judge that never says LGTM produces an infinite, billable loop.

**Memory (three types)** — The notebook's taxonomy: **short-term** = the current prompt/context window, one session, your working memory while solving a problem; **episodic** = conversation history across sessions; **long-term** = a persistent vector store.

## O

**Observation** — The third ReAct step: the result of the action, fed back into the agent as its next input. In the trace: `[Observation from lookup_order] Order 1002: … Status: Shipped …`.

## P

**Plan-and-Solve** — Plan every step first, then execute the plan in order. More adaptable than CoT, less adaptable than ReAct: re-planning is possible but costly, so the plan stays rigid once made. The notebook's analogy: following a recipe.

**Pseudocode** — Illustrative code that is **not meant to be executed**. Both memory cells (`AgentMemory`, `MemoryAugmentedAgent`) are explicitly labelled this way — they show architecture, not a runnable system.

## R

**ReAct (Reason + Act)** — Interleaving reasoning with tool use: **Thought → Action → Observation**, repeated until the agent can answer. Its strength over the alternatives is **error recovery**: because it observes after every action, it adapts to whatever actually came back. The notebook's analogy: debugging a live production system.

**Reflection** — An agent evaluating its own output, identifying weaknesses, and producing an improved version — a built-in quality-assurance loop. Two architectures: **(1) one model, two roles** (same LLM, generator and judge system messages — used here; simpler, but the model may be blind to its own systematic errors) and **(2) two different models**, often a stronger judge over a cheaper generator (more robust, more expensive).

**Relevance scoring** — Ranking retrieved memories by similarity and discarding anything below a threshold, so irrelevant history doesn't pollute the prompt.

## S

**Short-term memory** — Everything currently assembled into the prompt. Note the subtlety in the pseudocode: retrieved long-term memories and recent episodic turns *become* short-term memory the moment they're pasted into the prompt.

## T

**Thought** — The first ReAct step: the agent reasoning about what it knows and what it needs next. Not always visible — see the Brief's gotcha about empty reasoning steps.

**`@tool` decorator** — LangChain's way of turning a plain Python function into something an LLM can call. Critically, **the function's docstring becomes the tool's description** — the only text the model reads when deciding whether this tool applies. A vague docstring is a broken tool.

**Tool** — A function the agent can invoke. Three here, all backed by hardcoded Python dictionaries so there are no network failures: `lookup_order`, `get_product_details`, `check_inventory`.

**`tool_calls`** — The structured list attached to an `AIMessage` when the model wants to invoke tools. Note it is a *list* — a model can request several tools in one turn, and the notebook's traces show exactly that.

**`temperature=0`** — Maximum determinism. Chosen throughout so the traces are reproducible in a classroom.

## V

**Vector store** — A database that indexes embeddings and answers "what's most similar to this?" — ChromaDB, FAISS, Pinecone. The storage layer for long-term memory.

[🔝 Back to top](#top)
