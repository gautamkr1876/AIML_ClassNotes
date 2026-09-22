<a id="top"></a>
# AutoGen (AG2) Multi-Agent Patterns — Jargon Card

> **How to use.** Skim once (~5 min) before the notebook, then keep open as a side reference. Plain English first, formal term second.
>
> **Companion:** read [`AutoGen_MultiAgent_Reading_Brief.md`](./AutoGen_MultiAgent_Reading_Brief.md) first — it carries the punchline, the agenda, and the concept primers.
>
> **Notebook:** `FinTrack_MultiAgent_AutoGen_Gemini_Lab__1_.ipynb` — 11 sections, ~45 min, Gemini only (no OpenAI key anywhere).

---

## A

**AG2 / AutoGen** — The same project under two names. **AutoGen** is Microsoft's original multi-agent framework; **AG2** is the actively-maintained fork. You `pip install ag2` but you `import autogen`. The notebook pins `ag2[gemini]==0.14.0` deliberately: the `0.x` line has the classic API this lab teaches (`AssistantAgent`, `GroupChat`, code executors), while `1.x` is a ground-up rewrite (`Agent`, `Task`, `Toolkit`) with an incompatible programming model.

**AssistantAgent** — The "brain" agent. It is backed by an LLM and generates replies guided by its `system_message`. Every one of the five case-study analysts is an `AssistantAgent`; they differ *only* in their system message.

## C

**ConversableAgent** — The base class that both `AssistantAgent` and `UserProxyAgent` inherit from. It supplies the send/receive/reply machinery; the subclasses just pick sensible defaults.

**`chat_history`** — The transcript of a conversation, a list of `{"role": ..., "content": ...}` dictionaries. `result.chat_history[-1]["content"]` is the idiom for "give me the last thing that was said" — used throughout the notebook to chain one agent's output into the next agent's input.

**Code execution** — The agent *writes* a Python snippet and something actually runs it. Distinct from tool calling: tool calling invokes one predefined safe function; code execution runs arbitrary (sandboxed) code the model just composed.

**`content_str`** — An AG2 helper that safely turns a message's `content` into a string. Needed because a message's content can be `None` (e.g. a pure tool-call turn) or a list of blocks, not always plain text.

## G

**Gemini** — Google's LLM family. Wired in via `LLMConfig({"model": "gemini-3.5-flash", "api_type": "google", "api_key": ...})`. The `api_type` field is what tells AG2 to route through its Gemini client rather than OpenAI's.

**GroupChat** — The shared "meeting room": a list of participating agents plus one shared message history that **every agent sees in full**. That shared visibility is precisely why agents can agree, disagree, and build on each other.

**GroupChatManager** — The facilitator that sits inside the room and decides whose turn it is to speak next, according to the `GroupChat`'s rules.

## H

**Human-in-the-loop (HITL)** — A deliberate pause where a person approves or rejects what the AI proposes. In this notebook it's literally one `input("approve/reject")` call gating what happens next. Simple, but it's the pattern that stops an AI team from acting unilaterally.

**`human_input_mode`** — Controls when an agent stops to ask a real person. `"NEVER"` (used everywhere here) means fully automated; `"ALWAYS"` prompts on every turn; `"TERMINATE"` prompts only before ending.

## I

**`initiate_chat(other, message=..., max_turns=n)`** — How one agent starts a conversation with another. `max_turns=1` means "send one message, get one reply, stop" — the workhorse for single Q&A.

**`is_termination_msg`** — A predicate you supply that inspects each message and returns `True` when the conversation should stop. The coding section uses `lambda msg: "DONE" in content_str(msg.get("content"))`.

## L

**LLMConfig** — The object holding model name, API type, and key. Every agent in the notebook is handed the *same* `llm_config`; the only thing that makes them different agents is their system message.

**LocalCommandLineCodeExecutor** — The component that actually runs LLM-written Python, as a local subprocess in a `work_dir` folder, with a `timeout`. No Docker required, no shell commands.

## M

**`max_consecutive_auto_reply`** — How many times an agent will auto-respond without human input before stopping. `0` means "receive a reply and go quiet" (the plain "asker" agents); `3` allows a couple of tool round trips.

**`max_round`** — The hard cap on total turns in a `GroupChat`. Set to `11` here. Without it the discussion could run indefinitely and burn API budget — always set one.

**`max_turns`** — The per-conversation cap on `initiate_chat` exchanges. Different from `max_round`, which is per-GroupChat.

**MultimodalConversableAgent** — A vision-capable `AssistantAgent` variant that understands `<img PATH>` tags in a message and ships the image bytes to the model alongside the text.

## O

**OCR (Optical Character Recognition)** — Turning pixels of text into machine-readable characters. The Technical Analyst cites OCR + LLM extraction as the feasible approach for the receipt scanner; Section 11 actually does it with Gemini Vision.

## R

**ReAct (Reason + Act)** — A loop where the agent reasons about which action is needed, takes it, observes the result, and only then answers. Section 10 implements it as Question → Think → Act (run SQL) → Observe (rows) → Answer. The notebook's own framing is the useful one: **ReAct is tool calling applied to reasoning.**

**Reflection** — A three-step self-improvement loop: *draft → critique → revise*. The Writer drafts a one-sentence recommendation, the Critic lists what's missing (privacy, accuracy, cost), and the Writer rewrites addressing it. The revision is visibly more cautious than the draft.

**`register_for_execution()` / `register_for_llm(description=...)`** — The two halves of tool calling, applied as stacked decorators on one Python function. `register_for_llm` **tells the model the tool exists** (the `description` is what it reads to decide); `register_for_execution` **says which agent actually runs it**. Register only one half and the tool either is never called or is called and never executed.

**`round_robin`** — A `speaker_selection_method` where agents speak in fixed rotation. Predictable and demo-safe. The alternative, `"auto"`, lets the manager's LLM pick the next speaker — more natural, less reproducible.

**ROI (Return on Investment)** — `(revenue − cost) / cost × 100`, as a percentage. The tool-calling demo computes it in Python precisely because LLMs are unreliable at exact arithmetic.

## S

**Sequential chain** — The simplest composition: run agent A, take its text reply, paste that text into agent B's message. No framework feature needed — it's an f-string.

**`speaker_selection_method`** — The GroupChat's turn-taking rule. See `round_robin`.

**`system_message`** — An agent's job description and personality, given once at construction. It is *the* lever in this notebook: five identical `AssistantAgent`s become a Product Manager, Market Analyst, Technical Analyst, Security Analyst and Finance Analyst purely through this string.

## T

**Text2SQL** — Turning a natural-language data question into a SQL query, running it, and answering from the real rows. Section 10 does this over an in-memory SQLite `sales` table (4 rows; North = 200000, South = 165000).

**Tool calling / function calling** — The LLM decides, on its own, that it needs a real function, emits a structured call, the framework runs the Python, and feeds the result back. Used for anything the model shouldn't guess: arithmetic, lookups, database queries.

## U

**UserProxyAgent** — The "code/human side" agent. It represents the human or the execution environment: it can send messages, run tools, and execute code, and it does **not** need an LLM. The notebook reuses one called `runner` across several sections purely to send single messages and collect replies.

[🔝 Back to top](#top)
