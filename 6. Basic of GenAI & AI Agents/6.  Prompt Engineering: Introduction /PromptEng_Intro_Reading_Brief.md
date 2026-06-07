<a id="top"></a>
# Prompt Engineering (Intro) — Reading Brief

> **Read this ONCE, end to end, before opening the notebooks.** Target time: ~22 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`PromptEng_Intro_Jargon_Card.md`](./PromptEng_Intro_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebooks:** `L6(Jun 4th)_ Prompt Engineering_ Introduction.ipynb` (main) + `L6 (Jun 4th)_ opik-demo.ipynb` (a 5-minute observability demo) in this folder.

---

## 🎯 30-second TL;DR

A **prompt** is your only steering wheel for an LLM — and the notebook shows that *how* you phrase the prompt changes *what* the model can do, no retraining required. It walks up a ladder of prompting techniques, each one unlocking a harder class of task:

> **Simple → Few-shot → Chain-of-Thought → ReAct → RAG → Templating**

The punchline: the *same model* goes from "can answer trivia" to "can reason through multi-step math" to "can use a calculator and search the web" — purely by changing the prompt structure. The companion `opik-demo` notebook adds the missing piece: once your pipeline has many steps, you need **observability** to see which step broke.

---

## 🗺️ Agenda — what the notebooks teach, in order

1. **Setup** — install Opik, define reusable `generate_openai` / `generate_hf` helpers (OpenAI's `gpt-4o-mini` + local `Phi-2`).
2. **What a prompt is** — and the `P(Y | X, Θ)` view: you control the model's "features" through text.
3. **Next-token prediction** — the one mechanism all LLMs run on.
4. **Simple prompting** — anatomy (task / context / output spec); when it works and when it fails.
5. **Few-shot prompting** — teaching by 2–3 examples; financial-sentiment classification.
6. **Chain-of-Thought (CoT)** — make the model show its reasoning; zero-shot vs few-shot vs structured.
7. **ReAct (Reason + Act)** — interleave thinking with real tool calls (calculator, prime check, web search).
8. **RAG (Retrieval-Augmented Generation)** — ground answers in retrieved documents; good vs bad RAG prompts.
9. **Prompt templating with Jinja2** — separate prompt structure from content for clean, reusable prompts.
10. **(opik-demo)** — trace a pipeline with `@track`, wrap OpenAI with `track_openai`, score outputs with an LLM-as-a-Judge `Hallucination` metric.

---

## 🧠 The big idea — the prompt *is* the program

In traditional software you write exact commands. With an LLM you don't — you **communicate intent**, and the model interprets it based on patterns it learned from text.

The notebook frames this mathematically as **`P(Y | X, Θ)`**: the model computes the probability of a response **Y** given your prompt **X** and its frozen weights **Θ** ("theta"). Compare to classic machine learning:

- **X (prompt)** ↔ your features
- **Y (response)** ↔ your prediction
- **Θ (weights)** ↔ the trained model

**The transferable analogy: prompting is feature engineering in plain English.** In classic ML your features are fixed columns in a table. With an LLM, *you invent the features on the fly by how you write the prompt.* Add examples → you've engineered a "pattern-matching" feature. Add "think step by step" → you've engineered a "reasoning" feature. That's the whole reason prompt engineering is a skill: same model **Θ**, different **X**, dramatically different **Y**.

---

## 📖 Core concept primers

Six primers cover the heart of the notebooks. Each has a **mental model**, plain-English meaning, a tiny example from the notebook, and why it matters here.

### 1. Next-token prediction

> **🪜 Mental model:** an autocomplete that has read the entire internet — it always picks a likely next word, never "knows" the answer.

Every generative LLM does exactly one thing: given the text so far, it assigns a probability to every possible next token and picks one, then repeats. Type *"The customer service representative"* and the model rates "responded" at ~40%, "helped" ~30%, … "disappeared" ~2%. Stitch these one-at-a-time picks together and you get fluent text. **Why it matters here:** this is *why* prompting works at all — by choosing your words you shift those next-token probabilities, and that's your only lever. It also explains hallucination: the model will always produce a plausible next token, even when it has no real knowledge.

### 2. Simple vs Few-shot prompting

> **🪜 Mental model:** simple prompting = *asking*; few-shot = *showing worked examples first, then asking*.

**Simple prompting** (a.k.a. standard/direct) is a bare question or instruction → direct answer. Perfect for facts ("capital of France?"), classification, and format conversion. It **breaks** on multi-step reasoning and complex calculation. **Few-shot prompting** adds 2–3 labeled examples inside the prompt, then a new case; the model infers the pattern and applies it — *no fine-tuning*. The notebook's example:

```
"Revenue grew by 15% in Q2."          // Positive
"Acquisition talks were terminated."   // Negative
"The CEO will retire in December."     // Neutral
Now classify: "Strong earnings, raised guidance." //   ← model fills in "Positive"
```

**Why it matters here:** few-shot is the cheapest way to specialize a general model to *your* labels and *your* style.

### 3. Chain-of-Thought (CoT)

> **🪜 Mental model:** "show your work" — the model that writes out the steps gets more of them right.

CoT asks the model to produce **intermediate reasoning** before the final answer, instead of leaping to a conclusion. Trigger it three ways: **zero-shot** (just add *"Let's think step by step"*), **few-shot** (show examples that each include reasoning), or **structured** (name the stages: "First… then… finally…"). The source paper reports 10–50% accuracy gains on math, commonsense, and symbolic tasks, because making each step explicit lets the model catch its own errors mid-stream. **Rule of thumb:** if a human expert would think it through, use CoT; if it's instant recall, skip it (CoT adds cost and can make creative output mechanical). **Why it matters here:** the notebook directly contrasts a "direct" prompt that fumbles a word problem against a CoT prompt that nails it.

### 4. ReAct (Reason + Act)

> **🪜 Mental model:** CoT is thinking to yourself; ReAct is thinking *while doing research*.

ReAct lets the model **interact with the outside world** by looping through three steps: **Thought** (reason about what to do) → **Action** (request a tool, e.g. `calculator("17*29")`) → **Observation** (your code runs the real tool and feeds back the result). Repeat until the model emits `FinalAnswer`. Crucially, the LLM only *emits text*; the **orchestrator** (plain Python — the notebook's `react_agent()`, capped at `max_steps=5`) parses the action, runs the actual function (`calculator`, `prime_checker`, `search_knowledge`, `web_search`), and returns the observation. **Why it matters here:** ReAct is the seed of every AI *agent*. Use it when the task needs live or external info (weather, stock price, web search); it's overkill for self-contained problems and adds latency (multiple LLM calls).

### 5. RAG (Retrieval-Augmented Generation)

> **🪜 Mental model:** an open-book exam — instead of memorizing everything, the model looks up the relevant page and answers from it.

RAG has three steps: **Retrieve** relevant documents from a knowledge base → **Augment** the prompt by pasting them in as context → **Generate** an answer grounded in that context. This reduces hallucination, keeps knowledge up-to-date without retraining, and lets answers cite sources. The notebook stresses that a *good* RAG prompt does three things a bad one doesn't:

1. **Enforce grounding** — "Answer using ONLY the provided context."
2. **Give a fallback** — "If the context is insufficient, say 'I don't know based on the provided context.'"
3. **Require citations** — "Cite the document ID, e.g. [Doc 1]."

**Why it matters here:** without grounding, the model invents a generic answer ("most companies allow 2–3 days remote…"); with it, the model answers from *your* actual policy doc. Respect the **context window** (4K–128K tokens) — retrieve only the top 3–5 documents.

### 6. Prompt templating with Jinja2

> **🪜 Mental model:** a fill-in-the-blanks form — write the structure once, drop in values forever.

As prompts get complex (especially RAG, which interleaves instructions, looping documents, and a question), hardcoding them with f-strings gets messy and error-prone. **Jinja2** separates *structure* from *content*: you write a template with `{{ question }}` placeholders and `{% for doc in documents %}` loops, then call `.render(documents=..., question=...)` to produce the final prompt. **Why it matters here:** the notebook's RAG template loops over retrieved docs to build `[doc_1] ...`, `[doc_2] ...` automatically — clean, reusable, and easy to maintain.

---

## 🔥 The technique ladder — at a glance

Each rung handles what the rung below it can't. **Real behaviors from the notebook:**

| Technique | What you add to the prompt | Unlocks | Notebook example |
|---|---|---|---|
| **Simple** | Just the question | Facts, classification | "Function of mitochondria?" → direct answer |
| **Few-shot** | 2–3 labeled examples | Your labels / style, no training | Financial sentiment → Positive/Negative/Neutral |
| **CoT** | "Let's think step by step" | Multi-step reasoning (+10–50% acc) | Word problem: direct fails, CoT solves |
| **ReAct** | Tool list + Thought/Action/Observation loop | Live data, exact math, web search | "17×29, then is it prime?" → calls calculator + prime_checker |
| **RAG** | Retrieved docs + grounding rules | Answers from *your* documents | "What's our return policy?" → answers from doc_1 |
| **Templating** | Jinja2 placeholders | Clean, reusable complex prompts | RAG template loops over retrieved docs |

> **Observability layer (opik-demo):** once a pipeline chains retrieval + LLM + tools, a wrong answer could come from *any* step. Opik's `@track` decorator records the full trace tree (inputs, outputs, latency, tokens, cost) so you can see exactly *where* it broke — and its `Hallucination` metric (an **LLM-as-a-Judge**) scores whether the answer is actually supported by the context.

---

## 🧮 The one "formula" to internalize

This notebook is light on math — its single load-bearing idea is the LLM equation:

```
P(Y | X, Θ)
```

**Word-by-word translation:** "the probability of generating response **Y**, *given* prompt **X** and the model's fixed learned weights **Θ** (theta)."

**Worked reading:** You can't change **Θ** (that's baked in at training). You *can* change **X**. So all of prompt engineering is the search for the **X** that makes the desired **Y** most probable. Few-shot, CoT, and RAG are all just different ways of reshaping **X**.

**Temperature** is the other knob worth memorizing — not a formula, a setting:

```
temperature ↑  ⇒  more random / creative
temperature ↓  ⇒  more focused / deterministic
```

Notebook guidance: **0–0.3** for math, classification, factual answers; **0.5–0.7** for analysis; **avoid 0.8+ for CoT** (it injects logical inconsistencies).

---

## 🗺️ Notebook reading map

**Main notebook** (`...Introduction.ipynb`, ~89 cells):

| Cells | What it teaches | How to read |
|---|---|---|
| 2–12 | Setup: installs, API keys, `generate_openai`/`generate_hf` helpers, smoke tests | **Skim.** Just note the two helpers exist and which model each uses. |
| 13–15 | What a prompt is, `P(Y\|X,Θ)`, next-token prediction | **Focus.** This is the conceptual core. |
| 16–29 | Simple prompting + few-shot prompting | **Focus.** Compare the prompt shapes; note `temperature` choices. |
| 31–46 | Chain-of-Thought: zero-shot, structured, few-shot | **Focus.** Watch direct-vs-CoT on the same word problem. |
| 48–74 | ReAct: the loop, simulated tools, then the real `react_agent` orchestrator | **Focus + slow down.** The orchestrator parsing logic (cells 70–71) is the key takeaway. |
| 75–79 | RAG: why, prompt template, bad vs good prompt | **Focus.** Memorize the three good-RAG rules. |
| 80–88 | Jinja2 templating for RAG | **Read once.** Note `Template(...).render(...)`. |

**opik-demo notebook** (~24 cells):

| Cells | What it teaches | How to read |
|---|---|---|
| 0–1 | The problem: "the answer is wrong — but *which step* failed?" | **Focus.** This motivates everything. |
| 2–7 | Demo 1: `@track` a 3-function RAG pipeline → nested trace | **Read.** See how `@track` builds the tree. |
| 8–10 | Demo 2: trace an agent's multiple tool calls | **Skim.** Same idea, agent flavor. |
| 11–14 | Demo 3: `track_openai` auto-logs real OpenAI calls | **Read.** |
| 15–21 | Demo 4: LLM-as-a-Judge — the `Hallucination` metric | **Focus.** Connects back to RAG grounding. |

---

## ✅ Walk-away checklist

After the notebooks, you should be able to say, in your own words:

- [ ] Why an LLM is `P(Y|X,Θ)` and why "you control the features through X."
- [ ] The difference between simple, few-shot, and zero-shot prompting.
- [ ] When to reach for **CoT** vs when it's wasted overhead.
- [ ] The three steps of the **ReAct** loop and who runs the tool (hint: not the LLM).
- [ ] The three steps of **RAG** and the three rules of a *good* RAG prompt.
- [ ] Why **Jinja2** templating beats f-strings for complex prompts.
- [ ] What **observability** buys you and what `@track` records.

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. In `P(Y | X, Θ)`, which symbol can a prompt engineer actually change, and which is frozen?
2. You ask a model "Is the result of 17×29 a prime number?" and want it to *use a calculator*. Which technique do you need — CoT or ReAct — and why?
3. A RAG chatbot answers a question with a generic, plausible-sounding answer that isn't in your documents. Name the one prompt instruction most likely missing.
4. Your math word-problem prompt keeps producing inconsistent reasoning. You're running it at `temperature=0.9`. What change does the notebook recommend, and why?
5. A user says "the RAG answer is wrong." Without observability you only see `answer = "Paris"`. Name two things Opik's trace would let you inspect to find *where* it went wrong.

<details>
<summary>Answers</summary>

1. **X** (the prompt) is what you change; **Θ** (the model's trained weights) is frozen. (Y is the output, a consequence of X and Θ.)
2. **ReAct.** CoT only makes the model *reason in text* — it can't run a real calculator and might still do the arithmetic wrong. ReAct lets the model emit an Action (`calculator("17*29")`) that your orchestrator actually executes, then check the result with `prime_checker`.
3. **Grounding** — the instruction "Answer using ONLY the provided context" (and a fallback like "say 'I don't know based on the provided context'"). Without it the model falls back on its training memory and hallucinates.
4. Lower the temperature to **0.2–0.4**. High temperature (0.8+) adds randomness that introduces logical inconsistencies into chain-of-thought reasoning; math/logic want near-deterministic output.
5. Any two of: which documents were **retrieved**, the exact **prompt** sent to the LLM, the **LLM's raw response**, per-step **latency**, **token usage/cost**, or the nested **trace tree** showing whether retrieval or generation was the failing step.

</details>

[🔝 Back to top](#top)
