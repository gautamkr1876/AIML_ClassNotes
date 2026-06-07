<a id="top"></a>
# Prompt Engineering (Intro) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebooks. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`PromptEng_Intro_Reading_Brief.md`](./PromptEng_Intro_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebooks:** `L6(Jun 4th)_ Prompt Engineering_ Introduction.ipynb` (the main one) and `L6 (Jun 4th)_ opik-demo.ipynb` (a short observability demo) in this folder.

---

## A

**Action (ReAct)** — In the ReAct loop, the step where the model asks to use a **tool** (e.g., `calculator("25*17")`). The model doesn't run the tool itself — it emits the request as text, and your orchestrator code parses it, runs the real function, and feeds the result back. One of the three ReAct steps: Thought → Action → Observation.

**Agent** — A program that wraps an LLM in a loop so it can *take actions*, not just produce text. The notebook's `react_agent()` is a minimal agent: it calls the LLM, runs whatever tool the LLM asked for, and repeats up to `max_steps=5` times until the LLM emits a `FinalAnswer`. ReAct is the foundation pattern behind most agents.

**Augment (in RAG)** — The middle step of RAG: after retrieving relevant documents, you *augment* (enrich) the prompt by pasting those documents in as context before the user's question. "Retrieve → **Augment** → Generate."

## C

**Chain-of-Thought (CoT)** — A prompting technique that asks the model to write out its **intermediate reasoning steps** before giving a final answer, instead of jumping straight to the answer. Triggered by phrases like *"Let's think step by step."* Improves accuracy on math and logic problems (the source paper reports 10–50% gains) because the model can catch its own mistakes mid-stream. Contrast with **simple prompting**.

**Context (RAG)** — The retrieved text you paste into the prompt for the model to answer from. In a RAG prompt it sits between the instructions and the question, usually wrapped in delimiters (`---`). The model is told to answer *only* from this context.

**Context window** — The maximum amount of text (measured in **tokens**) a model can read at once, ranging from ~4K to 128K tokens depending on the model. This is why RAG retrieves only the **top-k** (3–5) most relevant documents — you can't paste in everything.

## D

**Direct prompting** — See **Simple prompting**. (Also called *standard prompting*.)

**`do_sample`** — A Hugging Face generation flag. When `True`, the model picks the next token randomly according to its probabilities (controlled by **temperature**); when `False`, it always picks the single most likely token (deterministic). The notebook sets it `True` with a low temperature for near-deterministic classification.

## F

**Few-shot prompting** — Teaching the model a task by putting **2–3 worked examples** directly in the prompt, then giving it a new case. No training or fine-tuning happens — the model just infers the pattern from the examples. The notebook uses it for financial-sentiment classification (`"Revenue grew 15%" // Positive`, etc.). Contrast with **zero-shot** (no examples).

**Few-shot CoT** — Combining the two: you provide examples that *each show step-by-step reasoning*, teaching the model your domain's reasoning style (the notebook's medical-diagnosis example). More control than zero-shot CoT, but uses more tokens.

**FinalAnswer** — The keyword the ReAct agent's system prompt tells the model to emit when it has enough information to stop looping and answer. The orchestrator watches for `"FinalAnswer:"` in the model output and ends the loop when it appears.

## G

**Generate (in RAG)** — The last RAG step: the model produces an answer grounded in the augmented context. "Retrieve → Augment → **Generate**."

**`generate_openai` / `generate_hf`** — The notebook's two helper functions. `generate_openai` calls the OpenAI API (model `gpt-4o-mini`); `generate_hf` runs a local Hugging Face model (`Phi-2`). Both take a prompt and `temperature`; they exist so every example can swap between a cloud model and a local one.

**Grounding** — Forcing the model to answer *only* from provided context rather than its own training memory. A RAG prompt enforces grounding with instructions like *"Answer using ONLY the information in the context."* Reduces **hallucination**.

**`gpt-4o-mini`** — The small, cheap OpenAI chat model used for every OpenAI call in the notebook. "4o" = the GPT-4o family; "mini" = the lightweight, low-cost variant.

## H

**Hallucination** — When an LLM produces confident-sounding but false information, because it filled a knowledge gap by guessing. RAG fights this by giving the model real source text to ground its answer in. The `opik-demo` notebook measures it with a `Hallucination` metric.

**Hugging Face** — A platform/library hosting open-source models you can download and run yourself. The notebook uses it to run **Phi-2** locally via the `transformers` library, as a free alternative to the paid OpenAI API.

## J

**Jinja2** — A Python templating engine. You write a prompt template once with `{{ placeholders }}` and `{% for %}` loops, then call `.render(...)` to fill in real values. The notebook uses it to build clean, reusable RAG prompts instead of messy f-strings. `Template(text).render(question=q, documents=docs)`.

## L

**LLM (Large Language Model)** — A model trained on huge amounts of text that generates language by repeatedly predicting the next token. Formally described as `P(Y | X, Θ)` — the probability of response **Y** given your prompt **X** and the model's learned weights **Θ**.

**LLM-as-a-Judge** — Using one LLM to *score* another LLM's output (e.g., "is this answer faithful to the context?"). The `opik-demo` notebook's `Hallucination` metric is an example — it uses an LLM to judge whether the answer contains claims not supported by the context.

## N

**Next-token prediction** — The single mechanism every generative LLM runs on: given the text so far, assign a probability to every possible next word and pick one. Typing *"The customer service representative"* makes "responded" (40%) more likely than "disappeared" (2%). Repeat one token at a time and you get full sentences.

## O

**Observability** — The ability to *see inside* a running LLM pipeline — every step's inputs, outputs, latency, token count, and cost — instead of only seeing the final answer. The `opik-demo` notebook is entirely about this. Without it, when an answer is wrong you can't tell *which* step failed (retrieval? the LLM? a slow tool?).

**Observation (ReAct)** — The third ReAct step: the result your code gets back from running the tool, fed to the model so it can reason about what to do next. E.g., `Observation: Tokyo weather - Sunny, 22°C`.

**Opik** — An open-source observability/tracing tool for LLM apps. You add the `@track` decorator to your functions and Opik records the full execution tree (every nested call, its inputs/outputs, timing, tokens, cost) to a dashboard. Also ships evaluation metrics like `Hallucination`.

**Orchestrator** — The plain Python code that *drives* an agent loop: it calls the LLM, parses the requested action, runs the real tool, feeds back the observation, and repeats. The LLM provides the "thinking"; the orchestrator provides the "doing." The notebook's `react_agent()` is one.

## P

**Phi-2** — A small (~2.7B parameter) open language model from Microsoft, run locally via Hugging Face in the notebook. Used wherever the notebook wants a free, on-device alternative to OpenAI.

**Prompt** — Any text you feed an LLM: a question, an instruction, a partial sentence, or a complex task. It's your entire means of controlling the model's behavior — there's no other "programming" surface.

**`P(Y | X, Θ)`** — The mathematical view of an LLM. **X** = your prompt (the input you control), **Y** = the response, **Θ** ("theta") = the model's fixed learned weights. Key insight: in classic ML your features are fixed, but here *you design the features in natural language through the prompt* — that's why prompt engineering matters.

## R

**RAG (Retrieval-Augmented Generation)** — A technique where, instead of relying on the model's memory, you **retrieve** relevant documents from a knowledge base, **augment** the prompt with them, and let the model **generate** a grounded answer. Analogy: an open-book exam. Reduces hallucination and lets you update knowledge without retraining.

**ReAct (Reason + Act)** — A prompting pattern that interleaves **reasoning** and **tool use**: the model alternates Thought → Action → Observation until it can answer. Where CoT just *thinks*, ReAct *thinks while doing research*. Best when the task needs live/external info (weather, stock prices, web search).

**Retrieve (in RAG)** — The first RAG step: pull the most relevant documents from a knowledge base (often a vector database) for a given question. "Retrieve → Augment → Generate."

## S

**Simple prompting** — The most basic interaction: ask a question or give an instruction, get a direct answer with no shown reasoning ("What's the capital of France?" → "Paris"). Also called *standard* or *direct* prompting. Great for factual recall and classification; **fails** on multi-step reasoning (that's when you reach for CoT).

**Structured CoT** — A CoT variant where you explicitly name the reasoning stages in the prompt ("First analyze X. Then calculate Y. Finally conclude Z.") for consistent, parseable output. The notebook's sales-trend analysis uses it.

**System prompt** — A separate instruction block that sets the model's role and rules ("You are a helpful financial research assistant…"), distinct from the user's message. In the OpenAI helper it's passed as `system_prompt=`. ReAct uses it to define the available tools and the Thought/Action/Observation format.

## T

**Temperature** — A dial (0–2) controlling randomness in generation. **Low (0–0.3)** = focused, near-deterministic — use for math, classification, factual answers. **Moderate (0.5–0.7)** = some creativity — analysis tasks. **High (0.8+)** = creative but error-prone — *avoid for CoT*, since it introduces logical inconsistencies.

**Thought (ReAct)** — The first ReAct step: the model's written reasoning about what to do next, before it chooses an Action. Makes the agent's decision process visible and debuggable.

**Token** — The unit an LLM reads and generates — roughly a word-piece (3–4 characters). Models are limited by a **context window** measured in tokens, and APIs bill per token, which is why `opik` tracks token usage as a cost signal.

**Tool** — An external function the model can ask to use during ReAct: the notebook implements a `calculator`, a `prime_checker`, a `search_knowledge` knowledge base, and a real `web_search` (via DuckDuckGo / `ddgs`). Tools give the LLM abilities it lacks (exact arithmetic, live data).

**`@track`** — Opik's decorator. Put it above any function and Opik automatically captures that function's inputs, outputs, runtime, and any nested tracked calls — building the full trace tree with zero extra code.

**`track_openai`** — Opik's wrapper for the OpenAI client (`client = track_openai(OpenAI())`). Every chat completion made through the wrapped client is automatically logged as a trace, including the prompt, response, tokens, and cost.

**Trace** — One complete recorded execution of your pipeline in Opik: the tree of every step that ran, with timings and data at each node. The thing you inspect in the dashboard to find *where* a wrong answer came from.

## Z

**Zero-shot prompting** — Giving the model a task with **no examples** — just the instruction and the input. The model relies entirely on its training. Contrast with **few-shot** (examples included).

**Zero-shot CoT** — The simplest way to trigger chain-of-thought: just append *"Let's think step by step"* to the prompt. No examples needed, works across domains, but gives you less control over the reasoning structure than few-shot CoT.

[🔝 Back to top](#top)
