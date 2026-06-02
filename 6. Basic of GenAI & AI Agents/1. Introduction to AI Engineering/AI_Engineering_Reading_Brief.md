<a id="top"></a>
# AI Engineering — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`AI_Engineering_Jargon_Card.md`](./AI_Engineering_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `Introduction to AI Engineering.ipynb` in this folder.

---

## 🎯 30-second TL;DR

**AI Engineering is the discipline of building production apps powered by pre-trained foundation models** (LLMs, diffusion models, multimodal models) — *without* training them from scratch.

Where **ML Engineering** asks *"how do I train a model?"*, **AI Engineering** asks *"how do I use the world's best model effectively?"*

The notebook is a **map of the entire AI Engineering stack**. It walks you through 4 layers in order:

1. **LLM Providers** — where models come from (OpenAI API vs Hugging Face open-source).
2. **App Development Ecosystem** — the supporting tools (vector DBs, RAG, evaluation, finetuning, experiment tracking).
3. **LLM Frameworks** — orchestration libraries (LangChain, LangGraph) and agent systems.
4. **LLM Infrastructure** — how to deploy efficiently (vLLM, quantization, local vs cloud).

It also demonstrates three patterns that are the daily bread of AI Engineering: **prompt engineering** (zero-shot, few-shot, Chain of Thought), **RAG** (retrieval-augmented generation), and **AI agents** (ReAct loop with tools).

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Definition** — AI Engineering vs ML Engineering. The work shifts from *training* models to *using* them.
2. **The 4-layer landscape** — providers → app dev → frameworks → infrastructure.
3. **LLM Providers** — OpenAI SDK (`gpt-4o-mini` chat completions) and Hugging Face (`TinyLlama` via `transformers.pipeline`); a side-by-side comparison.
4. **Prompt Engineering** — three techniques with running examples: zero-shot, few-shot (in-context learning), Chain of Thought (CoT).
5. **App Dev Ecosystem** — tools table: vector DBs, finetuning (TRL), quantization (GPTQ/AWQ/GGUF), evaluation (Ragas), security (garak), tracking (Comet, W&B, MLflow).
6. **RAG end-to-end** — 5 docs → embeddings (1536-dim) → FAISS index → top-2 retrieval → grounded answer.
7. **LLM Evaluation with Ragas** — measuring **Faithfulness** (did the answer stick to the retrieved context?).
8. **Comet ML + tiny finetuning** — tracking a `distilgpt2` (82M params) finetune on 5 Q&A pairs.
9. **LLM Frameworks** — table of orchestration (LangChain, LlamaIndex), agents (LangGraph, CrewAI, AutoGen), finetuning (TRL, Unsloth), structured output (Outlines, LMQL), programmatic (DSPy).
10. **LangChain basics** — LCEL pipe syntax: `chain = prompt | llm | StrOutputParser()`.
11. **AI Agents** — definition, 3 components (LLM + Tools + Control loop), ReAct pattern; a LangGraph math agent with `add`/`multiply`/`subtract` tools.
12. **LLM Infrastructure** — deployment types (local, demo, production, edge); vLLM features deep dive (PagedAttention, continuous batching, quantization, tensor parallelism, speculative decoding, prefix caching).

---

## 🧠 The big idea

> **ML Engineering builds engines. AI Engineering builds cars.**

For most of ML's history, building an AI application meant **training the model yourself** — collecting data, designing architectures, running gradient descent. That's ML Engineering.

But pre-trained foundation models (GPT-4o, Claude, Llama-3) have changed this. They already "know" language, reasoning, and most general tasks **out of the box**. You no longer need to build the engine — you assemble the car around it.

The new work is:

- **Prompting** — telling the model what to do (in English, not Python).
- **Orchestration** — chaining LLM calls with tools, retrievers, memory.
- **Grounding** — feeding the model your private data (RAG) so it answers from facts, not hallucinations.
- **Evaluation** — judging free-form text outputs, which is much harder than judging classification accuracy.
- **Deployment** — making it fast (vLLM), cheap (quantization), and safe (security tools).

This is a *different skill set* from classical ML — closer to systems engineering and API design than to statistics.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook.

### 1. The LLM call — chat completions, roles, temperature

> **🪜 Mental model:** an LLM call is a **function call over a list of messages**. The model reads the conversation so far and returns the next message.

A chat completion is a list of message dicts in, one new message dict out. Each message has a `role` and `content`:

```python
messages = [
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "Explain machine learning in one sentence."}
]
```

- **`system`** = the **persona / rules / context** — sets what the model is and what it should do.
- **`user`** = the question.
- **`assistant`** = the model's reply (used when continuing a multi-turn conversation).

**The four parameters that matter most:**

| Parameter | Meaning |
|---|---|
| `model` | Which model to call (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.). |
| `messages` | The conversation so far. |
| `max_tokens` | Cap on response length (one token ≈ 4 characters). |
| `temperature` | Randomness. `0` = deterministic (same input → same output). `0.7` = balanced. `2` = wild. |

**Why it matters in this notebook.** Cell 19 shows the canonical pattern. Cell 35 wraps it in a `get_response()` helper that the prompt-engineering examples reuse. Everything you do later (RAG, agents, evaluation) is **just orchestrating more of these calls**.

### 2. Prompt Engineering — zero-shot vs few-shot vs Chain of Thought

> **🪜 Mental model:** **how you ask** matters as much as **what you ask**. Prompt engineering is the dial you turn before reaching for a finetune.

Three techniques, ordered by how much guidance you give the model:

- **Zero-shot.** Just ask. *"Classify this sentiment as POSITIVE/NEGATIVE/NEUTRAL: ..."* Works when the task is well-known to the model.
- **Few-shot** (also called **in-context learning**). Give 2–5 worked examples in the prompt, then the real question. The model imitates the pattern. Use when you have a custom format or zero-shot is inconsistent.
- **Chain of Thought (CoT).** Add *"think step by step"* before the answer. Forces the model to write out its reasoning, which dramatically improves accuracy on math and multi-step problems.

**Worked example from the notebook (Cell 44).** *"A store sells apples for \$2 each. If you buy 5 or more, you get 20% off the total. How much would 7 apples cost?"* With zero-shot, the model might blurt out a wrong number. With CoT (*"think through this step by step"*), it writes:

> 7 × \$2 = \$14 (subtotal). 7 ≥ 5, so 20% discount applies. Discount = \$14 × 0.20 = \$2.80. Final = \$14 − \$2.80 = **\$11.20**.

**Why it matters.** Prompt engineering is **free** (no training, no GPUs) and is **always the first thing to try** when your model gives bad results. Only when prompting fails do you reach for finetuning.

### 3. Embeddings & Vector Databases

> **🪜 Mental model:** an embedding is a **GPS coordinate for meaning**. Two pieces of text close in coordinate space mean the same thing.

An **embedding** is a long list of numbers (1536 floats for OpenAI's `text-embedding-3-small`) that represents the **meaning** of a piece of text. Two key properties:

1. **Similar texts → similar embeddings.** "I love dogs" and "Puppies are great" land near each other; "I love dogs" and "The stock market crashed" land far apart.
2. **Distance metrics** let you measure "how similar?"

| Metric | Formula | When to use |
|---|---|---|
| **Cosine similarity** | `cos(θ) = A·B / (‖A‖‖B‖)` | Text embeddings (most common) |
| **Euclidean (L2)** | `√Σ(aᵢ − bᵢ)²` | Image embeddings; what FAISS's `IndexFlatL2` uses |
| **Dot product** | `Σ(aᵢ × bᵢ)` | When the embedding magnitude matters |

A **vector database** (FAISS, Chroma, Pinecone, Milvus, Weaviate) stores embeddings and supports **fast nearest-neighbour search**: "give me the 3 documents whose embeddings are closest to this query embedding."

**Why it matters in this notebook.** Vector DBs are the **foundation of RAG**. Cell 53 produces 5 embeddings of shape `(5, 1536)` from 5 sample documents. Cell 55 stores them in `faiss.IndexFlatL2(1536)`. Cell 56 retrieves the top-2 most similar docs for a query. Without embeddings, RAG can't exist.

### 4. RAG — Retrieval-Augmented Generation

> **🪜 Mental model:** an LLM with **open-book exam** access to your private data. Instead of relying on memory, it looks up relevant facts before answering.

LLMs have a problem: they were trained months ago and don't know your company's docs. They also **hallucinate** when asked things they don't know.

RAG fixes this with a **5-step pipeline**:

1. **Chunk + embed** your documents → store embeddings in a vector DB.
2. **Embed the user's question** with the same model.
3. **Retrieve** the top-k most similar docs from the vector DB.
4. **Stuff** those docs into the prompt as context.
5. **Generate** the answer from that context.

**Worked example from the notebook (Cells 52–57):**

- 5 docs about Python, FAISS, RAG, vector DBs, embeddings → 5 embeddings of dim 1536.
- Stored in `IndexFlatL2` (FAISS).
- Query: *"Who created Python?"* → embedded → top-2 retrieved.
- Retrieved docs become the **context** in the next OpenAI call. System message: *"Answer based only on the provided context."*
- Answer: grounded in the actual document text.

**Why it matters.** RAG is **the** dominant pattern for building LLM apps over private data. ChatGPT-with-your-docs, customer support bots, internal knowledge agents — almost all of them are RAG underneath.

### 5. LLM Evaluation — why it's hard

> **🪜 Mental model:** judging a classifier is "did you pick the right label?" (yes/no). Judging an LLM is "did your essay capture the truth?" (much fuzzier).

Classical ML has clean metrics: accuracy, precision, recall. LLM outputs are free-form text — accuracy doesn't apply. So how do you grade them?

**The Ragas approach** (used in the notebook): use **another LLM** as the judge, and ask it specific questions about the response.

The notebook uses two Ragas metrics:

- **Faithfulness** — "Are the facts in the answer actually supported by the retrieved context?" High = no hallucination. Cell 60 wraps `gpt-4o-mini` as an evaluator and computes a Faithfulness score on the RAG output.
- **Response Relevancy** — "Does the answer actually address the question?" (Listed in the notebook but not computed in code.)

**Why it matters.** In production, **you can't ship what you can't measure.** Evaluation is the #1 differentiator between a demo and a real LLM product. Tools: **Ragas** (RAG-focused), **DeepEval**, **LM Evaluation Harness** (academic benchmarks).

### 6. AI Agents + ReAct

> **🪜 Mental model:** a chatbot **answers**. An agent **does**. Give it tools and a goal, and it figures out the steps itself.

An **AI agent** has three components:

1. **An LLM** — the reasoning engine. Decides what to do.
2. **Tools** — functions the agent can call (calculator, web search, database query, API call).
3. **A control loop** — keeps cycling until the task is done.

**The ReAct pattern** ("Reason + Act") is the dominant loop:

```
1. REASON  — LLM thinks: "what should I do next?"
2. ACT     — LLM picks a tool and arguments: multiply(6, 7)
3. OBSERVE — your code runs the tool, returns 42
4. LOOP    — feed 42 back to the LLM, repeat from step 1
5. STOP    — when the LLM says "no more tools, here's the answer"
```

**Worked example from the notebook (Cells 83–86).** A math agent with three tools (`add`, `multiply`, `subtract`). Query: *"What is 6 times 7?"*

- Step 1: LLM reasons → decides to call `multiply(a=6, b=7)`.
- Step 2: Code executes the tool → result 42.
- Step 3: Result fed back to LLM.
- Step 4: LLM has enough info → returns final answer: *"42."*

**Safety guard:** `max_iterations=5` — agents can loop forever if they keep deciding to call more tools; cap it.

**Why it matters.** Agents extend LLMs from **passive responders** to **active problem-solvers**. They're how you get "GPT-4 with internet access" or "Claude that can edit your code." Every modern AI assistant uses some flavour of this loop.

### 7. Inference Infrastructure — vLLM, quantization, batching

> **🪜 Mental model:** in production, the bottleneck moves from *"does it work?"* to *"how many users per dollar?"*

**vLLM** is a high-throughput inference engine — ~24× faster than naive HuggingFace — that stacks several tricks: **PagedAttention** (manages the KV cache like OS memory pages), **continuous batching** (new requests join in-flight batches, no idle GPU), **quantization** (16-bit → 4/8-bit weights for 2–4× memory savings), **tensor parallelism** (split a 70B model across GPUs), **speculative decoding** (small draft model predicts; big model verifies), and **prefix caching** (reuse cached system-prompt KV values). All six combined are why vLLM dominates production LLM serving.

---

## 🔥 The headline pipeline — at a glance

The notebook teaches the **AI Engineering stack** as four layers. Here's the whole map on one screen:

| Layer | What it does | Tools the notebook touches | Code example |
|---|---|---|---|
| **1. LLM Providers** | Where models come from | OpenAI SDK, Hugging Face `transformers` | Chat completion with `gpt-4o-mini`; HF pipeline with `TinyLlama` |
| **2. App Dev Ecosystem** | Supporting tools for building LLM apps | FAISS (vector DB), Ragas (evaluation), Comet ML (tracking), Trainer (finetuning) | End-to-end RAG; Faithfulness scoring; finetune `distilgpt2` |
| **3. LLM Frameworks** | Orchestration libraries | LangChain (LCEL pipes), LangGraph (agents) | `prompt \| llm \| parser` chain; ReAct math agent |
| **4. LLM Infrastructure** | Deploying at scale | vLLM, Ollama, Gradio, SkyPilot | (Conceptual + shell commands for vLLM serving) |

**Patterns that recur across the stack:**

- **RAG** = layers 1 + 2 (provider + vector DB).
- **Agents** = layers 1 + 3 (provider + framework with tool use).
- **Production deployment** = layers 1 + 4 (provider + serving engine).

---

## 🧮 Key parameters / knobs to memorise

The knobs that show up over and over in the notebook code:

- **`temperature`** (0–2) — `0` = deterministic (use for RAG, classification); `0.7` = balanced; `2` = chaotic.
- **`max_tokens`** (OpenAI) / **`max_new_tokens`** (Hugging Face) — cap on response length. One token ≈ 4 chars.
- **`stream=True`** — token-by-token streaming for snappy UIs.
- **Embedding dim** — `text-embedding-3-small` → 1536; the dimension is fixed per model.
- **`k`** (top-k retrieval) — how many docs to retrieve. Notebook uses `k=2`. Higher = more context, more noise.
- **`max_iterations`** — agent loop cap (5 in the notebook). Prevents runaway tool calls.
- **Quantization bit-width** — 16→8→4 bits; each step halves memory.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–7** | Definition; AI Engineering vs ML Engineering; 4-layer landscape | **Read carefully** — ~3 min. This is the framing for the whole notebook. |
| **8–23** | LLM Providers; OpenAI SDK with `gpt-4o-mini`; chat completion params | **Focus** — ~5 min. The canonical LLM call pattern. |
| **24–32** | Hugging Face `transformers.pipeline`; TinyLlama; OpenAI vs HF table | **Read normally** — ~3 min. |
| **33–45** | Prompt engineering: zero-shot → few-shot → CoT, with running sentiment / math examples | **FOCUS** — ~6 min. The most reusable skill in the whole notebook. |
| **46–61** | App dev ecosystem table; **full RAG pipeline** (5 docs → embeddings → FAISS → retrieve → generate); Ragas Faithfulness eval | **FOCUS — this is the headline section** — ~10 min. |
| **62–69** | Comet ML + `distilgpt2` finetuning — experiment tracking | **Skim code, read the structure** — ~3 min. The point is the pattern, not the model. |
| **70–75** | LLM Frameworks table; LangChain LCEL basics | **Read normally** — ~3 min. |
| **76–87** | AI Agents — definition, ReAct, full LangGraph math agent | **FOCUS** — ~6 min. The agent loop is the future of LLM apps. |
| **88–95** | LLM Infrastructure; vLLM features deep dive | **Read carefully, no code to run** — ~4 min. Conceptual but important. |

**Total target read time for the notebook itself:** ~45 min. Add this brief's ~22 min and you're at **~65–70 min**, vs. a cold read (with jargon Googling) of 90–120+ min.

---

## ✅ Walk-away checklist

After reading the notebook, you should be able to say in your own words:

- [ ] **The difference between AI Engineering and ML Engineering** — using vs training pre-trained foundation models.
- [ ] **The 4-layer AI Engineering stack** — providers, app dev, frameworks, infrastructure — and one tool per layer.
- [ ] **The three prompt-engineering techniques and when to use each** — zero-shot for simple, few-shot for custom format, CoT for reasoning.
- [ ] **What an embedding is and why vector DBs exist** — meaning as coordinates; similarity search by distance.
- [ ] **The 5 steps of a RAG pipeline** — embed docs, store, embed query, retrieve top-k, generate with context.
- [ ] **The ReAct loop** — reason → act → observe → repeat → stop.
- [ ] **Why LLM evaluation is hard** — free-form text; you need LLM-as-judge metrics like Faithfulness.
- [ ] **One reason vLLM is faster than naive inference** — PagedAttention, continuous batching, quantization (pick one).

If any of these feel shaky after the notebook, come back to the relevant primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. You want a customer-support bot to answer based on your company's internal docs (which the LLM was never trained on). Which AI Engineering pattern do you reach for, and what are its 5 steps?
2. Your sentiment classifier is giving inconsistent answers with a single instruction prompt. Two prompt-engineering techniques you could try, in order of effort?
3. You set `temperature=0` and `temperature=1.5` for the same prompt. What's the difference in behaviour, and which is right for a RAG answer-generation step?
4. In the notebook's math agent, the LLM is asked *"What is 6 times 7?"*. List the 5 steps of the ReAct loop the agent goes through to answer.
5. You're paying \$X for inference on raw HuggingFace. Name two things vLLM does that would reduce that cost, and explain (in one sentence each) why.

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **RAG (Retrieval-Augmented Generation).** Steps: (1) chunk and embed all internal docs; (2) store embeddings in a vector DB (FAISS, Chroma, etc.); (3) embed the user's question with the same model; (4) retrieve the top-k most similar docs; (5) put those docs in the prompt as context, then call the LLM.

2. **Few-shot prompting first** (give 2–5 examples in the prompt — zero training cost). If still inconsistent, **Chain of Thought** (ask the model to "think step by step"). If both fail, *then* consider finetuning — but it's expensive and usually unnecessary.

3. **`temperature=0`** = deterministic; the model picks the most-likely next token every time, producing the same answer for the same input. **`temperature=1.5`** = creative/random; the model samples more freely. For RAG answer generation you want **`temperature=0`** — you need a faithful summary of the retrieved context, not creative writing.

4. **(1) Reason** — LLM decides it needs the `multiply` tool. **(2) Act** — LLM emits `multiply(a=6, b=7)`. **(3) Observe** — your code runs the tool, returns `42`. **(4) Feed back** — `42` is added to the conversation as a tool message. **(5) Stop** — LLM sees the tool result, decides no more tool calls are needed, returns the final answer "42." (`max_iterations=5` is the safety cap that prevents infinite loops.)

5. Any two of: **PagedAttention** (manages the KV cache in fixed-size blocks like OS memory pages → less wasted memory, more concurrent requests); **Continuous batching** (new requests join an in-flight batch instead of waiting → GPU stays 100% busy); **Quantization** (4-bit or 8-bit weights instead of 16-bit → 2–4× less memory, faster math); **Speculative decoding** (small draft model proposes tokens, big model only verifies → ~2× speedup); **Prefix caching** (cache common system-prompt prefixes → faster repeat queries). Combined gain is ~24× throughput vs raw HuggingFace.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./AI_Engineering_Jargon_Card.md)
