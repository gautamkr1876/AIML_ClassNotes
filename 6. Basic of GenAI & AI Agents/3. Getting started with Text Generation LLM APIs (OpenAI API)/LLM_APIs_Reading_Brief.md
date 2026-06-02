<a id="top"></a>
# LLM APIs & Decoding — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~20 minutes. By the time you reach the notebook, every word will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`LLM_APIs_Jargon_Card.md`](./LLM_APIs_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `Getting_Started_with_Text_Generation_LLM_APIs.ipynb` in this folder.

---

## 🎯 30-second TL;DR

**Two big ideas drive this notebook:**

1. **Not every LLM is the same.** There are **four flavours**: *base* (raw next-token predictor), *instruct* (follows instructions), *chat* (multi-turn dialogue), and *reasoning* (thinks before answering). The way you prompt each is different. Use the wrong flavour and you get nonsense.
2. **Generation = sampling from a probability distribution**, and the way you sample is controlled by a handful of knobs: **temperature**, **top-k**, **top-p**, **greedy**, **beam**. Same model, different decoding knob = different output.

The notebook builds intuition for both, working with **OpenAI**, **Anthropic**, and **HuggingFace** clients side-by-side.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Environment & multi-provider setup** — initialise OpenAI, Anthropic, and HuggingFace clients with API keys via `os.environ` and `getpass`.
2. **Generation models vs reasoning models** — when to use `gpt-4o-mini` (generation) vs `o1-mini` (reasoning). Worked example: a loan-approval decision.
3. **Base model behavior** — GPT-2 prompted with `"What is the capital of France?"` continues text instead of answering.
4. **Instruct model behavior** — TinyLlama responds to the same prompt correctly.
5. **Chat model behavior** — multi-turn conversation with context retention.
6. **The four-way taxonomy** — base / instruct / chat / reasoning, and which to use when.
7. **Chat templates** — the special-token formatting rules across Llama 3, DeepSeek, Qwen, Phi, Gemma. `apply_chat_template()` hides the differences.
8. **How LLMs actually generate text** — at each step there's a probability distribution over the full vocabulary; you have to *pick* one token.
9. **Temperature** — the dial that controls how *sharp* or *flat* that distribution is. Mathematical: `softmax(logits / T)`. Visualised at `T = 0.5, 1.0, 2.0, 5.0`.
10. **Decoding strategies** — greedy (pick top-1), beam search (track top-k partial sequences), top-k sampling, top-p / nucleus sampling.
11. **Phi-3 hands-on** — load the model with `float16` + `device_map="auto"`, use `apply_chat_template`, run greedy and temperature experiments.
12. **OpenAI Chat Completions API** — the canonical pattern with messages, finish reasons, and multi-turn flow.

---

## 🧠 The big idea

> **Understanding *how* a model was trained tells you *how* to prompt it.**

Every modern LLM starts as a **base model** — pretrained to predict the next token over trillions of tokens of text. A base model doesn't "know" what an instruction is. Prompt it with a question; it'll continue your text with more questions.

To make it useful, you add **stages of post-training**:

- **Instruction tuning (SFT)** — fine-tune on `(instruction, response)` pairs. Now it answers.
- **Chat fine-tuning** — fine-tune on multi-turn dialogues with role markers. Now it can hold a conversation.
- **RLHF / preference tuning** — align its answers with human preferences. Now it's helpful and safe.
- **Reasoning training** — train it to "think" before answering. Now it solves multi-step problems.

So when you're prompting:
- A **base** model — give it a partial paragraph to continue, not a question.
- An **instruct** model — give it a clear instruction.
- A **chat** model — give it a list of messages with roles.
- A **reasoning** model — *just give it the goal* and let it think. Don't add "step by step" — it does that itself.

---

## 📖 Core concept primers

### 1. The four flavours of LLM — base / instruct / chat / reasoning

> **🪜 Mental model:** each flavour is the previous one + a layer of training on top.

| Flavour | Trained on | What it does | Example | When to use |
|---|---|---|---|---|
| **Base** | Trillions of tokens of raw text | Predicts the next token | `gpt2`, `Llama-3-8B` (base) | Embedding extraction, finetune starting point — almost never raw |
| **Instruct** | + `(instruction, response)` pairs | Follows single-turn instructions | `Llama-3-8B-Instruct`, `Phi-3-mini-instruct` | One-shot classification, single-turn tasks |
| **Chat** | + multi-turn dialogues with role markers | Holds a conversation, remembers context | `gpt-4o-mini`, `Claude 3.5 Sonnet`, `Llama-3-Instruct` | Chatbots, assistants, agents |
| **Reasoning** | + structured reasoning data, often with RL | Thinks step-by-step internally before answering | `o1-mini`, `o1`, `DeepSeek-V3.1` (with `<think>` blocks) | Math, planning, multi-step logic |

**Worked example from the notebook (the loan approval).** Same task — *"Given debt $50K, income $100K, credit score 700, decide loan."*

- **Generation model (`gpt-4o-mini`):** you need to spell out the reasoning steps in the prompt. Otherwise the model gives a one-line answer with no justification.
- **Reasoning model (`o1-mini`):** you give *only the objective and constraints*. It does the step-by-step reasoning on its own.

### 2. Chat templates — why every model family is different

> **🪜 Mental model:** under the hood, a "conversation" is just one long token string with **special markers** telling the model where each turn starts and ends.

The chat completions API hides this. Internally, when you send:

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
]
```

Each model family wraps it in **its own** special-token format:

- **Llama 3.1:**
  ```
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are helpful.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  Hello<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  ```
- **Phi-3:** `<|system|>You are helpful.<|end|><|user|>Hello<|end|><|assistant|>`
- **ChatML (Qwen, OpenAI):** `<|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...`

**Why it matters.** If you send Llama-formatted tokens to a Phi model (or vice versa), the model has no idea what you're saying — those special tokens mean nothing to it.

**The fix:** always use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` — this uses the **right** template for whatever model the tokenizer belongs to.

### 3. How an LLM actually generates text (the probability distribution)

> **🪜 Mental model:** at each step, the model produces a list of probabilities over its **entire vocabulary** (50K–128K tokens). You have to pick one. *How* you pick is your decoding strategy.

For a vocab of 50,257 tokens, each generation step:
1. The model produces 50,257 **logits** (raw scores).
2. `softmax(logits)` → 50,257 **probabilities** that sum to 1.
3. You pick one token. Now it's in the sequence.
4. Repeat.

So the question isn't *"what is the model's answer?"* — it's *"how do you pick from the distribution?"*

### 4. Temperature — the sharpness knob

> **🪜 Mental model:** **temperature divides the logits before softmax.** Low `T` makes the distribution peaked; high `T` flattens it.

**The formula:**

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

**In words:** *divide every logit by temperature T, then take the standard softmax.*

- `T = 0` (limit): one-hot. Same as greedy. Always pick the top token. **Deterministic.**
- `T = 1`: no change. The original distribution.
- `T = 2`: distribution flattens. More tokens become viable.
- `T → ∞`: uniform. Every token equally likely. Pure chaos.

**Practical guide:**

| Use case | Temperature |
|---|---|
| Classification, factual lookup, RAG answer generation | `0` (deterministic) |
| Coding, structured output | `0.1 – 0.3` |
| General chat | `0.7` |
| Creative writing, brainstorming | `0.9 – 1.2` |
| Above 1.5 | Usually broken output |

### 5. Decoding strategies — greedy, beam, top-k, top-p

> **🪜 Mental model:** **greedy** = always the top token (boring but coherent). **Sampling** = pick from a distribution (creative but riskier). **Beam** = look ahead.

**Greedy decoding.** At each step, `argmax(probabilities)`. Always the most-likely token. Deterministic. Problem: gets stuck in repetitive loops (`"the the the the..."`) and misses better high-probability *sequences* because it's locally optimal.

**Beam search.** Keep the top-`k` partial sequences at each step (e.g., `k=5`). At the end, return the sequence with the highest cumulative probability. Better for translation and summarization where a globally good sequence matters.

**Top-k sampling.** Only consider the top `k` tokens (e.g., `k=40`). Zero out the rest. Renormalise. Sample from the remainder. Stops the model from picking truly absurd tokens.

**Top-p (nucleus) sampling.** Sort by probability; take the smallest set whose cumulative probability ≥ `p` (e.g., `p=0.9`). Sample from that set. **Adaptive** — uses fewer tokens when the distribution is peaked, more when it's flat. The modern default.

**Real-world default for chat.** `temperature=0.7, top_p=0.9, do_sample=True` — produces fluent, varied output without going off the rails.

### 6. The OpenAI Chat Completions API — the canonical pattern

> **🪜 Mental model:** an LLM call is a **function call over a list of messages**. Messages in, one new assistant message out.

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain ML in one sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)

reply = response.choices[0].message.content
finish_reason = response.choices[0].finish_reason  # "stop" | "length" | "tool_calls"
```

**Multi-turn:** keep appending `{role: "assistant", content: reply}` then `{role: "user", content: ...}` to the messages list across calls. The model is **stateless** — you have to send the whole conversation every time.

---

## 🔥 The headline taxonomy — at a glance

| Choose this model... | When your task is... | Prompt it like... |
|---|---|---|
| **Base** (`gpt2`) | An embedding source or finetune starting point | Don't, unless you're finetuning |
| **Instruct** (`Phi-3-mini-instruct`, `Llama-3-8B-Instruct`) | Single-turn instruction-following (classify, summarise) | `"Classify this sentiment: ..."` |
| **Chat** (`gpt-4o-mini`, `Claude 3.5 Sonnet`) | Multi-turn dialogue, agents, RAG | Messages list with system + user + assistant |
| **Reasoning** (`o1-mini`, `o1`, `DeepSeek-V3.1`) | Math, multi-step logic, planning | *Just state the goal and constraints* — don't add "think step by step" |

And once you've picked the model, pick the decoding knob:

| Goal | Decoding | Temperature | Top-p |
|---|---|---|---|
| Classification, RAG, deterministic output | Greedy | `0` | — |
| General chat | Top-p sampling | `0.7` | `0.9` |
| Creative writing | Top-p sampling | `1.0` | `0.95` |
| Translation / summarization | Beam search | — | — |

---

## 🧮 Key knobs to memorise

- **`temperature`** (0–2) — distribution sharpness. `0` = deterministic. `0.7` = balanced. `>1` = creative/chaotic.
- **`max_tokens`** (OpenAI) / **`max_new_tokens`** (HF) — length cap. Hitting it → `finish_reason="length"`.
- **`top_p`** (0–1) — nucleus threshold. `0.9` is the chat default.
- **`top_k`** (int) — keep only the `k` most-likely tokens. `40` is common.
- **`do_sample`** (HF, bool) — `True` = sample; `False` = greedy/beam.
- **`num_beams`** (HF, int) — beam-search width. `1` = greedy.
- **`finish_reason`** — `stop` (good), `length` (truncated; raise `max_tokens`), `tool_calls` (agent).

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–10** | Provider setup (OpenAI, Anthropic, HF); API keys | **Skim** — ~3 min. Just check the patterns. |
| **11–25** | Generation vs reasoning models (loan-approval demo) | **Focus** — ~5 min. The contrast is the takeaway. |
| **26–45** | Base / instruct / chat behaviour, side-by-side on GPT-2 → TinyLlama | **FOCUS** — ~7 min. This is the core taxonomy. |
| **46–55** | Chat templates across model families (Llama, Phi, Qwen, Gemma) | **Read normally** — ~4 min. The point is "they differ; `apply_chat_template` hides it." |
| **56–75** | How generation works → temperature math → distribution visualisations | **FOCUS** — ~8 min. The softmax-with-temperature formula is foundational. |
| **76–90** | Decoding strategies (greedy / beam / top-k / top-p), Phi-3 hands-on | **Focus** — ~6 min. Run the comparison cells. |
| **91–98** | OpenAI Chat Completions API, multi-turn conversation, finish reasons | **Read carefully** — ~3 min. The pattern you'll reuse everywhere. |

**Total target read time for the notebook itself:** ~35–40 min. Add this brief's ~20 min and you're at **~55–60 min** — much faster than a cold read.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **The four LLM flavours** — base, instruct, chat, reasoning — and which to pick for what task.
- [ ] **Why a base model can't answer questions** — it was only trained to continue text.
- [ ] **What a chat template is** and why every model family needs its own.
- [ ] **The temperature formula** — `softmax(logits / T)` — and what `T=0`, `T=1`, `T=2` do.
- [ ] **The four decoding strategies** — greedy, beam, top-k, top-p — and one sentence each on when to use them.
- [ ] **The Chat Completions API pattern** — messages in, one message out; stateless; multi-turn means re-sending the whole list.
- [ ] **The three finish reasons** — `stop`, `length`, `tool_calls` — and what each one means.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. You feed `"What is the capital of France?"` to a **base** GPT-2. Why does it output `"What is the capital of England? What is the capital of..."` instead of `"Paris"`?
2. Write the temperature-applied softmax formula. What happens when `T → 0`? When `T → ∞`?
3. Your chat model gets stuck in repetitive loops (`"the the the..."`). You're using `temperature=0`. What single change would you make and why?
4. You send a Llama-3-formatted prompt to a Phi-3 model and get garbage output. What went wrong?
5. You ask `o1-mini` to plan a multi-step task. Should you append `"Think step by step before answering"` to the prompt? Why or why not?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **A base model was only trained on next-token prediction over raw text** — it has never been shown `(instruction, response)` pairs. Statistically, the most likely text continuation after `"What is the capital of France?"` in its training data is a similar follow-up question or list, not an answer. To get a question answered, you need an **instruct** or **chat** model.

2. **`P(token_i) = exp(logit_i / T) / Σ_j exp(logit_j / T)`.** As `T → 0`, the distribution becomes one-hot — the top-logit token gets probability 1 and everything else gets 0 (same as greedy/argmax). As `T → ∞`, the distribution becomes uniform — every token has the same probability `1/vocab_size`, i.e., pure random.

3. **Switch to sampling** (e.g., `temperature=0.7, top_p=0.9`, or set `do_sample=True` in HF). Greedy decoding picks the highest-probability token at every step, which can lock into a self-reinforcing repetitive loop. Sampling injects controlled randomness that breaks the loop.

4. **You used the wrong chat template.** Each model family has its own special-token format. Llama uses `<|begin_of_text|>...<|eot_id|>`; Phi uses `<|system|>...<|end|>`. Phi doesn't recognise Llama's tokens — they look like random text. Fix: use `phi_tokenizer.apply_chat_template(messages, ...)` so the right template is applied.

5. **No.** Reasoning models like `o1-mini` already perform an internal chain-of-thought automatically — adding "think step by step" can actually *harm* their performance because they end up doing reasoning twice (once externally as you prompted, once internally as they were trained to). Just state the **goal and constraints**; let the model handle the reasoning structure.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./LLM_APIs_Jargon_Card.md)
