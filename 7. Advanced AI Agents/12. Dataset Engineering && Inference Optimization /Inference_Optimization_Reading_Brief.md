<a id="top"></a>
# LLM Inference Optimization — Reading Brief

> **Read this ONCE, end to end, before the notebook.** ~20 min. By the time you open the notebook, every term will make sense — you'll be confirming, not learning blind.
>
> **Side reference:** keep [`Inference_Optimization_Jargon_Card.md`](./Inference_Optimization_Jargon_Card.md) open in a tab; look up unknown words there.
>
> **Notebooks:** `LLM_Inference_Optimization.ipynb` (the full walkthrough) and `Live_Class_Notebook_Inference_July_23_2026.ipynb` (the live-coded version). Both run on a free Colab **T4 GPU** with tiny models (`gpt2`, `gpt2-medium`).
>
> **Theory source:** framing follows Aleksa Gordić's *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* (aleksagordic.com/blog/vllm).

---

## 🎯 30-second TL;DR

**LLM generation is slow because it is autoregressive — one forward pass produces exactly one token, and each token drags all the model's weights out of memory again.** This notebook teaches the two most important tricks that fight that:

1. **KV Caching** — *stop recomputing what you already computed.* Store each token's Key/Value vectors so every new token is O(1) attention work instead of O(sequence-length). The cost: GPU memory that grows **linearly** with sequence length — the notebook measures the exact slope.
2. **Speculative Decoding** — *stop wasting the big model on one token at a time.* A cheap **draft** model proposes `k` tokens; the expensive **target** model verifies all `k` in **one** parallel pass; you keep the agreed prefix. Best case: `k+1` tokens for the price of one big-model pass, with **zero** quality loss.

Everything else in modern serving (paged attention, continuous batching, GQA, MLA, prefix caching, chunked prefill) is built on top of these two ideas.

> **Note on the folder name.** The folder is titled *"Dataset Engineering && Inference Optimization"*, but the notebooks currently shipped cover **only the inference-optimization half** (KV caching + speculative decoding). This brief covers what the notebooks actually demonstrate.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Why KV caching exists** — the naive O(n²) recompute tax of autoregressive decoding.
2. **KV caching in action** — `model.generate(..., use_cache=True)` on `gpt2`; prefill vs decode.
3. **Measuring KV-cache memory** — sweep generation lengths `[50,100,200,400,800]`, plot peak GPU memory, read off the per-token slope (~36 KB/token for gpt2).
4. **Why the slope matters** — GQA, MLA, paged attention, KV quantization, prefix caching all attack it.
5. **The decode bottleneck** — even with a perfect cache, one big pass = one token, and it's memory-bandwidth-bound.
6. **Speculative decoding, one step** — draft (`gpt2`) proposes 5 tokens; target (`gpt2-medium`) verifies in one pass; greedy accept/reject; bonus token.
7. **Speculative decoding in a loop** — a minimal `speculative_generate()` vs a `baseline_generate()`, comparing target-model calls and wall-clock time.
8. **The honest caveat** — on tiny model pairs the wall-clock win is modest; the *structural* win (`target calls < N`) is always there; real 2–3× wins need a target *much* bigger than the draft.
9. **The bigger picture** — where these two ideas sit in a full vLLM serving stack.

---

## 🧠 The big idea — two different enemies

There are **two separate villains** in LLM inference, and each trick fights one of them.

**Villain 1: wasteful recomputation.** Naively, to generate token 100 you'd re-run the model over tokens 1–99 *again*, recomputing Keys and Values you already had. That's the O(n²) tax. **KV caching** is the obvious fix — remember the K/V, so each new token only computes its own. Analogy: **taking an exam where every question builds on the last.** Without a cache you re-solve every previous question before answering the next; with a cache you keep your worked answers on the desk and just do the new one.

**Villain 2: the one-token-per-pass ceiling.** Even with a perfect cache, decoding is *serial* and *bandwidth-bound*: each step you haul all the model's weights out of HBM just to emit a single token. That's a lot of memory traffic for one token. **Speculative decoding** breaks the ceiling with a scout-and-general trick: a fast **scout** (draft model) runs ahead and guesses the next few tokens; the **general** (target model) reviews the whole guess *in one glance* and keeps the correct prefix. Analogy: **a boss reviewing a junior's draft.** The junior writes five sentences quickly; the boss reads all five at once and approves up to the first mistake — far faster than dictating each sentence personally.

The punchline the notebook keeps returning to: **decode is memory-bandwidth-bound, and both tricks are ways to get more useful work out of each expensive trip to memory.**

---

## 📖 Core concept primers

Five primers cover the heart of the notebook — each with a **mental model**, plain-English meaning, and the real numbers.

### 1. Autoregressive decoding — the reason it's all slow

> **🪜 Mental model:** dominoes in a line — token *t+1* can't fall until token *t* has fallen. Strictly serial.

A decoder-only transformer generates one token at a time, each conditioned on all the tokens before it. You physically cannot compute the 10th token before the 9th exists. This serial dependency is why you can't just "parallelize your way out" of generation the way you can with training — and it's the shared root of both optimizations in this notebook.

### 2. Prefill vs Decode — the two phases with opposite personalities

> **🪜 Mental model:** prefill = reading the whole question at once (heavy thinking, done once); decode = writing the answer one word at a time (light thinking, done repeatedly).

Generation splits into two phases. **Prefill** processes the *entire prompt in one forward pass*, computing and caching K/V for every prompt token — it's **compute-bound** (lots of tokens, lots of FLOPs) and happens **once** per request. **Decode** is the token-by-token loop afterward: each step runs a forward pass over just the *newest* token, attends against the cache, and appends its K/V — it's **memory-bandwidth-bound** (tiny compute, but you reload every weight from HBM). This asymmetry is the single most important idea in the notebook: **prefill is compute-bound, decode is bandwidth-bound**, which is *why batching decode requests together is so valuable* — the weight I/O gets shared.

### 3. KV Caching — store, don't recompute

> **🪜 Mental model:** keep your worked answers on the desk instead of re-solving every prior question for each new one.

After each pass, the model saves the Key and Value vectors for every token at every layer (`past_key_values`). Next step, only the new token needs fresh K/V/Q; it queries against everything cached. Attention work per generated token drops from O(sequence-length) to O(1). **The price:** memory grows linearly — each token costs `2 × L × H × d × b` bytes (K and V × layers × KV-heads × head-dim × bytes-per-element). For `gpt2` that's ≈36 KB/token, and the notebook's memory-vs-length plot is a near-perfect straight line whose **intercept** ≈ model weights and whose **slope** ≈ per-token KV cost. In production, vLLM doesn't store this as one big tensor — it uses **paged attention** (fixed 16-token blocks from a shared pool, indexed by a block table + `slot_mapping`), which enables **continuous batching** of variable-length requests with no padding.

### 4. The decode bottleneck — why one token per pass is expensive

> **🪜 Mental model:** driving a truck to the store to buy a single apple — the trip (weight I/O), not the cargo (one token), is the cost.

At small batch size, a decode step is memory-bandwidth-bound: you've streamed *all* the model's weights through HBM just to predict one token. That's a terrible ratio of useful output to memory traffic. Two ways to improve it: **batch more requests** (amortize the weight trip across many tokens) or **get more than one token per trip** — which is exactly what speculative decoding does.

### 5. Speculative Decoding — draft, verify, accept

> **🪜 Mental model:** a scout runs ahead and guesses the path; the general reviews the whole guess in one glance and keeps it up to the first wrong step.

A small **draft** model (`gpt2`) cheaply proposes `k` candidate tokens by running its own short loop. The large **target** model (`gpt2-medium`) then runs **one** forward pass over `context + k drafts` — because transformers process all positions in parallel, this single pass reveals the target's preferred token at *every* draft position at once. You walk left-to-right and **accept** matches, **stopping at the first mismatch** (greedy version: accept iff draft == target's argmax). If everything matches you also get a **bonus token** for free (`k+1` total); if something mismatches you take the target's correction there. The accept/reject math guarantees the output is distributed **exactly** as if the target had generated alone — **no quality loss**. The structural win is always **`target calls < N`**; whether that becomes wall-clock speedup depends on how much cheaper the draft is than the target.

---

## ⚠️ Gotchas to watch for in the notebook

1. **`use_cache=True` is already the default** — the notebook sets it explicitly for clarity, but turning it *off* is what would expose the naive O(n²) path. The demo doesn't run the slow path; it reasons about it.
2. **Memory measurement needs a real GPU** — `torch.cuda.max_memory_allocated()` only works on CUDA. On CPU the KV-memory section is skipped; use a T4 runtime.
3. **Logits are offset by one** — the logit at position `t−1` predicts the token at position `t`. The verification slice (`start = len(context) − 1`) looks off-by-one until you remember this.
4. **The wall-clock speedup may be small or negative here** — `gpt2-medium` is only ~3× bigger than `gpt2`, so the draft overhead eats most of the savings. This is *expected*; real 2–3× wins need a much larger target (or a near-free draft like n-gram/EAGLE/Medusa). The reliable signal is `target calls < N`, not the stopwatch.
5. **Speculative decoding helps latency, not throughput** — at high batch size the target is already compute-bound, so speculative tokens compete with real requests. It's a low-batch-size, latency-first optimization.

---

## 🎓 What you should be able to say after reading

- *"Why is LLM decoding slow?"* → It's autoregressive (serial) and, at small batch, memory-bandwidth-bound — each token reloads all weights from HBM.
- *"What does KV caching buy and cost?"* → Buys O(1) per-token attention instead of O(n); costs GPU memory that grows linearly (~36 KB/token for gpt2), which every modern trick (paged attention, GQA, MLA, KV quant, prefix caching) tries to shrink.
- *"How can speculative decoding be free of quality loss?"* → The accept/reject (rejection-sampling) rule provably preserves the target model's distribution; the draft only proposes, the target always has the final say.
- *"When does speculative decoding actually pay off?"* → When the target is *much* more expensive than the draft and the accept rate is high, at low batch size where latency dominates.

[🔝 Back to top](#top)
