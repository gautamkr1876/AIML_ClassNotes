<a id="top"></a>
# LLM Inference Optimization (KV Caching + Speculative Decoding) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~7 min) before opening the notebooks. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Inference_Optimization_Reading_Brief.md`](./Inference_Optimization_Reading_Brief.md) FIRST for the story; this card is just the dictionary.
>
> **Covers the notebooks:** `LLM_Inference_Optimization.ipynb` (the full KV-cache + speculative-decoding walkthrough on `gpt2` / `gpt2-medium`) and `Live_Class_Notebook_Inference_July_23_2026.ipynb` (the live-coded version).
>
> **Theory source:** most of the framing follows Aleksa Gordić's *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* (aleksagordic.com/blog/vllm).

---

## A

**Accept / reject step** — The verification rule at the heart of speculative decoding. Walking left-to-right through the draft model's proposed tokens, you *accept* a token if the big model agrees, and *stop at the first disagreement*. Greedy version (used in the notebook): accept a draft token only if it equals the target model's `argmax` at that position. The math guarantees the surviving sequence is distributed **exactly** as if the big model had generated it alone — zero quality loss.

**Argmax** — "Pick the index of the largest value." Applied to the model's logits, `torch.argmax(logits, dim=-1)` returns the single most-likely next-token id. Greedy decoding = argmax at every step; the notebook uses it for both the draft proposal and the target's verdict.

**Autoregressive (decoding)** — Generating text one token at a time, each new token conditioned on every token before it. This left-to-right serial dependency is *why* inference is slow — you cannot compute token 10 until token 9 exists — and it is the bottleneck both KV caching and speculative decoding attack.

## B

**Batch size (and why it flips the bottleneck)** — How many requests the GPU processes together in one forward pass. At **small batch size**, decoding is memory-bandwidth-bound (you stream all the weights to produce very few tokens), so tricks like speculative decoding help. At **large batch size**, the GPU becomes compute-bound and is already busy, so speculative tokens compete with real requests and the trick stops paying off. This is the core "latency vs throughput" tension.

**Bonus token** — In speculative decoding, if the target model accepts *all* `k` draft tokens, its single verification pass also produced a prediction for position `k+1` "for free." You append that as a bonus, so a fully-accepted round yields `k+1` new tokens for the price of one big-model pass.

## C

**Compute-bound** — A workload limited by how fast the GPU can do arithmetic (FLOPs), not by memory movement. **Prefill** (processing the whole prompt at once) is compute-bound: many tokens, lots of matrix multiplies. Contrast with *memory-bandwidth-bound*.

**Continuous batching** — A serving technique where requests of *different lengths* are packed side-by-side and processed together, with finished requests dropping out and new ones slotting in mid-flight — no waiting for the whole batch to finish, no padding to a common length. Paged attention (block tables + `slot_mapping`) is what makes it possible. It is the main throughput lever in vLLM.

**Chunked prefill** — Splitting a long prompt's prefill into smaller pieces so it can be interleaved with ongoing decode steps instead of monopolizing the GPU. Named in the notebook's recap as one of the many layers in a real serving stack (not implemented in the demos).

## D

**Decode (phase)** — The token-by-token generation loop *after* prefill. Each step runs a forward pass over just the **one** newest token, which attends against the cached K/V of all earlier tokens, then appends its own K/V. Decode is **memory-bandwidth-bound**: you reload every model weight from HBM just to emit a single token — which is exactly why batching decode requests together is so valuable (the weight I/O gets amortized).

**Draft model** — The small, fast model in speculative decoding (here `gpt2`, 124M). It cheaply runs ahead and *proposes* `k` candidate tokens. Think "scout who runs ahead." Its guesses are only useful if the big model tends to agree with them.

**Distribution preservation (exactness)** — The guarantee that speculative decoding's output is statistically identical to sampling token-by-token from the target model. This is why it is a *pure latency optimization*, not an approximation — you are not trading quality for speed.

## E

**EAGLE** — A fast speculative-decoding proposal scheme that replaces most of the big model with a lightweight MLP "draft head" trained to predict the target's next hidden state. One of the schemes vLLM V1 ships instead of a generic second draft LM. Named, not implemented, in the notebook.

## G

**`generate()` (HuggingFace)** — The `transformers` one-call generation loop. It manages a (non-paged) KV cache under the hood via `use_cache=True` (on by default for causal models). The notebook uses it to *observe* KV-cache effects rather than reimplementing the cache by hand.

**GQA (Grouped-Query Attention)** — An architecture trick where several query heads *share* one key/value head, shrinking the KV cache (fewer distinct K/V vectors to store per token). One of the modern methods aimed at making the KV-cache memory slope smaller. Contrast with full multi-head attention (one K/V per query head).

**Greedy acceptance** — The deterministic special case of the accept/reject rule: accept a draft token iff it matches the target's argmax. Simpler than the full stochastic rejection-sampling rule, and what the notebook implements so the logic is visible.

## H

**HBM (High-Bandwidth Memory)** — The GPU's on-package VRAM where model weights and the KV cache live. "Memory-bandwidth-bound" means the bottleneck is *streaming data out of HBM*, not doing math. Understanding HBM traffic is the key to why decode is slow and why the KV cache matters.

## K

**KV cache** — The workhorse optimization. After each forward pass, the model **stores** the Key and Value vectors it computed for every token, at every layer, in a per-request buffer (`past_key_values`). On the next step only the newest token needs fresh K/V/Q; it queries against the cached K/V of all earlier tokens. This turns per-token attention work from O(sequence length) into O(1) — at the cost of memory that grows linearly with sequence length.

**KV cache memory formula** — Each generated token costs `2 × L × H × d × b` bytes: 2 (one K, one V) × `L` layers × `H` KV heads × head-dimension `d` × `b` bytes per element. For `gpt2` (12 layers, 12 heads, dim 64, fp16) that's ≈ 36 KB/token — the notebook measures this empirically as a near-perfect straight line.

**KV cache quantization** — Storing the cached K/V vectors in a lower-precision format (e.g. int8) to shrink the per-token memory cost. Another attack on the linear-memory-growth slope.

## L

**Latency vs throughput** — The two things a serving system trades off. **Latency** = time for one request (how fast a single user sees tokens). **Throughput** = total tokens/second across all requests. KV caching helps both; speculative decoding helps *latency* at low batch size; continuous batching + paged attention help *throughput*. They often pull in opposite directions.

**Logits** — The raw, un-normalized scores the model outputs over the whole vocabulary at each position (shape `[batch, seq_len, vocab_size]`). Softmax turns them into probabilities; argmax picks the top one. In verification, the target model's logits at each draft position reveal its preferred token.

## M

**Medusa** — A speculative-decoding scheme that bolts extra linear "heads" onto the model, each predicting a different future token (`t+1`, `t+2`, …) in parallel, so the draft is essentially free. Named as one of vLLM V1's proposal schemes.

**Memory-bandwidth-bound** — A workload limited by how fast data moves between HBM and the compute units, not by arithmetic. **Decode** is the canonical example: tiny compute (one token) but you must reload all weights. The fix is to amortize that weight traffic across many requests (batching) or across many tokens (speculative decoding).

**MLA (Multi-Head Latent Attention)** — DeepSeek's attention variant that compresses K/V into a small shared latent vector, drastically cutting KV-cache size. Listed alongside GQA as a modern slope-reducing trick.

## N

**n-gram lookup (drafting)** — A "draft model with no model": scan the existing context for a recurring n-gram and reuse whatever tokens followed it last time as the proposal. Cheap and surprisingly effective for repetitive text; one of vLLM V1's proposal methods.

## P

**Paged attention** — vLLM's memory manager for the KV cache. Instead of one big contiguous tensor per request, the cache is chopped into fixed-size **blocks** (typically 16 tokens) drawn from a shared pool; each request holds a list of block IDs, exactly like an OS paging virtual memory. This eliminates fragmentation and enables continuous batching. The notebook explains it but does not reimplement it.

**`past_key_values`** — The concrete object HuggingFace uses to hold the KV cache between decode steps. Passing it forward (or letting `use_cache=True` manage it) is what avoids recomputing K/V for the whole prefix each step.

**Prefill (phase)** — The **first** forward pass, which processes the *entire prompt at once*, computing and caching K/V for every prompt token. It happens once per request and is **compute-bound**. Splitting inference into prefill (compute-heavy, one-shot) and decode (bandwidth-heavy, looped) is the central mental model of the whole notebook.

**Prefix caching** — Reusing the KV cache of a shared prompt prefix across requests (e.g. a common system prompt), so you don't re-run prefill for text you've already seen. Another way to make KV memory *reusable* rather than merely smaller.

## R

**Rejection sampling (stochastic accept)** — The general (non-greedy) accept rule: accept a draft token with probability `min(1, p_target/p_draft)`; on rejection, resample from the rebalanced distribution `max(0, p_target − p_draft)` (renormalized). This is what preserves the target distribution exactly under sampling. The notebook uses the simpler greedy special case.

## S

**Speculative decoding** — The second big optimization. A cheap **draft** model proposes `k` tokens; the expensive **target** model verifies all `k` in a *single* parallel forward pass; you accept the agreed-upon prefix and correct at the first mismatch. Breaks the "one big-model pass = one token" coupling — best case, `k+1` tokens per big pass. Pure latency win, exact output. Introduced by Leviathan et al. and Chen et al. (2023).

**`slot_mapping`** — The per-token index telling the forward pass *which block slot* each token's K/V should be written into when many requests are flattened into one "super sequence." The plumbing that makes paged attention + continuous batching work.

**Super sequence** — The single long concatenation of many requests' tokens that vLLM feeds through one forward pass, using block tables and `slot_mapping` so each token reads/writes the right request's cache. Enables padding-free batching of variable-length requests.

## T

**Target model** — The large, accurate model in speculative decoding (here `gpt2-medium`, 355M) whose distribution you actually want. It has the final say: it verifies drafts and its verdict is what gets emitted. "The general who reviews the scout's plan."

**Target calls (< N)** — The structural win of speculative decoding: generating `N` tokens takes *fewer than* `N` expensive target-model forward passes (baseline needs exactly `N`). The notebook prints this ratio; whether it becomes wall-clock speedup depends on how much bigger the target is than the draft.

**TTFT (Time To First Token)** — Latency until the user sees the first output token — dominated by prefill. A key serving metric (carried over from the Scaling lecture); KV caching and chunked prefill both aim to keep it low.

## U

**`use_cache=True`** — The `generate()` flag that turns the KV cache on (default for causal LMs). With it off, every decode step recomputes K/V for the whole sequence — the O(n²) naive path the whole notebook is arguing against.

## V

**Verification pass** — The single target-model forward pass over `context + k draft tokens`. Because transformers process all positions in parallel, this one call yields the target's preferred token at *every* draft position simultaneously (plus the bonus position) — the reason `k` tokens can be checked for roughly the cost of one decode step.

**vLLM** — The high-throughput open-source LLM serving engine that popularized paged attention and continuous batching. It is the backdrop for the whole lecture; the notebook builds toy versions of ideas vLLM ships in production.

---

## The two headline ideas in one breath

- **KV caching** turns generation from O(n²) into O(n) by *storing* past K/V instead of recomputing them — the price is GPU memory that grows linearly with sequence length. Nearly every serving optimization (paged attention, prefix caching, GQA, MLA, KV quantization) is an attack on that slope.
- **Speculative decoding** breaks the "one big pass = one token" rule: a cheap draft proposes `k` tokens, the target verifies them in one parallel pass, and the accept/reject rule keeps the output *exactly* the target's distribution. Latency win; size of win depends on the draft-vs-target cost gap and the accept rate.

[🔝 Back to top](#top)
