<a id="top"></a>
# Scaling AI Applications — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before the notebook, then keep it in a side tab — when an unknown word appears, look it up here in 20 seconds instead of Googling.
>
> **Companion:** read [`Scaling_Reading_Brief.md`](./Scaling_Reading_Brief.md) FIRST for the punchline and big picture. This card is just the dictionary.
>
> **Notebooks:** `Scaling AI Applications.ipynb` (theory + CLI) and `Live Class Notebook - Scaling AI Applications.ipynb` (hands-on on a Colab T4 GPU).

---

## A

**All-reduce** — A group ("collective") step where every GPU sends its numbers to all others and everyone ends up with the combined (summed/averaged) result. In **data parallelism** it's how GPUs agree on one shared gradient (performed by NVIDIA's **NCCL** library over NVLink/InfiniBand). The key sync point in multi-GPU training; when it's slow you're "communication-bound."

**All-gather** — A collective where each GPU holds one *piece* and afterwards every GPU holds the *whole* thing. In **column-parallel** tensor parallelism it glues each GPU's partial output back into the full output. Sibling of all-reduce, but concatenates rather than sums.

**Arrival rate (request rate)** — How many new requests hit the server per second (the `--request-rate` flag). Low = GPU idle; high = requests queue and latency spikes. Sweeping it low→high finds your **saturation point**.

**Auto-tuning** — Automatically searching over serving knobs (batch size, request rate, parallelism degree) for the config that hits your latency/throughput goals. In this notebook it's done informally by sweeping batch sizes `[1,2,4,8,16,32]` and request rates.

## B

**Batch / Batch size** — A group of requests processed together in one GPU pass. Bigger batches keep the GPU busy (better **throughput**) but each request may wait longer (worse **latency**) — the core knob the notebook sweeps. Distinct from **streaming** (*when* tokens come back for one request); batching = *how many* requests share a pass.

**Bubble (pipeline bubble)** — Idle GPU time in **pipeline parallelism** at the start and end of a batch, when some GPUs have no work yet (waiting for an earlier stage to hand them data). Roughly `(p−1)/(p−1+m)` where `p` = stages and `m` = micro-batches — more micro-batches shrink it. The main efficiency cost of PP, which smart schedules (**GPipe**, **1F1B**, **ZB-H2**, **DualPipe**) try to minimise.

## C

**Collective (collective operation)** — Any multi-GPU communication where a *group* coordinate at once: all-reduce, all-gather, reduce-scatter. Tensor parallelism fires collectives *inside every layer*, which is why it needs a fast link and stays within one machine.

**Column parallelism** — A **tensor parallelism** style splitting a weight matrix by its *output* columns, so each GPU computes a slice of the output, stitched back with all-gather. In the notebook's NumPy demo, `W1` (4×6) splits into two 4×3 halves across "GPU 0/1" and concatenating their outputs reproduces the original exactly. Paired with **row parallelism** for the next layer.

**Compute-bound vs communication-bound** — The "what's limiting me?" mental model. **Compute-bound** = GPU math is the bottleneck (good — hardware is busy). **Communication-bound** = GPU-to-GPU sync is the bottleneck (bad — the network starves the math). Every parallelism strategy shifts you between these regimes.

**Concurrency** — How many requests are in flight at once from the client. Low = GPU under-used; moderate = vLLM batches them for more throughput; excessive = requests queue and tail latency explodes. The notebook raises it to watch the server **saturate**.

**Continuous batching** — vLLM's scheduling trick: instead of waiting for a fixed batch to all finish, it swaps completed requests out and new ones in every decode step, keeping the GPU busy. A big reason vLLM's throughput beats naive serving.

**CUDA** — NVIDIA's platform for running general computation on GPUs. vLLM requires a **CUDA-capable GPU**; the notebook runs on a Colab **T4**.

## D

**Data parallelism (DP) / Distributed Data Parallel (DDP)** — The simplest multi-GPU strategy: a *full copy* of the model on each GPU, splitting the *data* (batch) across them; gradients sync with **all-reduce**. Use when the model *fits* but traffic is high. Its limit: it does **not** shrink per-GPU memory (weights fully replicated), so it can't help a too-big model. Contrast with **tensor/pipeline parallelism** (split the model itself).

**Decode (phase)** — The token-by-token generation stage: after the prompt is read, the model produces one output token at a time, each step reading the **KV cache**. Memory-bandwidth-hungry; governs **ITL/TPOT**. Contrast with **prefill**, which reads the whole prompt in one shot.

## E

**E2E latency (End-to-End latency)** — The total wall-clock time to fully answer one request, from submission to the last output token. Formally `TTFT + sum(all ITLs)`. This is the number a single user actually feels for a complete answer.

**Expert Parallelism (EP)** — A parallelism style for **Mixture-of-Experts (MoE)** models that spreads the many "expert" sub-networks across GPUs. An edge case: combine EP for expert layers with DP for attention.

## F

**FSDP (Fully Sharded Data Parallel)** — A memory-saving upgrade to data parallelism that *shards* weights, gradients, and optimizer states across GPUs (gathering each only when needed) instead of replicating. What you reach for — with **ZeRO** — when even the model *state* won't fit; the gap plain DDP can't cover.

## G

**GEMM (General Matrix Multiply)** — The big matrix-times-matrix operation that dominates transformer compute (attention projections, MLP layers). Tensor parallelism works by splitting these GEMMs across GPUs. When GEMMs are large enough to keep GPUs busy you're in the good "compute-bound" regime.

**Goodput** — Throughput that *actually meets your quality bar* — counting only requests that finished within their **SLO**. Raw throughput can look great while goodput is poor (many "completed" requests were unacceptably slow). The metric that reflects real user happiness; the notebook computes it with explicit SLO limits.

**GPU memory utilization** — The fraction of GPU memory vLLM is allowed to grab (the notebook sets `gpu_memory_utilization=0.82`, leaving ~18% for overhead). Most of it goes to the **KV cache**, so this knob directly controls how many concurrent requests fit.

## H

**Hugging Face** — The hub the Qwen models are downloaded from. The notebook pre-downloads with `huggingface-cli download` so download time doesn't pollute the benchmark numbers.

## I

**ITL (Inter-Token Latency)** — The gap between two consecutive output tokens as a response streams — the cost of *one* **decode** step. High ITL = text "types" slowly. Sibling of **TPOT** (the average ITL over a whole response).

**Interconnect** — The physical link between GPUs. Fast links (**NVLink**, **InfiniBand**) support tensor parallelism's heavy per-layer chatter; slow links (Ethernet) push you toward pipeline parallelism, which talks less. The "which parallelism?" decision hinges on this.

## K

**KV cache (Key-Value cache)** — The store of "keys" and "values" the model computed for every past token, so it doesn't recompute them each **decode** step. Grows with sequence length and eats GPU memory fast — the biggest memory consumer in serving, and why long prompts spike tail latency and why **PagedAttention** exists.

## L

**Latency** — The wall-clock time for *one* request to finish, request-in to response-out (ms/seconds) — what a single waiting user feels. Its sparring partner is **throughput**; optimizing one often hurts the other. Don't confuse them: latency = "how long for *this* request", throughput = "how many *per second* overall".

## M

**Micro-batch** — A small slice of the global batch. Pipeline parallelism feeds micro-batches through the stages so multiple GPUs stay busy on different ones at once. More micro-batches → smaller **bubble** but more activation memory.

**Mixture-of-Experts (MoE)** — A model design with many specialist sub-networks ("experts") where each token uses only a few. Scaled with **Expert Parallelism**; an edge case in the vLLM decision guide.

## N

**NVLink** — NVIDIA's high-speed GPU-to-GPU link *inside a single machine*. Its bandwidth is why **tensor parallelism** stays within one node — TP's per-layer collectives need a fat pipe. Without NVLink (e.g., L40S cards) the notebook advises preferring **pipeline parallelism**.

**Node** — One physical machine holding one or more GPUs. Rule of thumb: **tensor-parallel within a node** (fast NVLink), **pipeline-parallel across nodes** (slower links). `tensor_parallel_size` ≈ GPUs/node, `pipeline_parallel_size` ≈ number of nodes.

## O

**Offline inference** — Running the model directly in Python (`llm.generate(prompts)`) with all prompts at once, no HTTP server. Measures peak batch throughput; the notebook's first baseline. Contrast with **online serving**.

**Online serving** — Running the model behind an HTTP server (`vllm serve`) that answers requests as they arrive, like a real app. The notebook exposes an **OpenAI-compatible API** and hits it with the OpenAI SDK.

**OpenAI-compatible API** — vLLM's HTTP endpoint mimics OpenAI's chat-completions format, so the standard **OpenAI Python SDK** points at your *local* vLLM server and "just works" — no OpenAI-hosted model involved. How the notebook calls its self-hosted Qwen model.

## P

**PagedAttention** — vLLM's signature memory trick: stores the **KV cache** in fixed-size "pages" (like OS virtual memory) instead of one contiguous block, so no memory is wasted on padding and more requests fit. The core reason vLLM serves so many concurrent users on modest hardware.

**Parallelism (model parallelism)** — Umbrella term for splitting the *model itself* across GPUs (tensor and pipeline parallelism), as opposed to splitting the *data* (data parallelism). You reach for it when the model is too big for one GPU.

**Percentile (P50 / P90 / P95 / P99)** — Describes the *distribution* of latencies, not just the average. "P95 = 400 ms" means 95% of requests beat 400 ms and 1 in 20 was slower. P50 = median (typical); P99 = the **tail** (unlucky few). SLOs use percentiles because averages hide slow outliers.

**Pipeline parallelism (PP)** — Splitting the model *by layers*: first block on GPU 0, next on GPU 1, etc.; data flows stage by stage. Talks little (point-to-point activation hand-offs), tolerates slow **interconnects**, used *across nodes*. Its cost is the **bubble**. Contrast with **tensor parallelism** (splits *within* a layer).

**Prefill (phase)** — The first stage: the model reads the *entire* prompt in one big forward pass before generating. Dominates **TTFT** — a long prompt = long prefill, spiking that request's TTFT and delaying others on the GPU. Contrast with **decode** (one token at a time, after).

## Q

**Qwen2.5-0.5B / 1.5B-Instruct** — The small open-weights chat models used in the notebook (0.5 billion params in the hands-on Colab run, 1.5 billion in the CLI benchmarks). Small on purpose so everything fits on a free **T4** GPU.

## R

**Reduce-scatter** — A collective that both combines numbers *and* splits the result across GPUs (each keeps only its slice). Used in **row-parallel** TP and, with all-gather, is how DDP's all-reduce works under the hood.

**Replica** — One complete copy of the model. **Data parallelism** makes multiple replicas and routes different requests to each for throughput. In vLLM, `--data-parallel-size 2` + `--tensor-parallel-size 4` = 2 replicas of 4 GPUs.

**Request rate** — See **Arrival rate**.

**Row parallelism** — A **tensor parallelism** style splitting a weight matrix by its *input* rows; each GPU computes a partial sum, combined with all-reduce/reduce-scatter. Paired with **column parallelism** (column first, row second) so each transformer block needs only one communication point.

## S

**Saturation (server saturation)** — The point where more incoming load no longer raises throughput — requests just queue and latency shoots up. Found by cranking **concurrency** or **request rate** until the throughput curve flattens and tail latency spikes. It marks your safe operating limit.

**Sharding** — Splitting one big thing (a weight matrix, KV cache, optimizer state) into pieces spread across GPUs so no single one holds it all. Tensor parallelism shards weights; FSDP/ZeRO shard model *state*.

**SLO (Service Level Objective)** — A concrete performance promise, e.g., "P95 TTFT under 500 ms." The yardstick that turns raw **throughput** into **goodput** — a request only "counts" if it met the SLO.

**Straggler** — The slowest GPU in a synchronized group. Because DDP's **all-reduce** waits for everyone, one slow GPU drags down the whole step.

**Streaming** — Sending output tokens to the client as they're generated rather than waiting for the full answer; it's what makes **TTFT** and **ITL** observable ("types out"). Distinct from **batching**: streaming = *one* request's token delivery; batching = grouping *many* requests.

## T

**T4 (Tesla T4)** — The modest single GPU in Colab's free tier (16 GB) that runs the whole notebook — its small memory is why the notebook uses a 0.5B model, `half` precision, and `max_model_len=2048`.

**Tail latency** — The latency of the *slowest* requests (P95/P99) — what users complain about even when the average looks fine ("averages lie"). LLM serving is prone to it from variable prompt lengths, **KV cache** pressure, and prefill/decode contention. Rule of thumb: P99 > 3–5× P50 = a tail problem.

**Tensor parallelism (TP)** — Splitting a *single layer's* weight matrix across GPUs so each does part of the matmul (intra-layer). Shrinks per-GPU model memory ~`1/TP_degree` and fits big models, but fires a **collective** every layer — needs fast **NVLink**, stays *within one node*. Contrast with **pipeline parallelism**. Typical degrees: 2, 4, 8.

**Throughput** — The sustained rate of completed work: requests- or tokens-per-second across *all* users — the fleet-wide, cost-efficiency metric. Its opposite pull is **latency**: a big **batch** raises throughput but can hurt any one request. Training is throughput-bound; inference is latency-sensitive but throughput-bound at scale.

**TPOT (Time Per Output Token)** — The average **ITL** across a response — a smoothed measure of decode speed, roughly `(E2E − TTFT) / output_tokens`. Lower = the answer streams out faster once started.

**TTFT (Time To First Token)** — The wait from sending a request to the *first* output token. Dominated by the **prefill** phase, so long prompts inflate it. The "is it even working?" latency before any text appears; SLOs frequently target it.

## V

**vLLM** — The high-performance open-source LLM serving engine at the center of this notebook. Provides **offline** (`llm.generate`) and **online** (`vllm serve`) inference, an **OpenAI-compatible API**, a `vllm bench` CLI, and flags for all three parallelism modes. Its speed comes from **PagedAttention** + **continuous batching**.

**`vllm bench`** — vLLM's built-in benchmark CLI with three modes: `latency` (best-case per-step timing), `throughput` (peak offline, all requests at once), and `serve` (realistic online traffic with random *Poisson* arrivals and a controllable **request rate**). Reports TTFT, TPOT, ITL, E2E, and **goodput**.

## Z

**ZeRO** — A family of memory-optimization techniques (Microsoft DeepSpeed) that **shards** optimizer states, gradients, and weights across GPUs so giant models fit. Named alongside **FSDP** as the answer when plain data parallelism runs out of memory. (Zero-*bubble* pipeline schedules like **ZB-H2** are a separate idea despite the name.)

---

[🔝 Back to top](#top) · [→ Reading Brief](./Scaling_Reading_Brief.md)
