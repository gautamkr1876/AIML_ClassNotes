<a id="top"></a>
# Scaling AI Applications — Reading Brief

> **Read this ONCE, end to end, before the notebook.** ~22 min. By the time you open the notebook, every term will make sense — you'll be confirming, not learning blind.
>
> **Side reference:** keep [`Scaling_Jargon_Card.md`](./Scaling_Jargon_Card.md) open in a tab; look up unknown words there.
> **Notebooks:** `Scaling AI Applications.ipynb` (theory + CLI) and `Live Class Notebook - Scaling AI Applications.ipynb` (hands-on, Colab T4 GPU).

---

## 🎯 30-second TL;DR

**Scaling an AI application is NOT "just add more GPUs." First measure, then diagnose the real bottleneck, then pick the matching strategy.**

The whole notebook orbits one tension: **latency** (how long *one* user waits) vs **throughput** (how many requests/sec the system finishes). Every scaling knob — batch size, streaming, percentiles, parallelism — trades between these two.

The hands-on part serves a small chat model (**Qwen2.5-0.5B-Instruct**) on a free Colab **T4 GPU** with **vLLM** — offline (`llm.generate`) then online (`vllm serve` behind an OpenAI-compatible API) — measuring **TTFT, ITL, TPOT, E2E latency, throughput, goodput**, sweeping batch sizes `[1,2,4,8,16,32]` and rising concurrency until the server **saturates**.

The theory part explains the three ways to use many GPUs:

- **Data Parallelism (DP/DDP)** — copy the whole model on each GPU, split the *data*. Use when the model fits but traffic is high.
- **Tensor Parallelism (TP)** — split *within* a layer, across GPUs in one machine (needs fast NVLink). Use when the model won't fit on one GPU.
- **Pipeline Parallelism (PP)** — split *by layers* across machines (talks little). Use when the model won't fit even on one machine.

---

## 🗺️ Agenda — what the notebooks teach, in order

1. **Latency vs throughput** — the two metrics that define scale, and why they're not just inverses (training = throughput; inference = latency-sensitive, throughput at scale).
2. **The inference metric vocabulary** — TTFT, ITL, TPOT, E2E latency, throughput, goodput.
3. **Prefill vs decode** — the two phases of generation and which metric each drives.
4. **Setting up vLLM on a Colab T4** — install, load Qwen2.5-0.5B, configure precision/context/memory.
5. **Offline inference + batch-size sweep** — `llm.generate`, batch `[1,2,4,8,16,32]`, requests/sec and tokens/sec.
6. **Online serving** — `vllm serve`, OpenAI-compatible API, called with the OpenAI SDK.
7. **Streaming metrics** — measure TTFT, ITL, TPOT from a streamed response.
8. **Percentiles & tail latency** — why P50/P90/P95/P99 beat averages; SLOs; goodput.
9. **Concurrency & saturation** — raise concurrency, watch throughput plateau and tail latency spike.
10. **The `vllm bench` CLI** — latency / throughput / serve modes, request-rate control, goodput.
11. **Three parallelism strategies** — Data, Tensor, Pipeline: how each works, communicates, bottlenecks; plus the NumPy tensor-parallel demo (split `W1` by columns, concatenate, same answer).
12. **The vLLM decision guide** — flags for TP / PP / DP and when to use each.

---

## 🧠 The big idea — the see-saw

Scaling has **one see-saw**: **latency on one side, throughput on the other.**

Analogy — **a coffee shop.** *Latency* is how long *your* cup takes from order to hand-off; *throughput* is how many cups the shop serves per hour. A one-barista shop making your drink instantly has great latency but tiny throughput; a shop that batches ten orders into one espresso pull has great throughput but you wait longer. **Batching** lifts throughput but can hurt your personal wait. **Adding baristas** (GPUs) helps — but only if the bottleneck is the coffee-making (compute), not the baristas shouting orders across the room (communication overhead).

That's the notebook's real thesis: **before adding baristas, find out whether the queue is slow because they're busy (compute-bound — good, add more) or because they keep shouting across the room (communication-bound — more baristas makes it worse).** That's why you *measure first*.

---

## 📖 Core concept primers

Seven primers cover the heart of the notebook — each with a **mental model**, plain-English meaning, and the real numbers.

### 1. Latency vs Throughput

> **🪜 Mental model:** the coffee-shop see-saw — latency is *your* wait; throughput is *cups per hour* for the whole shop. Pushing one down often pushes the other up.

**Latency** is the wall-clock time for *one* request to finish (ms/seconds) — what a single waiting user feels. **Throughput** is the sustained rate of completed work — requests/sec or tokens/sec across *all* users. They are **not simple inverses**: a fleet can have huge throughput while any one request is slow, and a single GPU serving one user can have tiny latency but terrible throughput.

**Why it matters in this notebook.** Every knob — batch size, concurrency, parallelism — moves you along this see-saw. The notebook asks: *is training a latency or throughput problem? Inference?* Answer: **training is throughput** (push tokens/sec; tolerate slow steps); **inference is latency-sensitive** (a user waits) **but becomes throughput at scale** (millions of users, cost per request).

### 2. The inference metrics — TTFT, ITL, TPOT, E2E, goodput

> **🪜 Mental model:** TTFT = "when does it *start*?", ITL/TPOT = "how fast does it *type*?", E2E = "when is it *done*?", goodput = "how many finished *well enough*?"

Generation isn't one number — it's a timeline:

- **TTFT (Time To First Token)** — wait until the first token appears. Set by the **prefill** phase (primer 3). The "is it even working?" latency.
- **ITL (Inter-Token Latency)** — gap between two streamed tokens. The cost of one **decode** step; how fast text "types out."
- **TPOT (Time Per Output Token)** — average ITL over the response; smoothed decode speed.
- **E2E latency** — total time to the last token: `TTFT + sum(all ITLs)`.
- **Throughput** — tokens/sec or requests/sec across the system.
- **Goodput** — throughput *that met the SLO*; the metric that reflects real user happiness.

**Why it matters in this notebook.** The live notebook computes these from streamed responses and offline batch runs (a helper divides `completion time / output tokens` for TPOT); the `vllm bench serve` CLI reports all of them plus goodput against explicit SLO limits.

### 3. Prefill vs Decode — the two phases

> **🪜 Mental model:** prefill = "read the whole question at once" (one gulp); decode = "answer one word at a time" (small sips).

LLM inference has two phases. **Prefill** processes the *entire* prompt in one forward pass before any output — it dominates **TTFT** (long prompt = long prefill = slow first token). **Decode** then generates tokens *one at a time*, each step reading the **KV cache** (saved keys/values of all previous tokens); it's memory-bandwidth-bound and drives **ITL/TPOT**.

**Why it matters in this notebook.** This split explains the metric zoo *and* the tail-latency section: prefill and decode **compete** for the GPU. A burst of long-prompt prefills can stall the decodes of users mid-answer, inflating their ITL — one of the three named causes of **tail latency**.

### 4. Batching & the GPU-utilization trade-off

> **🪜 Mental model:** batching is carpooling — packing more riders (requests) into one trip (GPU pass) uses the vehicle better, but each rider may wait longer to leave.

A **batch** is a group of requests processed together in one GPU pass. Small batch = short waits but the GPU sits half-idle. Large batch = GPU fully fed (high **throughput**) but each request waits longer (worse **latency**).

**Why it matters in this notebook.** The offline experiment sweeps batch sizes `[1, 2, 4, 8, 16, 32]`, measuring wall time, **requests/sec**, and **output tokens/sec**. Tokens/sec climb as batch grows while per-request latency creeps up — the see-saw made concrete. vLLM's **continuous batching** (swapping finished requests out, new ones in, every step) keeps it efficient across the sweep.

### 5. Percentiles & tail latency

> **🪜 Mental model:** "averages lie." A shop with a 5-min average can still leave one customer stuck 40 minutes — the tail is what people rage-tweet about.

A **percentile** describes the *distribution* of latencies. **P95 = 400 ms** means 95% of requests beat 400 ms and 1 in 20 was slower. **P50** = median (typical); **P99** = the **tail** (unlucky 1 in 100). A system can show a 50 ms average while 1 in 100 users waits 2+ seconds — which is why **SLOs** (service promises) are written in percentiles, not averages.

**Why it matters in this notebook.** LLM serving is *especially* tail-prone: (a) **variable prompt lengths**, (b) **KV cache pressure** as sequences grow, (c) **prefill/decode contention**. Rule of thumb: **P99 > 3–5× P50 = a tail problem**. The live notebook's `summarize_latency_df` helper prints mean, P50, P90, P95, P99.

### 6. Concurrency, saturation, and goodput

> **🪜 Mental model:** a highway — light traffic flows freely, moderate traffic packs cars efficiently, but past a point it jams and everyone crawls.

**Concurrency** is how many requests are in flight at once. As you raise it: low = GPU under-used; moderate = vLLM **batches** them for more **throughput**; excessive = requests **queue**, tail latency spikes, throughput stalls — the server has **saturated**. **Goodput** (throughput meeting the SLO) is the honest scorecard: past saturation raw throughput can look OK while most requests are too slow to "count."

**Why it matters in this notebook.** It deliberately raises concurrency (and, via the CLI, `--request-rate`) to *find* the saturation point — your safe operating limit. Tip: sweep request rate 1–2 → 32+ and watch for the latency knee.

### 7. The three parallelism strategies — DP, TP, PP

> **🪜 Mental model:** DP = photocopy the recipe for each cook (split the *orders*); TP = many cooks share *one* dish's steps (split *within* a task); PP = an assembly line, each cook owns one course (split *across* tasks).

When one GPU isn't enough, there are exactly three ways to spread the work, chosen by *what* your bottleneck is (traffic, model-vs-GPU, or model-vs-machine):

| | **Data Parallelism (DP/DDP)** | **Tensor Parallelism (TP)** | **Pipeline Parallelism (PP)** |
|---|---|---|---|
| What's split | The **data** (batch) | **Within** a layer (one matrix) | **Across** layers (blocks of layers) |
| Model copy | Full copy per GPU | Sharded across GPUs | Sharded across GPUs |
| Fixes | Too much *traffic* | Model *too big for one GPU* | Model *too big for one machine* |
| Communication | Gradient **all-reduce** once per step | **Collective inside every layer** (heavy) | Point-to-point activation hand-offs (light) |
| Needs | — | Fast **NVLink**, stays *in one node* | Tolerates slow links, spans *nodes* |
| Cost / limit | Doesn't shrink per-GPU memory | Comm-bound at high degree | The **bubble** (idle GPU time) |

**Why it matters in this notebook.** This table *is* the decision guide. **DDP** replicates + splits data + syncs with all-reduce (identical to one big-batch GPU), but can't help a model that won't fit. **TP** splits each weight matrix — the NumPy demo splits `W1` by columns across two "GPUs" and concatenation reproduces the exact output — but its per-layer collectives demand NVLink, so it stays within a node. **PP** puts layer-blocks on different GPUs with cheap hand-offs, spanning nodes, at the cost of the **bubble**. Rule of thumb: `tensor_parallel_size` = GPUs per node, `pipeline_parallel_size` = nodes; add DP replicas for throughput.

---

## 🔥 The headline experiment — at a glance

Real runs on a single Colab **T4** GPU serving **Qwen2.5-0.5B-Instruct** (`half` precision, `max_model_len=2048`, `gpu_memory_utilization=0.82`).

| Experiment | Knob swept | What you watch | The lesson |
|---|---|---|---|
| **Offline batch sweep** | batch size `1 → 32` | requests/sec, tokens/sec, per-request latency | Bigger batch → higher throughput, but rising per-request latency (the see-saw). |
| **Streaming metrics** | one online request | TTFT, ITL, TPOT, E2E | Generation is a *timeline*; prefill sets TTFT, decode sets ITL. |
| **Percentile analysis** | repeated requests | P50 / P90 / P95 / P99 | Averages hide the tail; watch for P99 > 3–5× P50. |
| **Concurrency ramp** | client concurrency ↑ | throughput vs tail latency | Throughput climbs, plateaus, then **saturates** and tail latency spikes. |
| **`vllm bench serve`** | `--request-rate` | TTFT/TPOT/ITL/E2E + **goodput** vs SLO | Raw throughput ≠ goodput; find the limit. |

**Headline takeaway** (the notebook's central idea): *"Scaling does not simply mean 'add more GPUs.' First measure latency + throughput + tail behaviour + goodput + memory pressure. Then determine whether the real limit is traffic, model size, GPU memory, or GPU communication. Only then choose the serving and parallelism strategy."*

---

## 🧮 Formulas & rules to memorise

This notebook is mostly **conceptual + measured**, not equation-heavy. Only five relationships are load-bearing — memorise these.

### 1. End-to-end latency

```
E2E = TTFT + sum(all ITLs)
```

**In words:** total time for a request = time to the first token + the sum of every gap between later tokens. TTFT comes from **prefill**; the ITLs from **decode**.

### 2. Time Per Output Token (approximation)

```
TPOT ≈ (E2E − TTFT) / output_tokens
```

**In words:** average per-token decode speed = (total time − first-token wait) ÷ tokens produced. **Worked example:** E2E = 2.0 s, TTFT = 0.4 s, 64 tokens → `(2.0 − 0.4)/64 = 0.025 s = 25 ms/token`.

### 3. Throughput (the see-saw formula)

```
throughput ≈ batch_size / latency        (requests or tokens per second)
```

**In words:** work finished per second ≈ how many you process together ÷ how long each pass takes. This is *why* bigger batches raise throughput even as each request's latency rises — the numerator grows faster than the denominator, up to saturation.

### 4. Pipeline bubble fraction

```
bubble ≈ (p − 1) / (p − 1 + m)
```

**In words:** GPU time wasted idling = (stages − 1) ÷ (stages − 1 + micro-batches). More micro-batches `m` shrink the bubble. **Worked example:** `p = 4`, `m = 8` → `3/11 ≈ 27%` idle; push `m` to 32 → `3/35 ≈ 8.5%`.

### 5. Tail-latency rule of thumb

```
if P99 > 3–5 × P50  →  you have a tail-latency problem
```

**In words:** if your worst-1%-of-requests time is more than 3–5× your typical (median) time, investigate scheduling, chunked prefill, or a few long-context requests hogging the GPU.

---

## 🗺️ Notebook reading map — where to spend your attention

The two notebooks split cleanly — theory (`Scaling AI Applications.ipynb`) and hands-on (`Live Class Notebook`); read them together.

| Section | What it teaches | How to read |
|---|---|---|
| **Theory §1–2** (latency/throughput, metrics, prefill/decode) | The vocabulary everything rests on | **Read carefully** — ~8 min. The foundation; the Jargon Card mirrors it. |
| **Live: setup cells** (`nvidia-smi`, install, `MODEL`, helpers) | vLLM install, Qwen2.5-0.5B config, latency helpers | **Skim** — ~3 min. Note the config numbers (T4, half, 2048, 0.82). |
| **Live: offline batch sweep** | `llm.generate`, batch `[1..32]`, results table | **FOCUS** — ~6 min. The see-saw live in the throughput-vs-batch table. |
| **Live: online serving + streaming** | `vllm serve`, OpenAI SDK, TTFT/ITL/TPOT | **Read normally** — ~6 min. How metrics come off a streamed response. |
| **Live: percentiles + concurrency** | `summarize_latency_df`, saturation ramp | **FOCUS** — ~6 min. The "averages lie" and saturation lessons. |
| **Theory §3.2 + `vllm bench` CLI** | latency/throughput/serve modes, goodput | **Reference** — ~4 min. Skim commands; note goodput vs throughput. |
| **Theory §5–7** (DP, TP, PP) | The three strategies in depth | **Read carefully** — ~10 min. Anchor on the primer-7 table. |
| **Live: NumPy TP demo** (splitting `W1`) | Column-parallel matmul via concatenation | **Read normally** — ~4 min. Makes TP concrete. |
| **Theory §8** (vLLM decision guide) | Flags + when to use each | **Reference** — ~3 min. The quick-reference table is the keeper. |

**Total notebook read time:** ~50 min. Add this brief's ~22 min = **~72 min** — and it's *understanding*, not decoding as you go.

---

## ✅ Walk-away checklist

After the notebooks, you should be able to say in your own words:

- [ ] **Latency vs throughput** — what each measures, why they're not inverses, and which one training vs inference cares about.
- [ ] **The inference metric timeline** — TTFT (prefill), ITL/TPOT (decode), E2E, and how goodput differs from raw throughput.
- [ ] **Why bigger batches raise throughput but can hurt per-request latency** — the GPU-utilization trade-off.
- [ ] **Why averages lie** — what P50/P90/P95/P99 mean, why SLOs use percentiles, and the P99 > 3–5× P50 rule.
- [ ] **What server saturation looks like** — throughput plateaus, tail latency spikes; how you find it by ramping concurrency/request-rate.
- [ ] **The three parallelism strategies** — DP (split data, model fits), TP (split within a layer, model too big for a GPU, needs NVLink), PP (split across layers, spans nodes, has a bubble) — and how to pick.
- [ ] **The central rule** — measure and diagnose the bottleneck (traffic / model size / memory / communication) *before* adding GPUs.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. A single GPU serves one user with ultra-low latency but poor throughput. In one sentence, why aren't latency and throughput just inverses of each other?
2. A request has E2E latency 3.0 s, TTFT 0.6 s, and generated 80 output tokens. What's the approximate TPOT in milliseconds per token?
3. Your model *fits* on one GPU, but traffic has tripled and users are queuing. Which parallelism strategy do you reach for, and why not tensor parallelism?
4. Your service reports a 60 ms average TTFT, yet users complain it "sometimes hangs." What single measurement would you look at, and what threshold signals a real problem?
5. You have a 40-billion-parameter model that won't fit on one GPU, and your GPUs are in *two separate machines connected by slow Ethernet* (no NVLink between machines). What combination of parallelism would you use within vs across the machines, and why?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **They measure different things.** Latency = time for *one* request; throughput = work *per second across all users*. A fleet can finish thousands/sec (great throughput) while any one request is slow, and one GPU serving one user can be fast per-request yet finish few per second. Batching raises throughput while *increasing* per-request latency — proof they aren't inverses.

2. **≈ 30 ms per token.** `TPOT ≈ (E2E − TTFT) / output_tokens = (3.0 − 0.6) / 80 = 2.4 / 80 = 0.03 s = 30 ms/token`.

3. **Data Parallelism (DP/DDP)** — the model fits, so replicate it and route different requests to each replica; throughput scales with GPU count. **Not tensor parallelism** — TP is for when the model *doesn't fit*; its per-layer collective would add communication cost without solving a memory problem you don't have.

4. **Look at tail latency — P99 (and P95).** The 60 ms *average* hides slow outliers ("averages lie"). Rule of thumb: **P99 > 3–5× P50** = a real tail problem — likely a few long-prompt prefills or KV-cache pressure stalling others.

5. **Tensor Parallelism *within* each machine + Pipeline Parallelism *across* the two.** TP splits each layer's matrices across GPUs inside one node where fast NVLink handles its per-layer collectives. PP splits the *layers* into stages across nodes, sending only activations point-to-point — lightweight enough to tolerate the slow Ethernet. In vLLM: `--tensor-parallel-size` = GPUs per node, `--pipeline-parallel-size` = 2 (nodes).

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Scaling_Jargon_Card.md)
