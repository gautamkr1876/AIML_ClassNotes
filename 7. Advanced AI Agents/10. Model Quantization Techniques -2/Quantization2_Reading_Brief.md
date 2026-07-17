<a id="top"></a>
# Model Quantization — Part 2 · Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming, not learning blind.
>
> **Side reference:** keep [`Quantization2_Jargon_Card.md`](./Quantization2_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `v3_Class_Ref_Model_Quantization.ipynb` (the class reference; `Live_Class_Notebook_quantization_2.ipynb` is the shorter live version — same experiments).
> **Prerequisite:** this is the **second** quantization lecture. Part 1 (folder **`9.Model Quantization Techniques`**) covers the first-principles basics — what a bit is, why fp32 is 4 bytes, the intuition for reducing precision. This brief assumes those.

---

## 🎯 30-second TL;DR

One model, one dataset, one scorecard — **shrink a 45 MB model onto a 12 MB budget without losing accuracy.**

The case study is **Cairn**, a BERT intent classifier that must run on a 4 GB retail kiosk. Its fp32 baseline is **44.9 MB @ 93.5% accuracy** — nearly 4× over the ≤12 MB budget. The notebook closes that gap step by step and lands a real **~11 MB int8 file with accuracy intact (PASS)**.

The four numbers to carry out of the notebook:

- **fp16/bf16 casting is free**: 44.9 MB → **22.4 MB, 0% accuracy lost**. Always your first move.
- **Granularity beats bits**: int4 per-tensor = **75.3%**; int4 per-**group** = **93.5%** — *same 4 bits*, +18 points, for ~6% extra storage.
- **PTQ runs out of road below int4**: the bit ladder shows int3 leaks and int2 collapses.
- **QAT revives the dead**: int2 goes **8.2% → 89.6%** by training *with* the quantizer in the loop.

And the lesson tutorials skip: **quantization's accuracy is portable; its speed is not.** On the class's Mac, int8 was *slower* (0.50×) than fp32 — measure speed on the real target.

> Note: GPTQ, AWQ, bitsandbytes/NF4, QLoRA, GGUF, MXFP4 are all **named and explained** in this lecture (Parts 4–8 + the "Part 9" roadmap) as where these techniques scale up to LLMs — but the hands-on LLM run is a *roadmap pointer*, not code in this notebook. The hands-on model here is the little BERT.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **P0 Setup** — download CLINC150 + bert-mini; define the three measuring instruments (`evaluate`, `latency_ms`, `size_mb`).
2. **P1 Baseline** — fine-tune Cairn in fp32; record 44.9 MB / 93.5% / 1.36 ms; **census where the bytes live** (70% is the embedding table).
3. **P2 Inside a number** — sign / exponent / mantissa; cast to fp16 & bf16 (free 2× shrink); why int8 has *no exponent*.
4. **P3 The int8 map** — the affine map (quantize/dequantize); **symmetric vs asymmetric**; weights→symmetric, activations→asymmetric.
5. **P4 Granularity** — per-tensor vs per-channel vs per-group; the int4 rescue (75%→93%) for pennies of storage.
6. **P5 Calibration** — choosing activation ranges from real traffic; 128–512 samples is enough; min-max vs percentile vs MSE.
7. **P6 PTQ end-to-end** — assemble the recipe; run **real backends** (`quantize_dynamic`, then **torchao**); pack the ~11 MB file; measure latency (the int8 surprise).
8. **P7 The bit ladder** — sweep bits 8→2 × all granularities; find where PTQ breaks (int3 leaks, int2 collapses).
9. **P8 QAT** — put the quantizer in the training loop via the Straight-Through Estimator; revive int2 (8.2%→89.6%).
10. **P9 (roadmap)** — the LLM world: TinyLlama + GGUF / GPTQ / AWQ / bitsandbytes / FP8 / MXFP4 — described, not run.

---

## 🧠 The big idea — one analogy

**Quantization is choosing a good map from a handful of integer codes onto the real-number line.**

Analogy — **a hotel with a fixed number of room numbers.** A real number can be *anywhere* on an infinite corridor. int8 gives you only **256 room numbers**; int4 gives you **16**; int2 gives you **4**. Quantization is the job of deciding *which stretch of corridor those rooms cover* and *how far apart they sit* — so that the guests who actually show up (the values your model really uses) each get a room close to where they wanted to stand.

Every technique in the lecture is just a smarter way to place those rooms:

- **scale** = how far apart the rooms are. **zero_point** = which room sits at address 0.
- **symmetric vs asymmetric** = do you centre the rooms on zero, or shift them to cover a lopsided crowd?
- **granularity** = do you use *one* room map for the whole building, or a fresh map per floor (per-channel) / per-corridor (per-group)?
- **calibration** = watching real guests arrive so you know which stretch to cover.
- **PTQ vs QAT** = place rooms after the guests have booked (PTQ), or *ask the guests to stand where the rooms already are* by training them to (QAT).

The map that fits the data keeps accuracy; the map that wastes rooms on empty corridor loses it.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook. Each = a mental model + plain-English what + the notebook's real numbers.

### 1. Census first — where do the bytes live?

> **🪜 Mental model:** before dieting, weigh yourself part by part — you can't cut fat you didn't find.

Cairn has 11.2M parameters. Before touching any tool, the notebook runs a one-loop **census** and finds the split: **linears 29%, embeddings 71%, other 0.1%.** That single fact steers everything. A tool that "quantizes the Linear layers" (the common default) leaves **70% of Cairn untouched** — which is exactly what bites `quantize_dynamic` later (it shrinks the model to only 35.1 MB because the embedding table stays fp32). **Best practice #1: census your model before quantizing anything.** (LLMs are the mirror image — their giant Linear stacks dwarf embeddings, which is why *weight-only* quantization rules the LLM world.)

### 2. The number itself — sign / exponent / mantissa, and why int8 is different

> **🪜 Mental model:** exponent picks the *neighbourhood* (thousands? millionths?); mantissa is the *street address* inside it.

A float has three parts: **sign** (±), **exponent** (how big), **mantissa** (which value exactly). Cast fp32→**fp16/bf16** and you halve the bytes: 44.9 MB → **22.4 MB, accuracy unchanged (93.5%)** — the free reflex first move. But 22 MB still fails the 12 MB budget, and here the game changes: **int8 has *no exponent field*.** It's just 256 evenly-spaced codes. The "how big" information that the exponent used to carry must now be supplied from *outside* the tensor, as a shared **scale**. That external scale *is* quantization. (bf16 vs fp16: same 2 bytes; bf16 keeps fp32's range but fewer mantissa bits; fp16 keeps more precision but less range.)

### 3. The int8 map — symmetric vs asymmetric

> **🪜 Mental model:** symmetric = rooms centred on zero; asymmetric = rooms shifted to hug a lopsided crowd.

The affine map is one line: `q = clamp(round(x / scale) + zero_point)`, and back with `x_hat = (q − zero_point) * scale`. Two knobs: **scale** (step size) and **zero_point** (which code is real 0.0). Two ways to set them:

- **Symmetric** — force `zero_point = 0`, cover `[−a, +a]`. Best for **weights** (naturally centred on zero) and faster in the int-matmul. But on one-sided data it wastes ~half its codes.
- **Asymmetric** — a computed `zero_point`, cover any `[r_min, r_max]`. Best for **activations** (GELU/ReLU outputs are one-sided).

The notebook proves int8 weights are painless: per-tensor symmetric int8 keeps **93.5%** (no loss). Then it tries the same loop at int4… and accuracy craters to **75.3%**. The culprit isn't "4 bits is too few" — it's **one scale trying to serve a whole tensor.**

### 4. Granularity — the highest-leverage dial in quantization

> **🪜 Mental model:** one loud singer forces the whole choir to sing quietly. Give each section its own volume knob.

A scale must stretch to cover the **largest** value in its territory — so one loud channel forces every quiet channel into 1–2 usable codes. The fix is more scales, at no extra bits:

- **Per-tensor** — 1 scale for everything. int4: **75.3%**.
- **Per-channel** — 1 scale per output row. int4: **93.2%**, for <1.5% extra storage.
- **Per-group** — 1 scale per block of 64 weights. int4: **93.5%**, for ~6% extra storage.

**Same 4 bits, ~18 points apart.** This is the single most important practical fact: *when low-bit quantization fails, reach for finer granularity before reaching for more bits.* It's why int8 stacks default to per-channel, and why every serious 4-bit LLM format (GPTQ, AWQ, GGUF K-quants, bitsandbytes NF4, MXFP4) is **group-wise**.

### 5. Calibration — picking activation ranges

> **🪜 Mental model:** activations don't exist until traffic flows — so you have to watch a little real traffic and write down each layer's lo/hi.

Weights sit in a file (read their min/max directly). **Activations** only appear at runtime, so you run **calibration**: pass a sample of real utterances through and record each layer's range. The notebook's measured findings: **128–512 real samples is plenty** (GPTQ/AWQ default to exactly this) — calibration is *minutes*, not a training job. Range-picking menu: **min-max** (cover everything; safe at 8 bits), **percentile** (clip rare outliers; cheap insurance ≤6 bits), **MSE search** (least round-trip error; GPTQ/AWQ use it), **entropy/KL** (TensorRT's classic). A caution: bert-mini shrugs off even *garbage* calibration because **LayerNorm** keeps its ranges in a narrow band — this does **not** hold for big LLMs, where a few outlier dimensions blow up (the LLM.int8 discovery) and calibration becomes make-or-break.

### 6. PTQ vs QAT — round what you have, or train to be rounded

> **🪜 Mental model:** PTQ places the rooms after guests have booked; QAT trains the guests to stand where the rooms already are.

**PTQ (Post-Training Quantization)** takes a finished model and just chooses maps — no gradients, minutes of work. Everything through Part 6 is PTQ, and it delivers the ~11 MB ship file. But the **bit ladder** (Part 7) shows its limit: int8 free, int4 fine *with granularity*, int3 leaks, **int2 collapses** to near-random. The fix is **QAT (Quantization-Aware Training)**: put the quantizer *inside* the training loop so gradient descent learns weights that survive rounding. The obstacle is that `round()` has zero slope, so gradients die — solved by the **Straight-Through Estimator (STE)**: in the backward pass, pretend the staircase is the identity and pass gradients straight through (`w + (qdq(w) − w).detach()`). Result: int2 goes from **8.2% → 89.6%**, versus an equally-trained fp32 at 93.4% — an honest cost of ~3.8% for **four codes per weight**.

---

## 🔥 The headline experiments — at a glance

**A. The scorecard (real numbers from the notebook, val set):**

| Config | Size | Accuracy | Δ vs fp32 | ≤12 MB? |
|---|---|---|---|---|
| fp32 baseline | 44.9 MB | 93.5% | +0.0% | fail |
| fp16 weights | 22.4 MB | 93.5% | +0.0% | fail |
| bf16 weights | 22.4 MB | 93.5% | +0.0% | fail |
| torch `quantize_dynamic` (int8) | **35.1 MB** | 93.4% | −0.1% | fail ← left embeddings fp32! |
| **torchao int8 weight-only** | **~11 MB** | 93.6% | +0.1% | **PASS** |

**B. Granularity rescues int4 (same 4 bits):**

| Granularity | int4 accuracy | Extra storage |
|---|---|---|
| per-tensor | 75.3% | 0% |
| per-channel | 93.2% | <1.5% |
| per-group (g=64) | 93.5% | ~6% |

**C. The bit ladder — where PTQ breaks:**

| Bits | per-tensor | per-channel | per-group64 |
|---|---|---|---|
| int8 | 93.5% | 93.5% | 93.5% |
| int6 | 92.8% | 93.6% | 93.4% |
| int4 | 75.3% | 93.2% | 93.5% |
| int3 | 35.2% | 91.7% | 92.4% |
| int2 | 0.7% | 5.9% | 17.0% |

**D. QAT revives int2:** PTQ int2 = **8.2%** → QAT int2 = **89.6%** (fp32, equally trained = 93.4%).

**E. Latency surprise (Mac, batch 1, 4 threads):** fp32 = 1.36 ms, int8 dynamic = 2.72 ms → **0.50× — int8 was *slower*.** On kiosk-class ARM / x86 VNNI / NVIDIA int8 is 2–4× *faster*. **Speed is a kernel story, not physics — benchmark on the target.**

---

## 🧮 Formulas to memorise

### 1. Quantize / dequantize (the affine map)

```
quantize:    q = clamp( round(x / scale) + zero_point , q_min, q_max )
dequantize:  x_hat = (q - zero_point) * scale
```

**In words:** *to quantize*, divide the real value by the step size, round to the nearest integer, add the zero-code offset, and clip into the code range. *To dequantize*, subtract the offset and multiply back by the step size.
- `scale` = step size (real units per integer step); `zero_point` = the code that means 0.0.
- **Symmetric weights**: `scale = max(|x|) / q_max`, `zero_point = 0`, `q_max = 2^(bits−1) − 1` (int8 → 127).
- **Asymmetric activations**: `scale = (r_max − r_min) / q_max`, `zero_point = round(−r_min / scale)`, `q_max = 2^bits − 1` (int8 → 255).

**Worked example (from the notebook's toy tensor):** for `x = 2.99` with symmetric int8, `scale = 2.99/127 ≈ 0.02354`, so `q = round(2.99/0.02354) = 127`, and `x_hat = 127 × 0.02354 ≈ 2.99`. Round-trip nearly exact — because 8 bits has 256 codes to spare.

### 2. Model size = params × bytes-per-param

```
size = num_params × bytes_per_param
```

**In words:** total bytes equals the number of parameters times how many bytes each one takes.
- fp32 = 4 bytes, fp16/bf16 = 2 bytes, int8 = 1 byte, int4 = 0.5 byte (two codes per byte via **nibble packing**).

**Worked example (Cairn):** `11.2M × 4 bytes = 44.8 MB` (fp32). At int8: `11.2M × 1 = ~11 MB` — which is exactly why the torchao int8 file lands ~11 MB and **passes** the 12 MB budget. (Plus a tiny overhead for the per-channel scales — the notebook measured per-channel scales at only 19.2 KB.)

### 3. The STE idiom (why QAT can train through rounding)

```
w_fq = w + (qdq(w) - w).detach()
#  forward:  w + qdq(w) - w  = qdq(w)   (model computes with rounded weights)
#  backward: d/dw = 1 + 0    = 1        (.detach() hides the flat staircase)
```

**In words:** the forward pass uses the quantized-dequantized weight (so the model *feels* the rounding); the `.detach()` on the correction term makes the backward pass see slope 1, as if rounding were the identity — so gradients flow instead of dying on `round()`'s zero slope.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells (Class Ref) | What it teaches | How to read |
|---|---|---|
| **1–20** (P0–P1) | Case study, download, instruments, fp32 baseline, **byte census** | **Read** — ~8 min. Memorise 44.9 MB / 93.5% and "70% is embeddings." |
| **21–23** (P2) | sign/exponent/mantissa, fp16/bf16 cast, "int8 has no exponent" | **Read carefully** — ~6 min. The conceptual core of the day. |
| **24–31** (P3) | affine map, symmetric vs asymmetric, int8 free, int4 collapse | **FOCUS** — ~10 min. Watch the int4 crash to 75.3%. |
| **32–38** (P4) | granularity ladder; the int4 rescue table | **FOCUS — highest-leverage** — ~10 min. |
| **39–41** (P5) | calibration, sample counts, garbage-calibration surprise | **Read** — ~6 min. Note "LayerNorm forgives; LLMs don't." |
| **42–56** (P6) | PTQ recipe, `quantize_dynamic` gotcha, torchao ~11 MB, latency | **FOCUS** — ~12 min. The ship candidate + the speed lesson. |
| **57–59** (P7) | the bit ladder 8→2 | **Skim the table** — ~4 min. |
| **60–68** (P8 + memo) | QAT, STE, int2 revival, shipping memo | **Read** — ~8 min. |

**Total notebook read:** ~65 min. Add this brief (~22 min) ≈ 87 min, comfortably under a cold read of the ~9,000-word notebook.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why you census a model first** — Cairn is 70% embedding table, so Linear-only tools barely help (they leave it at 35.1 MB).
- [ ] **Why fp16/bf16 is the free first move** — half the bytes (22.4 MB), zero accuracy lost.
- [ ] **Why int8 needs a scale** — it has no exponent field, so "how big" must come from an external scale.
- [ ] **Weights→symmetric, activations→asymmetric** — and why (weights are zero-centred; activations are one-sided).
- [ ] **Granularity beats bits** — int4 goes 75%→93% by adding per-channel/per-group scales for pennies.
- [ ] **Where PTQ breaks and QAT takes over** — int8 free, int4 with care, int3 leaks, int2 needs QAT (8.2%→89.6% via STE).
- [ ] **Accuracy is portable, speed is not** — int8 was 0.50× on the Mac; benchmark on the deployment target.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Cairn is 11.2M parameters. What's its fp32 size, and roughly its int8 size — and which budget (≤12 MB) does each meet?
2. Same 4-bit weights, but int4 per-tensor gets 75.3% while int4 per-group gets 93.5%. In one sentence, *why*?
3. The notebook says weights use symmetric quantization and activations use asymmetric. Give the reason for each.
4. `quantize_dynamic` produced a 35.1 MB file even though it "quantized to int8." What went wrong, and what best practice would have caught it?
5. PTQ int2 scored 8.2%; QAT int2 scored 89.6%. What single trick makes QAT able to train through the `round()` operation?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **fp32 = 11.2M × 4 bytes = 44.9 MB (fails ≤12 MB).** int8 = 11.2M × 1 byte ≈ **11 MB (passes)** — which is exactly the torchao result. Size = params × bytes-per-param.
2. **Because per-tensor forces one scale on the whole matrix**, so one loud channel makes the shared step size so coarse that quiet channels get 1–2 usable codes; per-group gives each block of 64 weights its own scale, so no channel is starved. Same bits, finer map.
3. **Weights → symmetric** because they're naturally centred on zero (so `zero_point = 0` wastes no codes and removes cross-terms from the int-matmul → faster kernels). **Activations → asymmetric** because GELU/ReLU outputs are one-sided/skewed, so a non-zero zero_point is needed to cover the used range without wasting half the codes.
4. **It only quantizes `nn.Linear` layers — and 70% of Cairn is the embedding table, which it left in fp32**, so the file barely shrank. The best practice that catches it: **census the model first** (and audit what a tool actually touched — print dtypes) before trusting its savings.
5. **The Straight-Through Estimator (STE):** in the backward pass, pretend `round()` is the identity function (slope 1) and pass gradients straight through, via `w + (qdq(w) − w).detach()`. Without it, `round()`'s zero slope kills all gradients and QAT can't learn.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Quantization2_Jargon_Card.md)
