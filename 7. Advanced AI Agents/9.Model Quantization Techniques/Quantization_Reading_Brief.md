<a id="top"></a>
# Model Quantization — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~25 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Quantization_Jargon_Card.md`](./Quantization_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `Model_Quantization_Techniques.ipynb` in this folder (the big one). A shorter companion, `model_quantization.ipynb`, just contains the raw asymmetric-quantization NumPy demo.

---

## 🎯 30-second TL;DR

**Quantization = storing a model's numbers in fewer bits (e.g., 32-bit float → 8-bit integer) to shrink memory and speed up inference, while keeping accuracy almost intact.**

The core trick is a straight-line map between floats and integers built from two numbers — a **scale** and a **zero-point** — computed from the tensor's min and max.

The notebook proves the ideas end to end with real numbers:

- A 70-billion-parameter model needs **280 GB** in FP32 (full precision). That's why we quantize.
- On the same 3×3 test tensor, **asymmetric** quantization (MSE **1.57**) beats **symmetric** (MSE **2.51**) — a lopsided range fits better when you add a zero-point.
- **Per-channel** quantization cuts error **27.9%** below **per-tensor** on that tensor (MSE 2.51 → 1.81) for the cost of storing a few extra scales.
- Weight-only **INT8** on TinyLlama keeps output quality basically identical to FP32 — the "sweet spot" where you lose almost nothing.
- **QAT** (Quantization-Aware Training) recovers **45–70%** of the accuracy that naive 4-bit quantization throws away, with **zero** extra cost at inference time.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Why LLMs need quantization** — billions of parameters, 280 GB for a 70B model in FP32; the goal is fewer bits without losing accuracy.
2. **How numbers are stored in bits** — IEEE-754: sign + exponent + fraction; more bits = more precision and range.
3. **The data types** — FP32, FP16, BF16, INT8, INT4 — what each trades away.
4. **FP16 vs BF16 and mixed-precision training** — why BF16 is the modern default (range beats precision for gradients).
5. **Symmetric quantization (absmax)** — center on zero, no zero-point.
6. **Asymmetric quantization (zero-point)** — shift the mapping for lopsided data; a worked 3×3 example.
7. **Linear quantization in PyTorch** — quantize, dequantize, compute scale & zero-point, measure MSE.
8. **Outliers, clipping, calibration** — why one big value ruins everything; how to pick a range for dynamic values.
9. **Hands-on calibration** — hook TinyLlama's layers, collect activation min/max, compute per-layer scales.
10. **Granularity** — per-tensor vs per-channel vs per-group, with an MSE comparison.
11. **PTQ (Post-Training Quantization)** — dynamic vs static; hands-on INT8 with Optimum Quanto on TinyLlama.
12. **4-bit and beyond** — GPTQ, GGUF, AWQ, BitsAndBytes/NF4.
13. **QAT (Quantization-Aware Training)** — fake-quant, the Straight-Through Estimator, wide vs narrow minima; hands-on with Unsloth + TorchAO.
14. **Extreme quantization** — BitNet (1-bit) and BitNet 1.58b (ternary), plus how to load pre-quantized models.

---

## 🧠 The big idea — rounding a high-resolution photo to fewer colors

**Analogy: quantization is like saving a photo with fewer colors.** A full-color photo uses millions of shades (that's your FP32 model — precise but huge). Re-save it with only 256 colors (INT8) and the file gets ~4× smaller yet looks almost identical; squint and you'll spot **banding** where similar shades merged into one — that banding is **quantization error**. Push to 16 colors (INT4) and the savings grow but the banding is obvious. The notebook uses this exact image-color analogy.

The engineering question is: *given a fixed, small palette of integers, how do you choose which floats map to which integer so the picture still looks right?* The answer is a **linear (straight-line) map** defined by two numbers:

- **scale** — how many float-units one integer step is worth (the spacing of the palette).
- **zero-point** — which integer the real value 0 lands on (how the palette is shifted).

Everything else — symmetric vs asymmetric, per-tensor vs per-channel, PTQ vs QAT — is just different strategies for choosing scale and zero-point well, so the banding (error) stays small.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, and numbers straight from the notebook.

### 1. Number formats: FP32, FP16, BF16, INT8, INT4

> **🪜 Mental model:** bits are a budget split three ways — **sign** (±), **exponent** (how big), **fraction** (how precise). Different formats split the budget differently.

A floating-point number (IEEE-754) is stored as a **sign** bit, some **exponent** bits (which set the *dynamic range* — how large or small the number can be), and some **fraction/mantissa** bits (which set the *precision* — how finely-spaced neighboring values are). The bit layouts:

| Format | Sign | Exponent | Fraction | Total bits |
|--------|------|----------|----------|-----------|
| FP32 (full precision) | 1 | 8 | 23 | 32 |
| FP16 (half precision) | 1 | 5 | 10 | 16 |
| BF16 (brain float) | 1 | 8 | 7 | 16 |

**INT8** and **INT4** aren't floating point — they're plain integers (256 and 16 values). Quantization is the bridge from a float format to an integer one.

**Why it matters in this notebook.** This is the "how much can we save" table. A 70-billion-parameter model in FP32 needs **280 GB** (70e9 × 4 bytes). Drop to INT8 (1 byte each) → ~70 GB — the whole motivation.

### 2. FP16 vs BF16 (and why BF16 wins for training)

> **🪜 Mental model:** FP16 is a sharp knife with a short handle (precise but can't reach far); BF16 is a duller knife with a long handle (less precise but reaches the extremes).

Both are 16 bits, spent differently. **FP16** keeps 10 fraction bits (more precise) but only 5 exponent bits, so its max is ~65,504. During training, gradient/loss values exceed that and **overflow to infinity**; values below ~6×10⁻⁸ **underflow to zero**. To survive, FP16 needs **loss scaling** (multiply the loss before backprop, divide after). **BF16** keeps all 8 exponent bits, so it has FP32's full range (±3.4×10³⁸) and rarely overflows/underflows — at the cost of only 7 fraction bits. Training is noisy anyway (stochastic gradients, dropout), so lower precision barely hurts while avoiding overflow matters a lot.

**Where it's used in this notebook.** **Mixed precision training** does most math in 16-bit but keeps a **master copy of weights in FP32** (so tiny updates don't round to zero). With BF16 you skip loss scaling — which is why PyTorch FSDP, DeepSpeed, and HuggingFace Accelerate default to BF16 on A100/H100. The notebook's QAT run sets `bf16=True`.

### 3. Symmetric vs asymmetric quantization

> **🪜 Mental model:** symmetric centers the palette on zero (0↔0, no shift); asymmetric slides the palette so a lopsided range fits snugly (adds a zero-point shift).

**Symmetric (absmax)** maps ±|max| onto the integer range, forcing real 0 to integer 0. No zero-point (fixed at 0); scale = |max| / 127 for signed INT8. Simple and cheap, but on lopsided data (say, mostly positive) you waste half the palette on values that never occur. **Asymmetric (zero-point)** maps the true min (β) and max (α) onto the full integer range using **both** a scale and a **zero-point** shift, fitting skewed data better.

**Where it's used in this notebook.** On the same 3×3 test tensor, **asymmetric MSE = 1.57** vs **symmetric MSE = 2.51** — asymmetric wins because the range (-184 to 728.6) is lopsided. Symmetric shines when data is already centered on zero (fewer moving parts, and integer 0 stays exactly 0, which some hardware likes).

### 4. Scale, zero-point, and the quantize/dequantize roundtrip

> **🪜 Mental model:** scale = the ruler's tick spacing; zero-point = where you slide the ruler's "0" mark.

To **quantize**: divide the float by the scale, add the zero-point, round, and clamp to the integer limits. To **dequantize**: subtract the zero-point and multiply by the scale. The result is *approximate* — the gap is **quantization error**, because many different floats collapse onto the same integer.

**Worked example (the notebook's 3×3 tensor).** Range r_min = -184, r_max = 728.6, INT8 range [-128, 127]:
`scale = (728.6 − (−184)) / (127 − (−128)) = 912.6 / 255 ≈ 3.58`
`zero_point = −128 − (−184 / 3.58) ≈ −77`
Quantize the value 191.6: `round(191.6/3.58 + (−77)) = round(53.5 − 77) ≈ −23`.
Dequantize it back: `(−23 − (−77)) × 3.58 = 54 × 3.58 = 193.32`. Original was 191.6, so the error is **1.72**. (And notice: real value 0 maps to the zero-point −77 — that's the tell-tale sign of zero-point quantization.)

**Where it's used in this notebook.** This is the PyTorch section: `linear_q_with_scale_and_zero_point`, `linear_dequantization`, `get_q_scale_and_zero_point`. With *arbitrary* scale/zero-point the MSE is a terrible 170.88; with *computed* ones it drops to **1.57**. Choosing them well is the whole game.

### 5. Granularity: per-tensor vs per-channel vs per-group

> **🪜 Mental model:** one palette for the whole painting (per-tensor) vs a custom palette per stripe (per-channel) vs per tiny patch (per-group). Finer palettes = truer colors, more palettes to carry.

**Per-tensor** uses one scale for the whole tensor — simplest, but a single loud channel forces a coarse scale that crushes the quiet channels. **Per-channel** gives each row its own scale, so a high-magnitude "edge-detector" channel and a low-magnitude "fur-texture" channel each get a fitting mapping. **Per-group** goes finer: one scale per small block within a row.

**Where it's used in this notebook.** On the 3×3 tensor, per-channel drops MSE from **2.51 → 1.81 (a 27.9% reduction)**. On a 6×6 tensor with group_size=3, per-group stores **12 scales** vs per-tensor's **1**. **Rule of thumb:** start with weights-only per-tensor; if accuracy holds, stop; if it drops too much, go per-channel on layers whose channels vary a lot.

### 6. PTQ vs QAT (the two calibration philosophies)

> **🪜 Mental model:** PTQ = "quantize the finished cake and hope it still tastes good." QAT = "bake the cake already knowing it'll be squished, so it holds shape."

**PTQ (Post-Training Quantization)** quantizes an already-trained model, no real retraining. It comes in two flavors: **dynamic** (compute activation scale/zero-point on the fly at runtime — more accurate, slower) and **static** (pre-compute them once with a **calibration** dataset — faster, less accurate). Cheap and fast, but accuracy can drop.

**QAT (Quantization-Aware Training)** inserts **fake-quant** modules (quantize-then-immediately-dequantize during the forward pass) so the model *feels* rounding error while training and learns to compensate. Because rounding has no gradient, the **Straight-Through Estimator (STE)** lets gradients pass through as if the fake-quant were an identity function. Conceptually, QAT steers training toward **wide minima** in the loss landscape, where rounding weights barely changes the loss.

**Where it's used in this notebook.** PTQ hands-on quantizes TinyLlama to INT8 with Optimum Quanto (`quantize()` then `freeze()`), near-identical output. QAT hands-on fine-tunes Qwen3-4B with Unsloth + TorchAO (`qat_scheme="int4"`). Payoff: QAT recovers **45–70%** of naive-4-bit's lost accuracy, gives **+1–3%** on GPQA/MMLU-Pro, and adds **zero** inference overhead — the only cost is 1–5 epochs. **Choose QAT** when PTQ drops accuracy by >2–3%, when data has extreme outliers, or for safety-critical use.

---

## 🔥 The headline experiments — at a glance

The notebook runs several small experiments. Here are the ones with real numbers to remember.

**A. Symmetric vs asymmetric vs granularity (on the 3×3 test tensor, INT8):**

| Method | MSE (lower = better) | Note |
|---|---|---|
| Arbitrary scale/zero-point | **170.88** | proves you *must* compute them properly |
| Symmetric (absmax) | **2.51** | zero↔zero, no zero-point |
| Asymmetric (zero-point) | **1.57** | fits the lopsided range better |
| Per-channel (symmetric) | **1.81** | **27.9% lower** than per-tensor symmetric |

**B. FP32 vs INT8 weight-only on TinyLlama-1.1B (Optimum Quanto):**

| Metric | FP32 | INT8 |
|---|---|---|
| Model size | 4196.4 MB | 4196.4 MB* |
| Latency | 1.50 s | 2.65 s |
| Output quality | baseline | ~identical |

\* Theory says INT8 should be ~4× smaller; the demo's *reported* size and GPU latency didn't improve (weight-only INT8 on GPU adds dequant overhead — real wins are memory and CPU speed). **Takeaway: weight-only INT8 barely changes output quality — that's the sweet spot.**

**C. QAT vs naive PTQ at 4-bit (from the notebook's summary table):**

| Aspect | Naive PTQ (4-bit) | QAT (4-bit) |
|---|---|---|
| Accuracy recovery | — | recovers **45–70%** of lost accuracy |
| Benchmark gain | baseline | **+1–3%** (GPQA, MMLU-Pro) |
| Inference overhead | none | **none** (identical at runtime) |
| Cost | none | 1–5 epochs of fine-tuning |

---

## 🧮 Formulas to memorise

### 1. Scale (asymmetric)

```
scale = (r_max − r_min) / (q_max − q_min)
```

**In words:** scale = (real max − real min) ÷ (integer max − integer min) — "how much real-number range each integer step must cover."
Symbols: `r_max`/`r_min` = largest/smallest float; `q_max`/`q_min` = integer limits (127 / −128 for signed INT8).
**Worked example:** `(728.6 − (−184)) / (127 − (−128)) = 912.6 / 255 ≈ 3.58`.

### 2. Zero-point (asymmetric)

```
zero_point = q_min − (r_min / scale)
```

**In words:** the zero-point = the integer minimum, minus (the real minimum divided by the scale). It's the integer that the real value 0 maps to, shifting the palette to fit a lopsided range. Round it and clamp it into [q_min, q_max].
**Worked example:** `−128 − (−184 / 3.58) = −128 + 51.4 ≈ −77`.

### 3. Quantize (float → integer)

```
q = clamp( round(x / scale + zero_point), q_min, q_max )
```

**In words:** divide the float by the scale, add the zero-point, round to the nearest integer, then clamp to the allowed integer range so nothing overflows.
**Worked example:** for x = 191.6: `round(191.6/3.58 + (−77)) = round(53.5 − 77) = round(−23.5) ≈ −23`.

### 4. Dequantize (integer → float)

```
r ≈ scale × (q − zero_point)
```

**In words:** subtract the zero-point from the stored integer, then multiply by the scale to recover an approximate float. The "≈" is where quantization error lives.
**Worked example:** for q = −23: `3.58 × (−23 − (−77)) = 3.58 × 54 = 193.32`. Original was 191.6 → error 1.72.

### 5. Symmetric scale (absmax) — the simpler special case

```
scale = |max(x)| / q_max          (zero_point = 0)
```

**In words:** take the biggest absolute value in the tensor and divide by the integer maximum (127 for INT8). There's no zero-point because real 0 already maps to integer 0.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–8** | Why quantize (280 GB for 70B), IEEE-754 bits, FP16/BF16/INT8 types, mixed precision | **Read** — ~8 min. Absorb the bit-layout table. |
| **9–13** | Symmetric vs asymmetric + the worked 3×3 example (scale 3.58, zero-point −77) | **FOCUS** — ~10 min. The conceptual core; follow the arithmetic by hand. |
| **14–32** | PyTorch from scratch: quantize/dequantize, computing scale & zero-point, MSE (170.88 → 1.57 → symmetric 2.51) | **Read carefully** — ~12 min. Watch good params crush the error. |
| **33–45** | Outliers, clipping, calibration; hands-on hooking TinyLlama for per-layer activation min/max | **Read** — ~8 min. Why activations (dynamic) need calibration but weights (static) don't. |
| **46–54** | Granularity: per-tensor vs per-channel (27.9% drop) vs per-group (12 scales) | **FOCUS** — ~8 min. The MSE comparison is the payoff. |
| **55–68** | PTQ dynamic vs static; hands-on INT8 on TinyLlama with Optimum Quanto (quantize → freeze) | **Read carefully** — ~10 min. Note the honest latency result and the "sweet spot." |
| **69–87** | 4-bit realm (GPTQ, GGUF), QAT (fake-quant, STE, wide minima), Unsloth+TorchAO, BitNet 1-bit / 1.58b | **Skim-to-read** — ~12 min. Get QAT vs PTQ; BitNet = the extreme end. |
| **88–97** | Loading pre-quantized models (GPTQ, GGUF, AWQ, BitsAndBytes) + summary table | **Reference** — ~4 min. Loading snippets; the summary table is a great recap. |

**Total notebook read time:** ~70 min. Add this brief's ~25 min and you're at **~95 min**, all with full comprehension.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why quantization exists** — a 70B model is 280 GB in FP32; fewer bits = less memory, cheaper inference.
- [ ] **The three-way bit budget** — sign + exponent (range) + fraction (precision) — and why **BF16 beats FP16 for training** (range beats precision on gradients; no loss scaling).
- [ ] **Symmetric vs asymmetric** — symmetric forces 0↔0 (no zero-point); asymmetric adds a zero-point for lopsided data (MSE 1.57 vs 2.51).
- [ ] **Compute scale and zero-point** from min/max, and run the quantize→dequantize roundtrip to get an error (MSE).
- [ ] **Granularity trade-off** — per-tensor (1 scale, coarse) → per-channel (27.9% less error) → per-group (12 scales, finest).
- [ ] **PTQ vs QAT** — PTQ quantizes after training (needs calibration for activations); QAT fine-tunes with fake-quant + STE to recover 45–70% of lost accuracy at zero inference cost.
- [ ] **Why activations need calibration but weights don't** — weights are static (known); activations are dynamic (depend on input).

If any of these feel shaky, come back to the matching primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. A tensor's real values range from −10 to +30, and you want to quantize to signed INT8 (range −128 to 127). What is the **scale**?
2. Using that scale, what is the **zero-point**?
3. Both FP16 and BF16 use 16 bits. Why is BF16 preferred for training LLMs, despite being *less* precise per value?
4. On the notebook's 3×3 tensor, asymmetric quantization gave MSE 1.57 and symmetric gave 2.51. Why did asymmetric win *here*?
5. In one sentence: what does QAT do that plain PTQ does not, and what does it cost?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **scale = (30 − (−10)) / (127 − (−128)) = 40 / 255 ≈ 0.1569.** (Real range over integer range.)
2. **zero_point = q_min − r_min/scale = −128 − (−10 / 0.1569) = −128 + 63.7 ≈ −64** (rounded, then clamped into [−128, 127]).
3. **BF16 keeps FP32's full dynamic range** (8 exponent bits, max ~3.4×10³⁸), so gradients don't overflow/underflow during backprop and it needs no loss scaling. FP16's tiny max (~65,504) makes training fragile. Training is noisy anyway, so BF16's lower precision barely matters; not blowing up matters more.
4. **The range is lopsided** (−184 to 728.6). Asymmetric slides the palette with a **zero-point** to fit that skew using the full integer range; symmetric forces the mapping to be centered on zero, wasting part of the palette → coarser steps → higher error.
5. **QAT fine-tunes with fake-quant so the model learns to tolerate rounding** (recovering 45–70% of naive-4-bit's lost accuracy, zero inference overhead), costing 1–5 epochs — whereas PTQ just quantizes the finished model and accepts the accuracy drop.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Quantization_Jargon_Card.md)
