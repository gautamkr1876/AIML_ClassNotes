<a id="top"></a>
# Model Quantization — Part 2 · Jargon Card

> **Use this like a dictionary.** Skim once (~5 min), then keep it open in a side tab while reading the notebook — look up unknown words here in 20 seconds instead of Googling.
>
> **Companion:** read [`Quantization2_Reading_Brief.md`](./Quantization2_Reading_Brief.md) FIRST for the big picture and numbers. **Prerequisite:** this is the *second* quantization lecture — the first-principles intro (what a bit is, why fp32 is 4 bytes) lives in sibling folder **`9.Model Quantization Techniques`** (Part 1).

---

## A

**Activation** — The numbers that *flow between layers* while the model runs (one layer's output = the next layer's input). Unlike weights, activations don't exist until real traffic flows, so you can't read their range off a file — you watch them (**calibration**). Quantized **asymmetrically** here because GELU/ReLU outputs are one-sided.

**Affine map** — The formula quantization is built on: `q = round(x / scale) + zero_point` ("a straight-line stretch plus a shift"). Force the shift to 0 and it becomes **symmetric**. Everything in Part 3 is this map.

**AMX (Apple Matrix eXtension)** — Dedicated fp32 matrix-multiply hardware in Apple Silicon CPUs. It's why int8 was *slower* than fp32 on the class's Mac (0.50×): the fp32 path had special silicon, the int8 path (`qnnpack`) didn't.

**Asymmetric quantization** — A quantization map that can cover any interval `[r_min, r_max]`, using a non-zero `zero_point` to mark where real 0.0 lands. Best for one-sided data like activations (GELU output). Contrast with **symmetric** (below), which assumes data is centred on zero and wastes half its codes on skewed data.

**AWQ (Activation-aware Weight Quantization)** — A 4-bit LLM method that scales weights by how important each channel is *to the activations*, so the values that matter most get the most careful codes. Notebook mentions it as **group-wise** and MSE-calibrated. Sibling to **GPTQ** — the "int4-with-care" LLM toolkit. Roadmap only.

## B

**BERT (Bidirectional Encoder Representations from Transformers)** — A transformer that reads a whole sentence at once (both directions). The notebook uses **bert-mini** (4 layers, hidden 256, ~11M params) as Cairn's classifier — small on purpose so it trains in seconds. A pre-trained **backbone** (body) plus a fresh 150-way **head** (task layer).

**bf16 (brain float 16)** — A 16-bit float that keeps fp32's full exponent range but throws away mantissa bits. Same size as fp16 (2 bytes) but trades precision for range. In the notebook both fp16 and bf16 halve the file to 22.4 MB with **zero** accuracy loss — the "free reflex first move."

**bitsandbytes (bnb)** — A library for loading LLMs in 4-bit or 8-bit (home of NF4 and QLoRA). Named in the roadmap as a group-wise 4-bit LLM format. Not run here.

**Bit ladder** — The notebook's experiment (Part 7) sweeping bit-width from 8 down to 2 at each granularity, to find where post-training quantization breaks: int8 free, int4 needs granularity, int3 leaks, int2 collapses.

## C

**Cairn** — The notebook's running case study: a BERT intent classifier that must be squeezed from 45 MB to ≤12 MB to fit on a 4 GB retail kiosk, without dropping below 84% accuracy.

**Calibration** — Running a little real traffic through the model to *observe* activation ranges (their lo/hi per layer), so you can pick a good quantization map for them. The notebook's finding: **128–512 real samples** is plenty; more doesn't help. This is what makes **static PTQ** possible.

**CLINC150** — The real dataset: 22,500 crowd-sourced utterances across 150 intents (15,000 train / 3,000 val / 4,500 test). Exactly Cairn's job. **Clamp/clip** — the map forces any out-of-range value to the nearest code edge; calibration is choosing *where* to clip.

## D

**Dequantize** — Turning integer codes back into approximate real numbers: `x_hat = (q − zero_point) * scale`. The reverse of quantize. "Simulated" quantization does quantize→dequantize so accuracy can be measured while the file stays fp32.

**Dynamic quantization** — Quantizing activations *on the fly at runtime* (min/max per batch) — no calibration. PyTorch's `quantize_dynamic` does this, but it only touches Linear layers, so it left Cairn's **embedding table in fp32** → file still 35.1 MB. A cautionary tale.

## E

**Embedding table** — A lookup table mapping each vocabulary token to a vector. In Cairn it's **~70% of the whole model** — so a Linear-only tool barely helps. Best practice #1: census before quantizing.

**Exponent** — The float part that says "how big?" — it picks the *neighbourhood* (thousands vs millionths). int8 has **no exponent field**, so that "how big" info must be re-supplied externally as the **scale**. The central "why int8 is different" idea.

## F

**Fake quantization** — Computing with quantized-then-dequantized weights during the forward pass so the model *experiences* the rounding, while the tensor stays float so gradients can flow. The engine of **QAT**. Also called "quantize→dequantize" or "simulated" quantization.

**fp16 (half-precision float)** — A 16-bit float (1 sign, 5 exponent, 10 mantissa bits). Halves the size vs fp32 with essentially no accuracy loss in the notebook (22.4 MB @ 93.5%). Compared with **bf16**: same size, fp16 keeps more precision but less range.

**fp32 (single-precision float)** — The default 32-bit float, 4 bytes per number. Cairn's baseline: 11.2M params × 4 bytes = **44.9 MB @ 93.5%** — nearly 4× over the 12 MB budget. The number everything else is measured against.

**FP8** — An 8-bit *float* (keeps a tiny exponent, unlike int8). Roadmap mention (DeepSeek trains in FP8).

## G

**GELU (Gaussian Error Linear Unit)** — A smooth transformer activation. Its output is **one-sided/skewed** — exactly why activations are quantized asymmetrically (the zero-point recovers the wasted half of the map).

**GGUF** — llama.cpp's file format for quantized LLMs; its "K-quants" (`Q4_K_M`) are group-wise 4-bit with **nibble packing**. Roadmap only.

**GPTQ** — A 4-bit LLM post-training method that re-rounds weights layer by layer to minimise error; **group-wise**, MSE-calibrated, 128–512 samples. Sibling to **AWQ**. Roadmap only.

**Granularity** — *How many scales you buy* for one weight tensor. Per-tensor = 1 scale for everything; per-channel = 1 per output row; per-group = 1 per block of k weights. The single highest-leverage dial: at int4 it moves accuracy from 75% to 93% for pennies of storage.

**Group / group-wise (block-wise)** — Cutting each weight row into fixed-size blocks (e.g., g=64) and giving each block its own scale. The workhorse of 4-bit LLMs (GPTQ, AWQ, GGUF, NF4 all do this). In the notebook, per-group-64 costs ~6% extra storage but recovers full int4 accuracy.

## I

**int2 / int3 / int4 / int8** — Integer formats with 4, 8, 16, or 256 evenly-spaced codes respectively. Fewer bits = fewer codes = coarser steps. The notebook's ladder: int8 free, int4 fine *with granularity*, int3 leaks, int2 collapses (unless you use QAT).

**Intent classification** — The task: map a user's typed sentence to one of a fixed list of intents (returns, store hours, …). Cairn does 150-way intent classification.

## L

**LayerNorm (Layer Normalization)** — A step that rescales each layer's activations into a narrow, input-independent band. The notebook's surprise: because bert-mini is LayerNorm-heavy, it shrugs off even *garbage* calibration data — a property that does **not** carry to big LLMs.

**Latency** — Time per prediction (here, ms per utterance, batch 1, 4 threads — how a kiosk sees traffic). Key lesson: int8 was *0.50× the speed* of fp32 on the Mac. Quantization's accuracy is portable; its **speed is not** — benchmark on the target.

**LLM.int8() / outlier activations** — The 2022 discovery that big LLMs grow a few **massive outlier activations** (hundreds× the norm) that break naive int8 — spawning SmoothQuant (migrates outliers into weights) and AWQ. Why LLM quantization is harder than Cairn's tidy little BERT.

## M

**Mantissa (significand)** — The part of a float that says "which value exactly?" — the *address inside* the neighbourhood the exponent picked. Fewer mantissa bits = nearby values collapse onto each other (precision loss). fp16 keeps 10 mantissa bits; bf16 keeps 7.

**min-max calibration** — Pick the quantization range as the observed minimum and maximum — cover everything, clip nothing. The safe default at 8 bits; the notebook's recipe uses it (switching to percentile below 6 bits).

**MSE search calibration** — Try many clip points, keep the one with the smallest round-trip error. Used by GPTQ/AWQ to squeeze the last accuracy at low bits.

**MXFP4 (Microscaling FP4)** — A 4-bit float that attaches a **shared 8-bit exponent to every block of 32 numbers** (exponent idea meets grouping). What GPT-OSS was QAT'd into. Roadmap only.

## N

**NF4 (4-bit NormalFloat)** — A 4-bit format whose 16 codes are spaced to match a **bell-curve** of weights, not evenly like int4. Core to bitsandbytes/QLoRA. Roadmap only. *(Don't confuse with int4: int4 = 16 evenly-spaced codes; NF4 = 16 bell-curve-spaced codes.)*

**Nibble packing** — Storing two 4-bit codes in one byte (a "nibble" = half a byte). How int4 files actually get small on disk; every 4-bit GGUF/GPTQ file does this. The notebook *simulates* int4 but notes real files pack this way.

## P

**Per-channel (per-axis / per-row)** — One scale per output channel (row) of a weight matrix. The int8 production default; rescues int4 to 93.2% for <1.5% extra storage.

**Per-tensor (per-layer)** — One scale for an entire weight tensor. Cheapest, but at int4 a single loud channel forces the scale so coarse that quiet channels get 1–2 usable codes → collapse to 75.3%.

**Percentile calibration** — Pick the range to cover, say, 99.9% of values, deliberately clipping rare outliers so ordinary values get finer steps. Cheap insurance below 8 bits.

**PTQ (Post-Training Quantization)** — Quantizing a *finished* model: choose maps, done — no gradients, no training loop, minutes of work. Everything through Part 6 is PTQ. It covers most of life; it runs out of road below int4.

## Q

**QAT (Quantization-Aware Training)** — Training the model *with the quantizer in the loop*, so gradient descent finds weights that survive rounding. In the notebook it revives int2 from **8.2% → 89.6%** (vs an equally-trained fp32 at 93.4%). The only game in town below 3 bits. Contrast with **PTQ** (no training).

**QLoRA (Quantized Low-Rank Adaptation)** — Fine-tuning an LLM whose weights are frozen in **4-bit NF4** while training only tiny adapter layers — huge memory savings. The reason NF4 exists. Roadmap only.

**Quantize** — Map a real number onto an integer code: `q = clamp(round(x / scale) + zero_point)`. The forward half of the affine map. See **dequantize** for the reverse.

## S

**Scale** — The step size of a quantization map: how many real units one integer step covers. Computed from the tensor's range. More scales (finer granularity) = better fidelity. The knob that makes int8 possible without an exponent field.

**Simulated (fake) quantization** — quantize→dequantize while the tensor stays fp32: the *values* are exactly what int8 could store, but the file doesn't shrink yet. Splits "what does the map cost in accuracy?" from "how do we get the size?" — and is the mechanism inside QAT.

**STE (Straight-Through Estimator)** — The trick that makes QAT trainable: `round()` has zero slope everywhere, so in the backward pass you *pretend it's the identity function* and pass gradients straight through. One PyTorch idiom: `w + (qdq(w) − w).detach()`. Crude but has carried production QAT since 2017.

**Symmetric quantization** — A map centred on zero (`zero_point = 0`), covering `[−a, +a]`. Best for **weights** (naturally zero-centred) and cheaper in the int-matmul (no cross-terms). Wastes half its codes on one-sided data — which is why activations use **asymmetric** instead.

## T

**torchao** — The PyTorch-native successor library for quantization (PTQ + QAT). The classic `torch.ao.quantization` API is **deprecated** — 2026's key migration news in the notebook. torchao's `Int8WeightOnlyConfig` produced Cairn's ~11 MB ship candidate at per-channel int8.

**TinyLlama** — A 1.1B-parameter chat LLM, the roadmap's "Cairn v2" target for on-device LLM quantization. Named, not run.

## Z

**Zero-point** — The integer code that represents real value 0.0. `0` for symmetric maps (by construction); a computed offset for asymmetric maps. It's what lets an asymmetric map cover a lopsided range without wasting codes.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Quantization2_Reading_Brief.md)
