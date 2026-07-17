<a id="top"></a>
# Model Quantization Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Quantization_Reading_Brief.md`](./Quantization_Reading_Brief.md) FIRST — it gives the big picture, the formulas, and a reading map. This card is just the dictionary.
> **The notebook:** `Model_Quantization_Techniques.ipynb` in this folder.

---

## A

**Absmax (Absolute Maximum)** — The simplest way to pick a scale for **symmetric** quantization: take the biggest absolute value in the tensor (ignoring sign) as the range edge, so `scale = |max| / 127`. Used by symmetric quantization, GGUF sub-blocks, and BitNet activation quantization.

**Absmean** — The rule BitNet 1.58b uses for **ternary** weights: scale by the *mean* absolute value, then round each weight to -1, 0, or 1. Contrast with absmax (uses the *max*).

**Activations** — The intermediate numbers a model computes as data flows through it (input × weights, layer by layer). Unlike weights, activations are **dynamic** — they change with every input — so their range can't be known ahead of time and needs **calibration**. In the notebook, layer 2's activations span ~-140 to +21, far wider than layer 0's -0.46 to +0.50.

**Asymmetric Quantization (zero-point quantization)** — A float→int mapping that is **not** centered on zero. It maps the tensor's true min (β) and max (α) onto the full integer range using a **scale** *and* a **zero-point**. Best for lopsided data (e.g., all-positive activations after ReLU). Twin of **symmetric**: asymmetric adds a zero-point shift; symmetric forces 0↔0. In the notebook it gives lower error (MSE 1.57) than symmetric (MSE 2.51) on the same tensor.

**AWQ (Activation-aware Weight Quantization)** — A 4-bit post-training method that protects the few weights tied to large activations while quantizing the rest aggressively. In the notebook it appears only as a vLLM loading snippet. Twin of **GPTQ** — AWQ ranks weight importance from activations, GPTQ from the Hessian.

## B

**BF16 (bfloat16, "brain float 16")** — A 16-bit float: 1 sign, **8 exponent**, 7 fraction bits. Keeps FP32's full range (±3.4×10³⁸) but sacrifices precision. The default for modern LLM training because it doesn't overflow/underflow on gradients. Twin of **FP16** — same 16 bits, but BF16 trades precision for range while FP16 does the reverse.

**BitLinear** — The layer BitNet uses in place of a normal Linear (fully-connected) layer. Stores weights in 1 bit and activations in INT8, and does fake quantization during training like QAT.

**BitNet** — Extreme-quantization models. Plain **BitNet** uses 1-bit weights (each is -1 or +1, chosen by the **signum function**). **BitNet 1.58b** uses ternary weights {-1, 0, 1} — "1.58" because log₂(3) ≈ 1.58 bits store three values.

**BitsAndBytes** — HuggingFace library for on-the-fly 4-bit quantization, usually with **NF4**. You pass a `BitsAndBytesConfig` when loading (`load_in_4bit=True`). Loading snippet only in the notebook.

**Bit-width** — How many bits store one number (FP32=32, INT8=8, INT4=4). Fewer bits = smaller model but coarser precision. Shrinking bit-width while keeping accuracy is the whole point of quantization.

## C

**Calibration** — Finding a good clipping range (and thus scale + zero-point) for **dynamic** values like activations, by running a small representative dataset through the model and recording the min/max seen. In the notebook, calibrating on 5 prompts through TinyLlama yields per-layer scales (layer 0 ≈ 0.0038, layer 2 ≈ 0.636).

**Calibration dataset** — The small sample (notebook: 100–200 in production; demos with 5 prompts) fed through the model to observe activation ranges. If it doesn't cover production values, the quantized model clips or loses precision on unseen inputs.

**Clamp / Clipping** — Forcing values outside the allowed range back to the nearest edge. After rounding you `clamp` to the integer limits (e.g., [-128, 127]) so nothing overflows. Also used deliberately to set a *tighter* range so **outliers** get clamped instead of stretching everyone's scale.

**Compression ratio** — How many times smaller the model got. FP32→INT8 is theoretically **4×** (32→8 bits). (The notebook's Quanto demo *reported* 1.0× — a size-reporting quirk of frozen weights, not a claim that INT8 doesn't shrink models.)

## D

**Dequantize** — The reverse of quantize: turn the stored integer back into an approximate float via `r ≈ scale × (q − zero_point)`. Approximate — the gap versus the original is the **quantization error**.

**Dynamic Quantization (PTQ)** — Computes activation scale/zero-point on the fly at runtime, per layer. More accurate than static (adapts to each input) but slower. Twin of **static quantization**.

**Dynamic range** — The interval of numbers a format can represent (e.g., BF16's ±3.4×10³⁸), set by the **exponent** bits. Don't confuse with **precision** (spacing of values, set by fraction bits).

## E

**Exponent** — The float bits that set the **range** (how big/small a number can be). FP32 and BF16 both have 8 (same range); FP16 has 5 (small range → overflow risk).

## F

**Fake-Quant (fake quantization)** — A training trick in **QAT** and BitNet: during the forward pass, quantize weights/activations to low bits then *immediately dequantize* back to float. This injects realistic rounding error so the model learns to compensate — without actually running in low precision yet.

**Fraction (mantissa)** — The float bits that set **precision** (how many meaningful digits). FP16 has 10 (more precise), BF16 has 7, FP32 has 23.

**Freeze (Quanto `freeze()`)** — In Optimum Quanto, the second step: `quantize()` installs placeholders; `freeze()` computes final scales and permanently converts weights to true INT8. Before freeze the model is "fake quantized."

**FP16 (16-bit float, "half precision")** — 1 sign, 5 exponent, 10 fraction bits. More precise per value than BF16 but max ~65,504, so training gradients can **overflow to infinity** or **underflow to zero** — needs **loss scaling**. Twin of **BF16**.

**FP32 (32-bit float, "full precision")** — The default training format: 1 sign, 8 exponent, 23 fraction bits. Accurate but heavy — a 70-billion-parameter model needs **280 GB**. Quantization shrinks away from FP32.

## G

**GGUF** — A quantization file format built to **offload layers to CPU** (CPU+GPU hybrid). Uses a two-level hierarchy: **super blocks** of **sub blocks**, where each sub block is absmax-quantized and its scale is itself quantized by the super block's scale. Supports mixed 2/4/6-bit. Loaded via `llama-cpp-python`.

**GPTQ (Generative Pre-trained Transformer Quantization)** — A 4-bit PTQ method that quantizes **one layer at a time**, uses the **inverse Hessian** to rank weight importance, and redistributes each weight's error onto the not-yet-quantized ones. GPU-focused. Twin of **AWQ**.

**Granularity** — How finely you slice a tensor before assigning scales: **per-tensor** (one for all) → **per-channel** (one per row) → **per-group** (one per block). Finer = lower error, more scales to store.

## H

**Hessian (inverse Hessian)** — A matrix of second derivatives describing how sensitive the output is to each weight. **GPTQ** uses it to order and error-correct quantization. Just read it as "weight-importance information."

## I

**IEEE-754** — The standard for storing floats in bits as **sign + exponent + fraction (mantissa)**. FP32, FP16, and BF16 are all variants of it.

**Inference** — Running a trained model to get predictions (vs training it). Quantization mainly targets inference — smaller weights, less memory, cheaper serving.

**INT4 (4-bit integer)** — Only 4 bits (16 values). The "extreme" tier — GPTQ, AWQ, GGUF, NF4, QAT target it. Big savings but real accuracy risk, which is why QAT is often paired with it.

**INT8 (8-bit integer)** — 8 bits (256 values: 0–255 unsigned, -128–127 signed). The **sweet spot** — ~4× smaller than FP32 with almost no quality loss on weight-only quantization. The notebook's main worked examples target INT8.

## K

**KV Cache (Key-Value Cache)** — A generation speedup: the model stores the Key and Value tensors of past tokens so it doesn't recompute them each step. For long sequences it can reach several GB, making it a quantization target alongside weights and activations. Like activations, it's **dynamic**.

## L

**Latency** — Time for one inference. Quantization aims to cut it, but the notebook's honest result shows INT8 was *slower* on GPU (2.65s vs 1.50s) because weight-only INT8 on GPU adds dequant overhead — the wins are memory and CPU speed.

**Linear Quantization** — The straight-line (affine) float↔int mapping used throughout: `q = round(x/scale) + zero_point`. Both symmetric and asymmetric are linear.

**Loss scaling** — A patch **FP16** training needs: multiply the loss by a big constant before backprop so tiny gradients don't underflow, then divide back. **BF16 doesn't need it** — a key reason BF16 wins.

## M

**Mixed Precision Training** — Trains in 16-bit (FP16/BF16) for speed/memory but keeps a **master copy of weights in FP32** so tiny updates don't round to zero. Frameworks (FSDP, DeepSpeed, Accelerate) default to BF16 on A100/H100.

**MSE (Mean Squared Error)** — The notebook's yardstick for quantization error: average of (original − dequantized)². Lower = closer to the original. Used to compare symmetric vs asymmetric (2.51 vs 1.57) and per-tensor vs per-channel (2.51 vs 1.81, a 27.9% drop).

## N

**NF4 (4-bit NormalFloat)** — A 4-bit data type used by **BitsAndBytes**, with its 16 levels spaced to match the bell-curve distribution of neural-net weights. Often paired with "double quantization."

## O

**Optimum Quanto** — HuggingFace's lightweight PTQ library used in the notebook: `quantize(model, weights=qint8)` then `freeze(model)`.

**Outlier** — A single value far larger than the rest of a tensor. Outliers wreck naive quantization by stretching the scale so normal values collapse into a few levels. Handled by **clipping** or finer **granularity**.

**Overflow** — When a number exceeds a format's max and becomes infinity. A real risk in **FP16** (max ~65,504); BF16 avoids it.

## P

**Parameters (weights)** — The billions of learned numbers in a model. **Static** (fixed after training), so their scale is computed directly — no calibration. Why LLMs are huge: 70B × 4 bytes = 280 GB in FP32.

**Per-channel quantization (per-row)** — A separate scale per row/channel, so a "loud" and a "quiet" channel each get an appropriate mapping. Lower error than per-tensor for a tiny metadata cost — the notebook shows a **27.9%** MSE drop. Twin of **per-tensor**.

**Per-group quantization** — Finer than per-channel: one scale per small group within a row. Most precise, most scales. In the notebook a 6×6 tensor with group_size=3 stores **12 scales** (vs 1 for per-tensor).

**Per-tensor quantization** — Coarsest: one scale and zero-point for the whole tensor. Simplest and fastest, but a few outliers ruin precision for everything. Twin of **per-channel** / **per-group**.

**Precision (floating point)** — How finely-spaced representable values are (gap between neighbors), set by **fraction** bits. Don't confuse with **dynamic range** (set by exponent).

**PTQ (Post-Training Quantization)** — Quantizing a model **after** training, with no/minimal retraining. Fast and cheap but can drop accuracy. Splits into **dynamic** and **static**. Twin of **QAT** — PTQ is quantize-then-hope; QAT retrains to compensate.

## Q

**QAT (Quantization-Aware Training)** — Inserts **fake-quant** modules and fine-tunes briefly so the model learns to tolerate rounding. Recovers 45–70% of accuracy lost to naive 4-bit, adds **zero** inference overhead, costs 1–5 epochs. Twin of **PTQ**. Use when PTQ drops accuracy by >2–3%.

**Quantization** — The core topic: reducing bit-width (FP32 → INT8/INT4) to shrink memory and speed inference while keeping accuracy high. Notebook analogy: like reducing the number of colors in an image.

**Quantization error** — The unavoidable gap between an original value and its quantize→dequantize roundtrip, because many floats collapse onto one integer. Measured with **MSE**.

## S

**Scale (scale factor)** — How many float-units each integer step is worth. Asymmetric: `(r_max − r_min)/(q_max − q_min)`. Symmetric absmax: `|max|/127`. Bigger scale = coarser steps = more error. In the 3×3 example, scale ≈ 3.58.

**Signum function** — Returns a number's sign (+1 or -1). **BitNet** uses it to quantize each weight to a single bit.

**Static Quantization (PTQ)** — Pre-computes activation scale/zero-point **once** with a calibration dataset, then fixes them. Faster than dynamic (no runtime work) but less accurate. Twin of **dynamic quantization**.

**STE (Straight-Through Estimator)** — Makes **QAT** trainable. Rounding has zero gradient (non-differentiable), which would kill backprop; the STE lets gradients pass through the fake-quant module as if it were an identity function, so the float weights still update.

**Symmetric Quantization** — A mapping centered on zero: real 0 ↔ integer 0, range ±|max|, no zero-point (it's 0). Simpler/cheaper than asymmetric but wastes range on skewed data. Notebook: MSE 2.51 vs asymmetric's 1.57. Twin of **asymmetric**.

## T

**Ternary weights** — Weights restricted to {-1, 0, 1}, used by **BitNet 1.58b**. The "power of 0": a zero weight drops out of the matrix multiply, so multiplication becomes pure addition.

**TinyLlama** — The 1.1-billion-parameter chat model used for the hands-on demos (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, FP32 ≈ 4196 MB).

**TorchAO** — PyTorch's optimization library providing the real QAT machinery (`qat_scheme`, `quantize_`, `QATConfig`) used with Unsloth.

## U

**Underflow** — When a number is smaller than a format's smallest value and becomes exactly zero. A risk in **FP16** (values below ~6×10⁻⁸ vanish), silently zeroing small gradients — another reason BF16 wins for training.

**Unsloth** — A fast fine-tuning library used to run QAT on Qwen3-4B; enable QAT with `qat_scheme="int4"` when building the PEFT model.

## Z

**Zero-point** — The integer that real value 0 maps to in **asymmetric** quantization; it shifts the mapping so a lopsided range fits. `z = q_min − r_min/scale`. In the 3×3 example z = -77 (and real 0 shows up as -77 in the quantized tensor). Symmetric always has zero_point = 0.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Quantization_Reading_Brief.md)
