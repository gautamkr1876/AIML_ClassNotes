<a id="top"></a>
# Fine-Tuning LLMs — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`FineTuning_Reading_Brief.md`](./FineTuning_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebooks:** `fine_tuning_llm_july_04_2026.ipynb` (the full LoRA build) + `dummy_case_study_july_04_2026.ipynb` (a numpy toy that shows LoRA's math). An earlier, shorter copy lives in `../5. Fine Tuning Existing Models/`.

---

## A

**Adapter (LoRA adapter)** — The small set of trainable weight matrices LoRA inserts into a frozen model. Training updates *only* the adapter; the base model is untouched. You can enable it (`model` as-is) or turn it off (`model.disable_adapter()`) to compare fine-tuned vs base behavior — the notebook does exactly this side-by-side.

**AdamW** — The optimizer used for training (`lr=2e-4`). Only the LoRA parameters are passed to it, since everything else is frozen.

**`apply_chat_template`** — The tokenizer method that turns a list of `{"role","content"}` messages into the exact text format a chat model expects (with the model's special role markers). `add_generation_prompt=True` appends the "assistant's turn starts here" marker so the model knows to generate a reply.

## B

**Base model** — The pretrained model before any fine-tuning. The notebook uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (and `distilgpt2` for a warm-up next-token demo). Fine-tuning adapts a base model to a specific behavior/domain.

## C

**Causal LM (CausalLM)** — A model that predicts the next token given all previous tokens (GPT-style, left-to-right). `AutoModelForCausalLM` loads one; `TaskType.CAUSAL_LM` tells LoRA it's fine-tuning this kind of model.

**Chat template** — The model-specific text scaffolding (system/user/assistant markers) that a chat model was trained on. You must format inputs with the *same* template the model expects, which is why `apply_chat_template` matters.

## D

**delta-W (ΔW)** — The change LoRA adds to a weight matrix: effective weights `= W_base + ΔW`. The `dummy_case_study` notebook shows this literally with numpy — a small `delta_W` added to a base classifier's weights is enough to *flip* its prediction, demonstrating why a tiny adapter can change behavior without retraining the whole model.

## E

**EOS / PAD token** — EOS = "end of sequence" marker. Llama tokenizers often lack a dedicated **PAD** (padding) token, so the notebook reuses EOS as PAD (`tokenizer.pad_token = tokenizer.eos_token`) — needed because batched training requires all sequences to be the same length.

## F

**Fine-tuning** — Continuing to train a pretrained model on your own data so it adapts to a specific task, domain, or persona. Two flavors here: **full** fine-tuning (update all weights — expensive) vs **PEFT/LoRA** (update a tiny adapter — cheap).

**Frozen (weights)** — Parameters that don't get gradient updates during training. In LoRA, the entire base model is frozen; only the adapter trains. The notebook prints each layer as `trainable` or `frozen`.

## I

**IGNORE index (-100)** — The special label value PyTorch's cross-entropy **skips**. Used in **response-only masking**: prompt tokens get label `-100` (no loss), assistant-reply tokens keep their real token id (loss counts). This teaches the model to produce good *replies* without wasting learning on reproducing the prompt.

## K

**KV cache (`use_cache`)** — An inference speed-up that caches attention keys/values. Disabled during training (`use_cache=False`, not needed with full-sequence gradients) and re-enabled for fast generation afterward.

## L

**Label masking (response-only)** — Building the `labels` tensor so loss is computed **only on the assistant's reply**: prompt + padding positions are set to `-100`, reply tokens keep their ids. The model still *reads* the whole conversation (via `attention_mask`) but is only *graded* on the reply.

**LoRA (Low-Rank Adaptation)** — A **PEFT** method: freeze the base model and inject small trainable low-rank matrices into chosen layers. Trains a tiny fraction of parameters, so it's fast and memory-light, and the resulting adapter is a few MB. Configured via `LoraConfig`.

**`LoraConfig`** — The PEFT config object. Key args in the notebook: `r=8` (rank), `lora_alpha=16` (scaling), `target_modules=["q_proj","k_proj","v_proj","o_proj"]` (which layers get adapters), `task_type=CAUSAL_LM`.

**`lora_alpha`** — A scaling factor for the adapter's contribution; the effective scale is `lora_alpha / r` (here `16/8 = 2.0`), i.e. `W + ΔW·2`. Higher alpha = stronger adapter influence.

## M

**MPS** — Apple's "Metal Performance Shaders" GPU backend for PyTorch (Apple Silicon). The notebook's device-detection cell picks MPS → CUDA → CPU, whichever is available.

## P

**PEFT (Parameter-Efficient Fine-Tuning)** — The family of techniques that adapt a model by training only a small number of new/selected parameters instead of all of them. LoRA is the most popular member. Hugging Face's `peft` library provides `get_peft_model(base_model, lora_config)`.

## R

**Rank (`r`)** — The size of LoRA's low-rank matrices (the notebook uses `r=8`). Lower rank = fewer trainable params (cheaper, less capacity); higher rank = more capacity (closer to full fine-tuning). The core LoRA knob.

## S

**SFT (Supervised Fine-Tuning)** — Fine-tuning on labeled **(input, ideal output)** pairs — here `(customer_message, ideal_agent_reply)`. The model learns to imitate the target replies. The notebook's dataset is a small set of support messages with a deliberately distinctive ("pirate") reply style, so the fine-tuning effect is obvious.

## T

**`target_modules`** — The specific weight matrices LoRA adapts. The notebook targets the attention projections `q_proj, k_proj, v_proj, o_proj` (query/key/value/output) — the standard choice for transformer LoRA.

**TinyLlama-1.1B-Chat** — The small (1.1-billion-parameter) chat model fine-tuned in the notebook. Small enough to train on a laptop GPU/MPS, real enough to show meaningful behavior change.

**`trainable` params** — After attaching LoRA, only adapter weights are trainable; the notebook sums and prints them to show they're a tiny fraction of the 1.1B total — the whole point of PEFT.

[🔝 Back to top](#top)
