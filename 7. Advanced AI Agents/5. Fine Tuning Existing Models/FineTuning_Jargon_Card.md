<a id="top"></a>
# Fine-Tuning LLMs (SFT + LoRA) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebooks. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`FineTuning_Reading_Brief.md`](./FineTuning_Reading_Brief.md) FIRST. This card is just the dictionary.
>
> **Covers the notebooks:** `digital_notes_Fine_Tuning_Existing_Models.ipynb` (theory), `fine_tuning_llm_july_04_2026.ipynb` (the TinyLlama SFT + LoRA run), and `dummy_case_study_july_04_2026.ipynb` (the numpy ΔW toy).

---

## A

**Adapter** — A small, trainable add-on bolted onto a frozen model to change its behaviour without retraining. In LoRA it's the two tiny matrices (A and B) beside each big weight. The run trains a 2.25M-param adapter instead of the 1.1B model.

**Alignment** — The post-training step making a model helpful, harmless, honest (via RLHF/DPO after SFT). The notebook stops at SFT; alignment is mentioned, not run.

**Alpaca format** — An instruction-tuning layout with fields `instruction`, `input`, `output` (classic set: ~52K GPT-4 examples). The theory notebook converts it to ChatML before training.

**Alpha (`lora_alpha`)** — A LoRA knob controlling *how loud* the adapter's edit is; the scale applied is `alpha / rank`. In the run `alpha=16, rank=8` → ×`2.0`. **Don't confuse with rank:** rank = adapter *size/capacity*; alpha = *strength* of its contribution.

**AdamW (optimizer)** — Nudges trainable weights downhill during training. It keeps momentum + variance per parameter (~6 bytes each) — why full fine-tuning is memory-hungry and why LoRA saves so much.

**Attention (self-attention)** — The transformer mechanism where each token "looks at" others to decide what matters. Its four matrices — query, key, value, output (`q_proj`, `k_proj`, `v_proj`, `o_proj`) — are where this run injects LoRA.

**Autoregressive** — Generating text one token at a time, each new token conditioned on all the ones before it. This is how `model.generate()` produces the customer-support replies you see in the notebook.

## B

**Base model** — The pretrained model *before* fine-tuning. Here `TinyLlama-1.1B-Chat-v1.0` (1.1B params); kept as a reference to show "base vs fine-tuned" replies.

**Batch** — A group of examples processed together in one step (`batch_size=2`). Since a batch must be a rectangle, every example is padded to the same length (`MAX_LENGTH=256`).

**bfloat16 / float16 (dtype)** — Compact 16-bit formats for storing weights so a big model fits in memory. TinyLlama in bfloat16 takes ~2.2 GB (vs ~4.4 GB at 32-bit); auto-picked on MPS.

**BPE (Byte-Pair Encoding)** — A tokenizer scheme splitting text into sub-word chunks — how raw text becomes tokens.

## C

**Catastrophic forgetting** — When fine-tuning on a narrow task makes the model *lose* general ability. A pitfall; cured by fewer epochs, lower LR, diverse data. LoRA lowers the risk (frozen base untouched).

**Causal (decoder-only) model** — A model that only looks *backward* at previous tokens. TinyLlama/GPT/Llama are all causal. In code: `AutoModelForCausalLM`, `TaskType.CAUSAL_LM`.

**ChatML / conversational format** — A dataset format where each example is a list of role-tagged messages: `system`, `user`, `assistant`. The notebook builds this per example; `apply_chat_template` renders it to text.

**Chat template** — A model-specific recipe (on the tokenizer) turning role messages into the exact string the model was trained on — inserting markers like `<|user|>`, `<|assistant|>`, end-of-turn tokens. `add_generation_prompt=True` appends the bare `<|assistant|>` header: "your turn."

**Checkpoint** — A saved snapshot of weights. A full 7B checkpoint is ~14 GB; a LoRA adapter is only ~20–60 MB — a headline LoRA win.

**Cross-entropy loss** — The "wrongness score" the model minimises: `-log(prob it gave the correct next token)`. Prob 1.0 → loss 0; 0.001 → loss 6.9. Drops 3.63 → 1.24 across the 5 epochs.

## D

**Decoder block** — One repeated transformer layer: self-attention + feed-forward (MLP) sub-layers. Stacking `L` of these forms the model body; LoRA is injected into the weights inside.

**DPO (Direct Preference Optimization)** — A simpler RLHF alternative for alignment, learning from "A beats B" pairs. A future step, not run here.

**Dropout (`lora_dropout`)** — Randomly zeroes some LoRA-branch values during training to fight overfitting. Default 0.05; 0 for speed. The run keeps the default.

## E

**Embedding (token embedding)** — The lookup table turning each token ID into a vector. In the param dump, `embed_tokens.weight` (65.5M) stays **frozen** — LoRA never touches it.

**EOS token (End-Of-Sequence)** — A special token marking "text ends here." Llama tokenizers often lack a pad token, so the notebook reuses EOS as the **pad token**.

**Epoch** — One full pass over the dataset. The run does 5 epochs over 15 examples; more = more learning but rising overfit risk.

## F

**Feed-forward network (FFN / MLP)** — The second sub-layer per block (`gate_proj`, `up_proj`, `down_proj`). LoRA on the MLP matters *more* than on attention — but the run adapts attention only.

**Fine-tuning** — Training an already-trained model a bit more on your data to change its behaviour. Umbrella term; **SFT** and **LoRA** are specific flavours.

**Foundation model** — A large model pretrained on massive general data, meant to be adapted downstream. The base here is a small one.

**Frozen (weights)** — Weights held fixed — no gradient, no update — during training. In LoRA the whole 1.1B base is frozen; only the 2.25M adapter trains. **Contrast with trainable.**

**Full fine-tuning** — Updating *every* parameter. Expensive: a 7B model needs ~70 GB GPU memory (weights + optimizer states + gradients). The costly baseline LoRA avoids. **Contrast with PEFT/LoRA** (<1% trained).

## G

**Gradient** — The signal telling each trainable weight how to change to reduce loss. `loss.backward()` computes it; `optimizer.step()` applies it. In LoRA only the adapter gets gradients — the frozen base gets none, the source of the memory saving.

**Gradient checkpointing** — Recomputes some activations during the backward pass instead of storing them, to save memory. An OOM fix in theory; not needed for the tiny run.

**Greedy decoding** — Always picking the highest-probability next token. The notebook does the *opposite* — `do_sample=True`, `temperature=0.7` — so replies vary and sound natural.

## H

**Hyperparameter** — A setting *you* choose before training (not learned): rank, alpha, learning rate, epochs, batch size, target modules. Choosing these well is most of the skill in fine-tuning.

## I

**Instruction tuning** — Fine-tuning on (instruction, response) pairs so a text-completer learns to *follow instructions*. SFT is how you do it; the support dataset is a mini instruction-tuning set.

**Intrinsic dimension / low intrinsic rank** — The idea that a big model's *useful* changes live in a much smaller space than its full parameter count — *why* a low-rank adapter works at all. Empirically even rank 1–2 can compete with full fine-tuning.

## K

**KV cache (key-value cache)** — Stores past attention keys/values during generation so they aren't recomputed each step. Useful for inference, off during training — the notebook toggles `use_cache` on for generation, off for training.

## L

**Labels** — The "correct answers" the loss is graded against — the target token at each position. Built by copying `input_ids` and setting everything that *isn't* the assistant reply to **-100** (next entry).

**-100 (the ignore index / loss mask)** — PyTorch's cross-entropy **skips any position whose label is -100**, so loss counts *only* on the reply: system prompt, user message, assistant header, and padding are all -100 (read for context, never graded). Example 0: 38 of 110 tokens (35%) contribute to loss.

**Learning rate** — How big a step the optimizer takes per update. LoRA wants a *bigger* LR than full fine-tuning — the run uses `2e-4` (~10× the typical `2e-5`).

**lm_head** — The final linear layer turning the model's hidden vector into one logit per vocabulary word; after softmax it's the next-token probability distribution.

**Logits** — The raw, un-normalised per-token scores before softmax. In the numpy toy, `base_scores = W_base @ x` are logits: urgent 0.3, normal 0.9.

**LoRA (Low-Rank Adaptation)** — The star technique: freeze the big weight `W₀` and learn a small correction `ΔW = (alpha/rank)·B·A` from two skinny matrices. Cuts trainable params 99%+ (here to 0.2044%) and can be *merged* back for full-speed inference. **Contrast with full fine-tuning.**

**Loss masking (response-only masking)** — Computing loss only on the tokens you want the model to *produce* (the reply), not the prompt — via the -100 trick. Teaches *how to respond*, not *how to echo the question*.

## M

**Merge (merging the adapter)** — Folding the learned `BA` back into `W₀` so `W' = W₀ + (alpha/rank)·BA` is one plain matrix. After merging, the model has the same architecture and speed as the original — LoRA's "zero inference latency."

**MPS (Metal Performance Shaders)** — Apple Silicon's GPU backend for PyTorch. The run detects `mps`; other machines fall back to `cuda` (NVIDIA) or `cpu`.

## N

**Next-token prediction (NTP)** — The one objective behind *all* LLM training: given the tokens so far, predict the next. Both pretraining and SFT use it — SFT just restricts *which* tokens get graded (via loss masking).

## O

**OOM (Out Of Memory)** — When training exceeds GPU memory. Fixes: gradient checkpointing, smaller batch, shorter sequences — or just use LoRA. The reason PEFT exists.

**Optimizer states** — AdamW's per-parameter bookkeeping (momentum + variance + fp32 copy), ~6 bytes each. In full fine-tuning this dwarfs the weights; LoRA keeps them only for the tiny adapter.

**Overfitting** — Memorising the training set and getting worse on new inputs. Symptom: training loss drops but reply quality degrades. Fixes: fewer epochs, more dropout, lower LR, more data.

## P

**Padding** — Filler tokens added so every example in a batch reaches the same length (256 here). Masked two ways: `attention_mask=0` (don't attend) and label `-100` (don't grade).

**Parameter (weight)** — A learnable number in the model. TinyLlama has 1,100,048,384; LoRA makes only 2,252,800 trainable.

**PEFT (Parameter-Efficient Fine-Tuning)** — The umbrella family that freezes the base and trains a small extra set; LoRA is dominant. Also the HuggingFace library (`peft`) used via `LoraConfig`, `get_peft_model`.

**Post-training** — Everything after pretraining to make a base model useful: SFT, then alignment (RLHF/DPO). Contrast with **pretraining**.

**Prefix tuning** — An older PEFT method prepending trainable "virtual tokens" to the input. A LoRA alternative; downside: eats context-window space.

**Pretraining** — The first, most expensive stage: training on trillions of raw-text tokens for language and world knowledge. Gives *knowledge*; SFT gives *behaviour/style*.

## Q

**QLoRA (Quantized LoRA)** — LoRA on top of a 4-bit quantized base for bigger memory savings. *Not* used here (the run uses standard 16-bit LoRA); flagged as the next topic.

**Quantization** — Storing weights in fewer bits (e.g. 4-bit) to shrink memory. The building block of QLoRA.

**`q_proj`, `k_proj`, `v_proj`, `o_proj`** — The query, key, value, output projection matrices in attention. These four are the `target_modules` the run adapts.

## R

**Rank (`r`)** — The inner dimension of the two LoRA matrices — sets adapter *capacity*. Small rank = fewer params, faster, less expressive. Run uses `r=8`; common default 16; range 4–64. **Don't confuse with alpha:** rank = capacity, alpha = strength.

**RLHF (Reinforcement Learning from Human Feedback)** — Classic alignment: train a reward model from human preferences, then reinforce the LLM toward high-reward replies. Comes *after* SFT; not run here.

## S

**SFT (Supervised Fine-Tuning)** — Fine-tuning on curated (prompt, ideal-response) pairs via next-token prediction, loss masked to the response. Teaches the model to *follow instructions and match a style* — here a pirate-voiced support agent. The notebook's main technique.

**SFTTrainer** — HuggingFace TRL's trainer automating SFT (chat template, response-loss masking, LoRA, loop). The theory notebook uses it; the run hand-writes the loop to expose the parts.

**Softmax** — Turns logits into probabilities that sum to 1: `exp(score)/sum(exp(all))`. Applied at the output to pick the next token. In the toy, argmax stands in for "which class softmax favours."

**Special tokens** — Reserved markers like `<|user|>`, `<|assistant|>`, `<eos>`, `<pad>` structuring a sequence. The chat template inserts them — why the prefix is tokenized with `add_special_tokens=False`.

**System prompt** — A hidden instruction setting the assistant's persona ("friendly, concise TechMart agent…"). Part of the prompt, so read for context but masked out of the loss.

## T

**`target_modules`** — The weight matrices that receive LoRA. The run targets attention only: `["q_proj","k_proj","v_proj","o_proj"]`. Best practice adds MLP matrices, but attention-only keeps the demo simple.

**TaskType.CAUSAL_LM** — The PEFT setting telling LoRA it's adapting a causal (decoder-only) model, so it wires adapters correctly.

**Temperature** — A generation randomness dial. `0.7` (used here) gives varied, natural replies; near 0 is deterministic; higher is wilder.

**TensorBoard** — A live dashboard for `loss`, `learning_rate`, `grad_norm` during training. The theory notebook logs to it; the run prints a text progress bar instead.

**Token** — A sub-word chunk (via the tokenizer), the model's atomic unit of text. Prompts, replies, padding — all counted in tokens, not characters.

**Tokenizer** — Converts text ↔ token IDs. `AutoTokenizer.from_pretrained` loads TinyLlama's; it also owns the chat template and pad/eos setup.

**Trainable parameters** — The weights that actually update (`requires_grad=True`). Here only the LoRA `lora_A`/`lora_B` matrices — 2,252,800 (0.2044%). All else frozen. **Contrast with frozen.**

**TRL (Transformer Reinforcement Learning)** — The HuggingFace library providing `SFTTrainer` and `SFTConfig`, used in the theory notebook's higher-level path.

## W

**W₀ (W_base) / ΔW / W′ (W_effective)** — The three LoRA matrices. `W₀` = frozen weight; `ΔW = (alpha/rank)·B·A` = learned correction; `W′ = W₀ + ΔW` = what acts on the input. The toy's hand-picked `ΔW` flips a prediction "normal" → "urgent."

---

[🔝 Back to top](#top) · [→ Reading Brief](./FineTuning_Reading_Brief.md)
