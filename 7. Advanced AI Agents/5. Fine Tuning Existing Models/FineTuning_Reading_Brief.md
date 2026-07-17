<a id="top"></a>
# Fine-Tuning LLMs (SFT + LoRA) — Reading Brief

> **Read this ONCE, end to end, before opening the notebooks.** Target time: ~22 minutes. By the time you reach the code, every term in it will already make sense — you'll be confirming what you know, not learning blind.
>
> **Side reference:** keep [`FineTuning_Jargon_Card.md`](./FineTuning_Jargon_Card.md) open in another tab while reading the notebooks. When an unknown word appears, look it up there.
> **The notebooks (in this folder):**
> - `digital_notes_Fine_Tuning_Existing_Models.ipynb` — the full theory (SFT + LoRA), with a HuggingFace `SFTTrainer` walkthrough.
> - `fine_tuning_llm_july_04_2026.ipynb` — the hands-on run: SFT a TinyLlama chat model into a pirate-voiced support agent, using LoRA, with a hand-written training loop.
> - `dummy_case_study_july_04_2026.ipynb` — a 4-feature numpy toy that shows `W_base + ΔW` flipping a prediction.

---

## 🎯 30-second TL;DR

You do **not** need to retrain a whole 1.1-billion-parameter model to change how it talks. Freeze the base, bolt on a tiny **LoRA** adapter, and train only that.

- **Model:** `TinyLlama-1.1B-Chat` — **1,100,048,384** parameters.
- **What LoRA trains:** only **2,252,800** of them → **0.2044% trainable**. The other 99.8% stay frozen.
- **Result:** 5 epochs over just **15 support examples** in **13.6 seconds** on a laptop GPU drops the loss from **3.63 → 1.24**, and the fine-tuned model answers in the trained pirate-support persona while the base model rambles generically.

The one-line lesson (theory notebook's own words): **"SFT teaches style, not knowledge."** Pretraining put the knowledge in; **Supervised Fine-Tuning (SFT)** aligns *how* the model responds; **LoRA** makes that alignment cheap.

---

## 🗺️ Agenda — what the notebooks teach, in order

1. **Where fine-tuning sits** — pretraining (knowledge) → post-training (SFT → alignment). SFT is the focus.
2. **Next-token prediction + cross-entropy loss** — the single objective behind all LLM training.
3. **What makes SFT special** — same loss, but **masked to the response only** (the -100 trick).
4. **Dataset formats** — Alpaca vs conversational/ChatML vs prompt-completion.
5. **The cost problem** — why full fine-tuning needs ~70 GB for a 7B model → motivation for PEFT.
6. **LoRA theory** — `W′ = W₀ + (alpha/rank)·B·A`; why low-rank works (intrinsic dimension).
7. **LoRA hyperparameters** — rank, alpha, dropout, target modules, learning rate.
8. **Hands-on setup** — load TinyLlama, pick device/dtype, set up the tokenizer and pad token.
9. **Baseline generation** — see how the base model replies *before* any fine-tuning.
10. **Build the dataset** — chat templates + response-only label masking, by hand.
11. **Attach LoRA + train** — `LoraConfig`, `get_peft_model`, a hand-written AdamW loop, 5 epochs.
12. **Base vs fine-tuned** — the same prompts, adapter off vs on.
13. **The numpy toy** — a 2×4 `W_base`, a hand-picked `ΔW`, and a prediction that flips.

---

## 🧠 The big idea — sticky-note edits on a frozen textbook

Imagine a giant printed **textbook** (the pretrained model). It already contains almost everything — grammar, facts, reasoning. You don't like *one* thing: the tone of its answers. **You are not going to reprint the textbook.** That would cost a fortune (for a 7B model, ~70 GB of GPU memory and a full ~14 GB reprint per task).

Instead you write **sticky notes** and stick them onto the relevant pages: "when you answer here, say it like *this*." The book stays frozen; the sticky notes are tiny; you can peel them off and swap a different set for a different task in seconds.

That's **LoRA**. The page is a frozen weight `W₀`; the sticky note is a low-rank correction `ΔW = (alpha/rank)·B·A` from two skinny matrices. At read time the model uses `W₀ + ΔW`. Training only writes the sticky notes — **2.25M numbers, 0.2044% of the model** — so it's fast, cheap, and doesn't damage the book.

The other half of the idea is **SFT with response-only loss**: you *show* the model the whole conversation (system + user + reply) so it has context, but you only **grade** it on the assistant's reply. It learns *how to answer*, not *how to parrot the question*.

---

## 📖 Core concept primers

Six primers cover the heart of the notebooks. Each has a **mental model**, plain-English meaning, and a worked detail tied to the actual run.

### 1. SFT (Supervised Fine-Tuning)

> **🪜 Mental model:** *show, don't tell* — feed the model paired (question, ideal answer) examples until it copies the answering style.

**Supervised Fine-Tuning** takes a pretrained model and trains it further on curated **(prompt, ideal-response) pairs**. "Supervised" means every example comes with a known right answer. It uses the exact same objective as pretraining — **next-token prediction** — just on a small, high-quality, task-shaped dataset. In this run the "supervision" is 15 hand-written support exchanges where every reply is in a specific pirate voice.

**Why it matters here.** A raw base model completes text but doesn't reliably follow instructions or hold a persona. SFT turns "text completer" into "assistant that answers like *this*." The theory slogan: **SFT teaches style, not knowledge** — pretraining already supplied the knowledge.

**The catch that makes it "SFT" and not plain training:** loss masking (primer 3). Without it you'd also teach the model to reproduce the question.

### 2. Next-token prediction + cross-entropy loss

> **🪜 Mental model:** *fill-in-the-next-blank, graded by surprise* — the loss is how surprised the model was by the correct next token.

Every LLM is trained to answer one question at every position: **"given the tokens so far, what's the next token?"** The model outputs a probability for each word in its vocabulary (via **softmax**), and the grade is **cross-entropy loss**: `-log(probability it gave the correct next token)`.

- If it gave the right token probability **1.0** → loss **0** (no surprise).
- If it gave it **0.001** → loss **6.9** (very surprised, heavily penalised).

Labels are the input shifted one position (predict `sky` after `The`, `is` after `The sky`, …), graded in one forward pass, then averaged. **In the run this loss falls 3.63 → 2.58 → 2.11 → 1.68 → 1.24 across 5 epochs** — the model gets steadily less surprised by the pirate replies.

### 3. Response-only loss masking (the -100 trick)

> **🪜 Mental model:** *the model reads the whole page but is only tested on its own answer.*

During SFT the training text contains **both** the prompt (system + user) **and** the assistant reply. If you graded all of it, the model would also learn to generate the question — pointless. So you **mask the loss**: keep the reply's labels, and set every prompt/padding label to **-100**.

**Why -100?** PyTorch's cross-entropy is built to **skip any position whose label equals -100** — it contributes zero to the loss and zero to the gradient. So those tokens are still *read* (they're context) but never *graded*.

The notebook builds labels in three lines:
```python
labels = input_ids.clone()
labels[:prefix_len]         = -100   # system + user + assistant header
labels[attention_mask == 0] = -100   # padding
```
**Real numbers:** in example 0, of 110 non-padding tokens, only **38 (35%)** are the assistant reply and actually contribute to loss. The other 65% are read-for-context but ignored by the grader.

### 4. Chat templates

> **🪜 Mental model:** *a form with labelled boxes* — the template drops each message into the exact slot the model was trained to expect.

A chat model wasn't trained on free-floating text — it was trained on text with **role markers** (`<|system|>`, `<|user|>`, `<|assistant|>`) and end-of-turn tokens. The **chat template** (stored on the tokenizer) turns a list of role messages into that exact string. You never hand-format it; you call `tokenizer.apply_chat_template(messages, ...)`.

**Two flags matter:**
- `add_generation_prompt=True` — appends the bare `<|assistant|>` header, meaning *"now it's your turn to answer."* Used to build the **prefix** (everything before the reply).
- `add_generation_prompt=False` — used to build the **full** text (prefix + the actual reply) for training.

**Why the notebook builds both:** it needs the prefix length (`prefix_len`) to know exactly where the reply begins — that's the boundary for the -100 mask in primer 3.

### 5. LoRA (Low-Rank Adaptation)

> **🪜 Mental model:** *sticky-note edit on a frozen page* — freeze the big matrix, learn a tiny correction beside it.

Fine-tuning a weight matrix means computing an update: `W′ = W₀ + ΔW`. Full fine-tuning lets `ΔW` be the same giant size as `W₀`. **LoRA constrains `ΔW` to be low-rank** — it factors it into two skinny matrices `B` (tall) and `A` (wide):

`ΔW = (alpha / rank) · B · A`

If `W₀` is `d × k`, then `B` is `d × r` and `A` is `r × k`, where `r` (the **rank**) is tiny (8 here). Trainable parameters drop from `d·k` to `r·(d + k)`. **A cornerstone detail:** `B` starts at **zero**, so at step 0 `ΔW = 0` and the model behaves *exactly* like the base — training only nudges it from there.

**Real numbers from the run:** LoRA on TinyLlama's four attention matrices → **2,252,800 trainable** out of 1,100,048,384 → **0.2044%**. After training you can **merge** `BA` back into `W₀`, so inference is exactly as fast as the original model (LoRA's "zero added latency").

**Why it even works:** the *intrinsic dimension* hypothesis — the useful changes to a big model live in a much smaller space than its full parameter count, so a low-rank `ΔW` is enough.

### 6. Rank vs Alpha (the twin you must disambiguate)

> **🪜 Mental model:** *rank = size of the sticky note; alpha = how boldly it's written.*

These two are the most-confused LoRA knobs.

- **Rank (`r`)** sets the *capacity* of the adapter — the inner dimension of `A` and `B`. Bigger rank = more parameters, more expressive, slower. Run uses `r=8`; the common default is 16; range 4–64.
- **Alpha (`lora_alpha`)** sets the *strength* of the adapter's contribution. The correction is scaled by `alpha / rank` before being added. Rule of thumb: alpha = 1–2× rank.

**Run values:** `alpha=16, rank=8` → scale = `16/8 = 2.0`, i.e. the learned edit is doubled before it's added to the frozen weight. The clever part: the `alpha/rank` form makes the best learning rate roughly *independent of rank*, so you can change rank without re-tuning LR.

---

## 🔥 The headline experiment — at a glance

The TinyLlama run trains one LoRA adapter, then compares the same prompts with the adapter **off** (base behaviour) and **on** (fine-tuned).

| | **Base TinyLlama** | **Fine-tuned (LoRA on)** |
|---|---|---|
| Total parameters | 1,100,048,384 | 1,102,301,184 (base + adapter) |
| **Trainable parameters** | — | **2,252,800** |
| **% of model trained** | — | **0.2044%** |
| Frozen base params | 1,100,048,384 | 1,100,048,384 (unchanged) |
| Training data | — | **15 examples**, 5 epochs |
| Training time | — | **13.6 s** (~2.7 s/epoch) on Apple MPS |
| Loss curve | — | **3.63 → 2.58 → 2.11 → 1.68 → 1.24** |
| Reply style | Generic, rambling, drifts off-format | Concise, on-persona (pirate support voice) |
| Adapter file size | — | a few MB (vs ~2.2 GB full model) |

**Why so few parameters still change behaviour:** you didn't touch the model's *knowledge* (the frozen 99.8%). You added a small, targeted correction to the attention projections that shifts its *output style* — exactly what SFT is for.

**The numpy toy proves the mechanism in miniature** (see formulas below): a hand-picked `ΔW` added to a 2×4 base weight flips the predicted class from **normal** to **urgent** — no retraining of the base, just a small additive correction.

---

## 🧮 Formulas to memorise

### 1. The LoRA weight update

```
W' = W_0 + (alpha / rank) · B · A
```

**In words:** the effective weight equals the frozen base weight **plus** a scaled product of two small matrices — the up-projection `B` times the down-projection `A`, multiplied by `alpha / rank`.

- `W_0` — frozen pretrained matrix, shape `d × k`
- `B` — trainable, shape `d × r` (starts at **zero**)
- `A` — trainable, shape `r × k`
- `r` (rank) — the small inner dimension (8 in the run)
- `alpha` — scaling factor (16 in the run)
- `alpha / rank` — the effective scale (`16/8 = 2.0`)

**Worked example (the numpy toy — real numbers).** Here `ΔW` is hand-picked to *stand in* for `(alpha/rank)·B·A`. Input `x = [refund=1, broken=1, thank_you=0, general_question=0]`.

```
W_base @ x  = [urgent 0.3, normal 0.9]         → argmax = "normal"

ΔW          = [[ 0.60,  0.50, 0, 0],           # urgent row gets boosted
               [-0.18, -0.15, 0, 0]]           # normal row gets suppressed

W_effective = W_base + ΔW
W_effective @ x = [urgent 1.40, normal 0.57]   → argmax = "urgent"  ← prediction FLIPPED
```

The frozen `W_base` was never edited; a tiny additive `ΔW` changed the answer. That is LoRA in one slide.

### 2. The -100 loss mask

```
loss = cross_entropy(logits, labels)   where every label == -100 is skipped
```

**In words:** grade the model's next-token predictions against the labels, but **ignore every position whose label is -100.** Set the prompt tokens and padding to -100 so only the assistant reply is graded.

**Worked example (real numbers from the run).** For a sequence of 110 real (non-pad) tokens where the assistant reply is 38 tokens:
- Positions of system + user + assistant-header → label `-100` (read, not graded).
- Padding positions → label `-100` (not read *and* not graded; also `attention_mask=0`).
- The 38 reply tokens → real labels.
- **Result:** loss is computed on **38 / 110 = 35%** of visible tokens.

### 3. (Reference) Trainable-parameter count for LoRA

```
params_LoRA (per matrix) = r × (d_in + d_out)
```

**In words:** each adapted matrix adds `rank` times (input width + output width) parameters — far fewer than the `d_in × d_out` of the full matrix. Summed over the four attention matrices across all TinyLlama layers, this totals the **2,252,800** trainable params (0.2044%) reported in the run.

---

## 🗺️ Notebook reading map — where to spend your attention

**Theory notebook — `digital_notes_Fine_Tuning_Existing_Models.ipynb`:**

| Section | What it teaches | How to read |
|---|---|---|
| **Part 0–1.1** | Pretraining vs post-training; the training pipeline | **Skim** — ~4 min. Just place SFT on the map. |
| **1.2–1.3** | Next-token prediction, cross-entropy, dataset formats | **Read normally** — ~6 min. Primers 2 & 4 cover this. |
| **1.4 + Part 2** | Cost of full fine-tuning; LoRA theory (`W′=W₀+BA`), rank, alpha | **FOCUS** — ~10 min. The core. Primers 5 & 6. |
| **2.5–2.6** | Hyperparameter cheat sheet + LoRA benefits | **Reference** — ~4 min. Skim, note the defaults. |
| **Part 3** | `SFTTrainer` path (Qwen), TensorBoard | **Skim** — ~5 min. The manual run below is clearer for learning. |
| **Part 4 appendix** | Cheat-sheet tables (full-vs-LoRA, pitfalls) | **Reference** — keep for later. |

**The run — `fine_tuning_llm_july_04_2026.ipynb`:**

| Cells | What it teaches | How to read |
|---|---|---|
| **1–6** | Device/dtype, load TinyLlama, chat template, baseline replies | **Read normally** — ~6 min. Note the base model's rambling replies. |
| **7, 10** | The 15-example dataset; the -100 mask demo table | **FOCUS** — ~6 min. The -100 table (primer 3) is the key insight. |
| **11** | `SupportChatDataset` — builds input_ids/attention_mask/labels | **FOCUS** — ~8 min. This is response-only masking in real code. |
| **13–14** | `LoraConfig`, `get_peft_model`, the 0.2044% printout | **FOCUS** — ~6 min. Primers 5 & 6. |
| **15** | Hand-written AdamW training loop, loss 3.63→1.24 | **Read carefully** — ~5 min. |
| **16** | Base vs fine-tuned, `model.disable_adapter()` | **FOCUS** — ~5 min. The payoff. |

**The toy — `dummy_case_study_july_04_2026.ipynb`:** ~5 min, read all of it — the whole LoRA idea in numpy (formula 1).

---

## ✅ Walk-away checklist

After the notebooks, you should be able to say in your own words:

- [ ] **Why we fine-tune instead of retraining** — pretraining gives knowledge; SFT gives style/behaviour; retraining a whole model is far too expensive.
- [ ] **What the -100 trick does** — masks loss so only the assistant reply is graded (35% of tokens in example 0), teaching *how to answer*, not *how to echo the question*.
- [ ] **The LoRA equation** — `W′ = W₀ + (alpha/rank)·B·A`; base frozen, only `A` and `B` train; `B` starts at zero so training begins as the pure base model.
- [ ] **Rank vs alpha** — rank = adapter capacity/size; alpha = contribution strength; scale = `alpha/rank` (= 2.0 in the run).
- [ ] **The headline numbers** — 0.2044% of parameters trained, 15 examples, 5 epochs, 13.6 s, loss 3.63 → 1.24.
- [ ] **Why chat templates + `add_generation_prompt`** matter — they build the exact prefix so you know where the reply starts (the mask boundary).
- [ ] **What "merge" buys you** — folding `BA` into `W₀` gives zero added inference latency and swappable few-MB adapters.

If any of these feel shaky, revisit the matching primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. TinyLlama has ~1.1B parameters but LoRA trains only 2,252,800 of them. What percentage is that, and why does changing so few still change the model's behaviour?
2. In the run, `lora_alpha=16` and `r=8`. What is the effective LoRA scale, and which knob would you raise to give the adapter *more capacity* (not just more strength)?
3. A training example has 110 non-padding tokens; 38 of them are the assistant reply. How many positions contribute to the loss, and what label value is assigned to the other 72 real tokens?
4. At the very start of training, why does the LoRA-wrapped model behave *identically* to the base model?
5. In the numpy toy, `W_base @ x` predicts "normal" but `(W_base + ΔW) @ x` predicts "urgent." In one sentence, how is this the same mechanism as LoRA?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **0.2044%.** (`2,252,800 / 1,100,048,384`.) It still shifts behaviour because pretraining already installed the *knowledge* in the frozen 99.8%; the small adapter only needs to nudge the model's *output style/formatting* (via the attention projections), which is a low-dimensional change — the intrinsic-dimension idea.
2. **Effective scale = `alpha/rank = 16/8 = 2.0`.** To add *capacity* you raise **rank (`r`)** (that grows the `A`/`B` matrices and the parameter count). Raising **alpha** only makes the existing adapter's contribution *louder*, not bigger.
3. **38 positions contribute to loss** (the reply tokens). The other **72 real tokens** (system + user + assistant header) get label **-100**, which PyTorch's cross-entropy skips. (Padding tokens also get -100 and `attention_mask=0`.)
4. **Because matrix `B` is initialised to zero**, so `ΔW = (alpha/rank)·B·A = 0` at step 0. The effective weight is `W₀ + 0 = W₀` — exactly the base model. Training then moves `B` away from zero.
5. **A small additive correction to a frozen weight changed the output** — `ΔW` is the sticky-note edit; `W_effective = W_base + ΔW` is `W′ = W₀ + (alpha/rank)·BA`. LoRA just *learns* that `ΔW` from data instead of hand-picking it.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./FineTuning_Jargon_Card.md)
