<a id="top"></a>
# Fine-Tuning LLMs — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~18 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`FineTuning_Jargon_Card.md`](./FineTuning_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebooks:** `fine_tuning_llm_july_04_2026.ipynb` (full build, ~16 cells) + `dummy_case_study_july_04_2026.ipynb` (a numpy toy showing LoRA's math). An earlier truncated copy is in `../5. Fine Tuning Existing Models/`.
> **Note:** this notebook is code-dense with little prose, so this brief front-loads the *concepts* the code assumes.

---

## 🎯 30-second TL;DR

Prompting changes what a model *does this turn*; **fine-tuning changes what the model *is*** by nudging its weights toward your task, domain, or style. But updating all 1.1 billion weights is expensive — so the notebook uses **LoRA**, the dominant **parameter-efficient** method:

> **Freeze the whole base model, insert tiny trainable "adapter" matrices, and train only those.**

The demo: take `TinyLlama-1.1B-Chat`, fine-tune it on a handful of customer-support messages written in a distinctive style, and watch the *same model* go from generic replies to on-brand ones — while training well under 1% of its parameters. Two more essential ideas ride along: **response-only loss masking** (grade the model only on its reply, not the prompt) and **chat templates** (format inputs exactly the way the model was trained).

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Warm-up** — `distilgpt2` next-token generation, to see a raw base model complete text.
2. **Environment + device** — detect MPS / CUDA / CPU; pick a dtype.
3. **Load base model** — `TinyLlama-1.1B-Chat`, set PAD = EOS.
4. **Baseline replies** — generate support answers *before* fine-tuning (the "before" picture).
5. **SFT dataset** — small `(customer_message, ideal_reply)` pairs in a distinctive style.
6. **Response-only masking** — build `labels` with `-100` on prompt tokens.
7. **Attach LoRA** — `LoraConfig` (r=8, alpha=16, target attention projections), `get_peft_model`.
8. **Train the adapter** — AdamW, 5 epochs, lr 2e-4; only adapter weights update.
9. **Side-by-side** — base vs fine-tuned on the same prompts (`disable_adapter()`).
10. **(dummy case study)** — a numpy toy: `W_base + ΔW` flips a prediction, showing LoRA's intuition.

---

## 🧠 The big idea — teach the model, don't just ask it

You already have three ways to steer an LLM: **prompting** (instructions), **few-shot** (examples in the prompt), and **RAG** (retrieved context). All leave the model's weights untouched — they change the *input*. **Fine-tuning** changes the *model itself*: it continues training on your examples so the desired behavior is baked into the weights, no longer needing to be re-specified every call.

**The transferable analogy: hiring vs training an employee.** Prompting is handing a temp a sticky note of instructions each morning — fast, but they forget it the moment they clock out, and you re-explain forever. Fine-tuning is *training* the employee so the behavior becomes second nature — a bigger upfront investment, but afterward they just *are* that specialist. You fine-tune when the behavior is stable, hard to fully specify in a prompt, and used often enough to be worth the training cost.

The catch: fully re-training a 1.1B-parameter model is slow and memory-hungry. **LoRA** is the clever shortcut. Instead of rewriting every weight matrix `W`, it learns a small **ΔW** (delta) and uses `W + ΔW` — and it makes ΔW *low-rank* (two skinny matrices) so there are very few numbers to learn. The base model stays **frozen**; only the tiny **adapter** trains. The `dummy_case_study` notebook shows the punchline with plain numpy: adding a small `delta_W` to a base classifier's weights is enough to flip its answer from "normal" to "urgent" — *behavior change without touching the base weights.* That is LoRA in one picture.

---

## 📖 Core concept primers

Four primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, notebook specifics, and why it matters.

### 1. SFT — supervised fine-tuning on (input, output) pairs

> **🪜 Mental model:** flashcards. Front = customer message, back = the ideal reply. The model studies the deck until it answers in that style.

**SFT** trains the model on labeled **(prompt, ideal response)** examples so it learns to imitate the responses. The notebook's dataset is a small list of `(customer_message, ideal_agent_reply)` pairs written in a deliberately distinctive voice (a playful "pirate" support style). The distinctiveness is pedagogical: after fine-tuning, the style change is unmistakable, so you can *see* that learning happened. **Why it matters here:** SFT is the workhorse of practical fine-tuning (it's how base models become chat models). The "before" step — generating replies pre-training — exists so the side-by-side "after" makes the effect concrete.

### 2. Chat templates — format inputs the way the model expects

> **🪜 Mental model:** a form with labeled fields. Put the system note, the user line, and the assistant marker in the exact boxes the model was trained to read.

A chat model was trained on text with specific **role markers** (system / user / assistant). Feed it plain text and it underperforms; feed it the right scaffolding and it behaves. `tokenizer.apply_chat_template(messages, ...)` produces that exact format, and `add_generation_prompt=True` appends the "assistant turn starts now" marker so the model generates a reply rather than continuing the user's text. **Why it matters here:** it's easy-to-miss but essential — both generation *and* the training data must use the same template, or the fine-tune teaches the wrong pattern. (Also why the notebook sets `pad_token = eos_token`: batched training needs equal-length sequences, and Llama tokenizers ship without a dedicated pad token.)

### 3. Response-only loss masking (the -100 trick)

> **🪜 Mental model:** grading an exam. The question is printed on the page (the model reads it), but you only mark the *student's answer* — you don't award points for copying the question back.

During training the model sees the whole conversation, but you only want it to learn to produce good **replies**, not to reproduce prompts. So you build the `labels` tensor with the special value **`-100`** on every system/user/padding token (PyTorch's cross-entropy skips `-100`), leaving real token ids only on the assistant reply. The `attention_mask` still marks all real tokens as visible (1) vs padding (0). **Why it matters here:** without this mask, the model wastes capacity learning to echo prompts, and the fine-tune is weaker. The notebook devotes a whole toy cell (`tokens`/`section`/`labels`) to making the mask concrete before applying it to the real dataset.

### 4. LoRA / PEFT — train a tiny adapter, freeze the rest

> **🪜 Mental model:** sticky notes on a textbook. You don't rewrite the book (frozen base); you add a few annotations (the adapter) that change how it reads.

**LoRA** freezes the base model and inserts small trainable low-rank matrices into chosen layers — here the attention projections `q_proj, k_proj, v_proj, o_proj`, via `LoraConfig(r=8, lora_alpha=16, ...)` and `get_peft_model(base_model, config)`. Only the adapter trains (the notebook prints layers as `trainable`/`frozen` to prove it's a tiny fraction of 1.1B). Two knobs: **`r`** (rank) sets adapter capacity — bigger = more expressive, more params; **`lora_alpha`** scales the adapter's influence (effective `alpha/r = 2.0`, i.e. `W + ΔW·2`). After training you can call `model.disable_adapter()` to instantly get the *base* behavior back — the notebook uses this for the base-vs-fine-tuned comparison. **Why it matters here:** LoRA is why fine-tuning is feasible on a laptop — small memory, fast training, and a portable few-MB adapter you can swap on or off.

---

## 🔥 The experiment — at a glance

| | Setting |
|---|---|
| **Base model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (warm-up: `distilgpt2`) |
| **Method** | LoRA (PEFT) — base frozen, adapter trained |
| **LoRA config** | `r=8`, `lora_alpha=16` (scale 2.0), targets `q/k/v/o_proj` |
| **Data** | small SFT set: `(support message, ideal reply)`, distinctive style |
| **Loss** | cross-entropy on assistant reply only (prompt tokens = `-100`) |
| **Training** | AdamW, lr `2e-4`, 5 epochs, `use_cache=False` |
| **Eval** | same prompts, `disable_adapter()` (base) vs adapter on (fine-tuned) |
| **Result** | same model, ~<1% params trained, visibly on-style replies |

---

## 🧮 The one relationship to internalize — LoRA's ΔW

```
W_effective = W_base + (lora_alpha / r) · ΔW
            = W_base + (16 / 8) · ΔW
            = W_base + 2 · ΔW
```

**Word-by-word translation:** "the weights the model actually uses = the frozen pretrained weights **plus** a scaled version of the small learned adapter delta." `W_base` never changes; all learning lives in `ΔW`; `lora_alpha/r` sets how loudly the adapter speaks. **Worked reading (from the dummy case study):** a base classifier scores `urgent` vs `normal` from features; its `W_base` predicts *normal* for "refund + broken." Add a small `delta_W` and `W_base + ΔW` now predicts *urgent* — the adapter changed the decision without retraining the base. Turn the adapter off (`disable_adapter`) and you're back to `W_base` exactly.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0 | `distilgpt2` next-token warm-up | **Skim.** Just "this is a raw base model." |
| 1–3 | Env/device detection; load TinyLlama; PAD=EOS; inspect chat template | **Read.** Note the template + pad trick. |
| 4–5 | Baseline `generate_reply`; replies **before** fine-tuning | **Focus.** The "before" picture. |
| 6 | SFT dataset of `(message, ideal_reply)` pairs | **Focus.** |
| 8–11 | Response-only masking — the `-100` label trick (toy + real dataset) | **Focus + slow down.** The subtlest idea. |
| 12–13 | `LoraConfig` + `get_peft_model`; print trainable vs frozen | **Focus.** The LoRA knobs. |
| 14 | Train the adapter (AdamW, 5 epochs) | **Read.** |
| 15 | Side-by-side base vs fine-tuned (`disable_adapter`) | **Focus.** The payoff. |
| *(dummy)* | numpy `W_base + ΔW` flips a prediction | **Read first, actually** — best LoRA intuition. |

---

## ✅ Walk-away checklist

After the notebook, you should be able to say, in your own words:

- [ ] How fine-tuning differs from prompting / RAG (changes weights, not input).
- [ ] When fine-tuning is worth it vs just prompting.
- [ ] What **SFT** is and why the demo uses a distinctive reply style.
- [ ] Why inputs must use the model's **chat template**.
- [ ] What **response-only masking** does and why `-100` matters.
- [ ] What **LoRA** freezes vs trains, and what `r` and `lora_alpha` control.
- [ ] How `W_base + ΔW` lets a tiny adapter change behavior (and `disable_adapter` reverts it).

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. You need a model to reply in a fixed brand voice on every request, forever. Give one reason to fine-tune rather than just put the style in the system prompt.
2. In response-only masking, what value goes in the `labels` for the prompt tokens, and what does it cause PyTorch to do?
3. LoRA is attached to a 1.1B-parameter model. Roughly what fraction of parameters actually get trained, and what happens to the rest?
4. `lora_alpha=16` and `r=8`. What is the effective adapter scale, and what does raising `lora_alpha` do?
5. After fine-tuning, how do you make the model produce its *original* base-model reply for comparison?

<details>
<summary>Answers</summary>

1. Any of: the behavior is baked into the weights so it doesn't consume prompt tokens every call (cheaper/faster at scale); it's more reliable than instructions the model can drift from; and it can capture a style/behavior that's hard to fully specify in words. (Prompting re-explains every turn; fine-tuning makes it the model's default.)
2. **`-100`** (the IGNORE index). PyTorch's cross-entropy **skips** those positions, so no loss is computed on prompt/padding tokens — the model is graded only on its assistant reply (which it still *reads* via the attention mask).
3. A **tiny fraction — well under 1%** (just the LoRA adapter matrices on the targeted attention projections). Everything else — the full base model — is **frozen** (no gradient updates).
4. Effective scale = `lora_alpha / r = 16/8 = 2.0`, i.e. `W + 2·ΔW`. Raising `lora_alpha` increases the adapter's influence on the effective weights (the fine-tuned behavior speaks "louder").
5. Call **`model.disable_adapter()`** — with the LoRA adapter switched off you're back to `W_base` exactly, i.e. the untuned base model's behavior. (The notebook uses this for its base-vs-fine-tuned side-by-side.)

</details>

[🔝 Back to top](#top)
