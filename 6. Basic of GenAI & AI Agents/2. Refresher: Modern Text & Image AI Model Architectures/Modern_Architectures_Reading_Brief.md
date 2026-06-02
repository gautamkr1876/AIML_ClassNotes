<a id="top"></a>
# Modern Text & Image AI Architectures — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Modern_Architectures_Jargon_Card.md`](./Modern_Architectures_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up there.
> **The notebook:** `L2_Refresher_Modern_Text_&_Image_AI_Model_Architectures.ipynb` in this folder.

---

## 🎯 30-second TL;DR

**Every modern LLM (GPT, Llama, Claude) is the same architecture — a stack of decoder-only transformer blocks** trained on next-token prediction. The notebook builds one from scratch in PyTorch and then shows the **same architecture works on images** too (Vision Transformer / ViT) and **across modalities** (CLIP).

The headline numbers it builds toward:

- **Vocabulary:** GPT-2 has 50,257 tokens. Llama 3.1 has **128,000** — bigger vocab = fewer tokens per sentence = better multilingual coverage.
- **Context window:** GPT-2 = 1,024 tokens → Llama 3.1 = **128,000** → Claude 3.5 = 200,000. Two enablers: **RoPE** (smart positional encoding that extrapolates) and Flash Attention (memory-efficient attention).
- **ViT:** a 224×224 image is sliced into **14×14 = 196 patches** of 16×16, each patch becomes one token, plus a CLS token = 197 tokens, run through 12 transformer blocks.
- **CLIP:** trained on **400 million** (image, text) pairs → enables zero-shot image classification.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Evolution of transformer architectures** — encoder-decoder (2017) → encoder-only (BERT) → encoder-decoder (T5) → **decoder-only** (GPT, Llama). Why decoder-only won.
2. **Training-inference alignment** — decoder-only learns `P(x_t | x_<t)` at pretraining and uses *exactly the same thing* at inference. No mismatch.
3. **Tokenization & vocab size** — BPE, GPT-2's 50,257 vs Llama 3.1's 128K.
4. **Embedding lookup** — turning token IDs into `d_model`-dimensional vectors.
5. **Positional encoding evolution** — learned absolute embeddings (GPT-2) vs **RoPE** (Llama) and why RoPE enables 128K contexts.
6. **Scaled dot-product self-attention** — Q, K, V projections; the `softmax(QK^T / √d_k) · V` formula and why the `√d_k` scaling matters.
7. **Causal masking** — upper-triangular `-inf` mask so each token only sees the past.
8. **Multi-head attention** — splitting `d_model` into `n_heads` parallel attention slices.
9. **Transformer decoder block** — Pre-LN + attention + residual, then Pre-LN + FFN (4× expansion + GELU) + residual.
10. **Greedy decoding** — at each step pick `argmax(logits)`; visualised as a decision tree on GPT-2.
11. **Pretraining → instruction tuning → RLHF** — the three stages that turn raw GPT into a chatbot.
12. **Vision Transformer (ViT)** — patch embedding via strided Conv2d, CLS token, positional embeddings, 12 transformer encoder blocks, CLS-token classifier head.
13. **CLIP & zero-shot classification** — image encoder + text encoder trained on 400M pairs with contrastive loss; cosine similarity at inference.

---

## 🧠 The big idea

> **One architecture conquers everything.** The same transformer block — attention + FFN + LayerNorm + residual — works for text (GPT, Llama), images (ViT), and multimodal (CLIP). What changes is **what you feed in**: text tokens, image patches, or both.

For decades, NLP used RNNs and CV used CNNs. Each had hand-crafted inductive biases (sequential ordering for text, local-receptive-fields for images). Then 2017's "Attention Is All You Need" showed that **attention alone** is enough — and it transfers across modalities.

The recipe:
1. **Chop your input into tokens.** Text → BPE subwords. Image → 16×16 patches.
2. **Embed each token as a vector.**
3. **Add position info.** Learned (GPT-2) or rotated (RoPE).
4. **Stack N transformer blocks.** Each block: attention + FFN + LayerNorm + residual.
5. **Read out.** For language: predict the next token. For vision: feed the CLS token to a classifier.

That's the entire menu of modern AI.

---

## 📖 Core concept primers

### 1. Self-attention (Q, K, V)

> **🪜 Mental model:** every token "asks a question" (Q), every other token "advertises what it knows" (K), and what gets returned is a weighted blend of "content" (V).

For each token, the model learns three projections: a **Query**, a **Key**, and a **Value**. To compute the new representation for token `i`:

1. Take the dot product of `i`'s Query with **every** Key in the sequence → a row of relevance scores.
2. Divide by `√d_k` (where `d_k` is the per-head dimension) — keeps the scores from getting huge.
3. Apply softmax → these become probabilities that sum to 1.
4. Use them to take a weighted sum of every Value.

**The formula:**

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**In words:** *each token's new vector = a softmax-weighted average of all the Value vectors, where the weights come from how much each token's Key matches that token's Query.*

The notebook implements this in 3 lines of PyTorch (Cell ~45):

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

### 2. Causal masking — what makes a decoder a decoder

> **🪜 Mental model:** during training, you hide the answers. Token at position `i` can only see positions `0…i`, never `i+1, i+2, …`.

GPT-style models generate text left-to-right. So during training, each position must learn to predict the next token using **only what came before**. Causal masking enforces this.

Implementation: build an upper-triangular mask of `-inf`, add it to the attention scores **before** the softmax. Since `softmax(-inf) = 0`, future tokens get zero weight.

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)   # -inf becomes 0
```

**Why this matters in the notebook.** This is the single line that separates BERT (no mask, every token sees everything) from GPT (mask, each token only sees the past). Same architecture; one different line of code.

### 3. Multi-head attention — many parallel attention slices

> **🪜 Mental model:** instead of one big attention head looking at everything, run `n` smaller ones in parallel. Each learns a different relationship.

Single-head attention is too narrow — one head can't simultaneously track "subject-verb agreement," "coreference," and "long-range dependencies." Solution: **split** the `d_model` dimension into `n_heads` parallel "heads" of size `d_k = d_model / n_heads`.

For GPT-2 small: `d_model = 768`, `n_heads = 12`, so `d_k = 64`. Twelve attention computations run in parallel, then their outputs are concatenated and projected back to 768.

Efficient implementation: do **one** big Linear that projects to `3 × d_model` (Q, K, V at once), then `reshape` to `(batch, n_heads, seq_len, d_k)`. Cheaper than three separate Linears.

### 4. Positional encoding — learned absolute vs RoPE

> **🪜 Mental model:** attention is order-blind by default. You have to *tell* the model where each token sits.

Without position info, "the dog chased the cat" and "the cat chased the dog" look identical to attention. Two solutions:

- **Learned absolute embeddings (GPT-2, BERT).** A lookup table of shape `(max_seq_len, d_model)`. Position 0 has its own vector, position 1 its own, etc. Simple, but **can't extrapolate** — train at 1024, you're stuck at 1024.
- **RoPE — Rotary Position Embedding (Llama).** Instead of adding a vector, **rotate** the Q and K vectors by an angle that depends on the token's position. The dot product `Q · K` then naturally encodes the **relative distance** between two tokens. Crucial property: you can train at 4K tokens and infer at 128K because the math extrapolates.

**Why this matters.** This is *the* reason Llama 3.1 can handle 128K-token contexts. Learned absolute embeddings would have stopped at the training length.

### 5. The complete decoder block (Pre-LN + residual)

> **🪜 Mental model:** every block is `x = x + sublayer(LayerNorm(x))`, twice — once for attention, once for FFN.

Each transformer block is:

```python
x = x + self.attn(self.ln1(x))      # multi-head causal attention
x = x + self.mlp(self.ln2(x))       # FFN: Linear(d→4d) + GELU + Linear(4d→d)
```

Two parts to notice:

- **Pre-LN** (LayerNorm **before** each sublayer) — stabilises deep training. The 2017 paper used Post-LN; modern GPT uses Pre-LN.
- **Residual connections** (`x + ...`) — lets gradients flow through 48+ layers without vanishing.

**The FFN's 4× expansion.** Inside the MLP: `Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model)`. The 4× factor is convention. Most of the model's parameters live here, not in attention.

### 6. Vision Transformer (ViT) — images as patches

> **🪜 Mental model:** a CNN sees images one tiny window at a time. A ViT chops the image into 196 patches and lets every patch attend to every other patch from layer 1.

**Recipe (224×224 RGB image → classification):**

1. **Patchify.** Split into 14×14 = **196 patches** of 16×16 pixels. Implemented as one `Conv2d(3, 768, kernel_size=16, stride=16)` — a strided conv naturally slices the image into non-overlapping patches.
2. **Add a CLS token.** Prepend a learnable `(1, 768)` vector to the 196 patches → 197 tokens total.
3. **Add position embeddings.** Learnable `(197, 768)` matrix added to the token embeddings.
4. **Stack 12 transformer encoder blocks** (encoder = no causal mask; bidirectional).
5. **Read out.** Take the CLS token's final embedding; feed it to a linear classifier.

**Why ViTs work.** They have less inductive bias than CNNs (no locality, no translation equivariance baked in) — but with enough data (e.g., JFT-300M pretraining), they **learn** those properties and beat CNNs.

### 7. CLIP & zero-shot classification

> **🪜 Mental model:** train two encoders (one for images, one for text) so that matching pairs land near each other in a shared 512-dim space.

**Training.** 400 million (image, caption) pairs from the internet. Contrastive loss: for each image, its matching caption should be the most similar text out of all captions in the batch.

**Inference (zero-shot).** Given an image and a list of candidate labels:
1. Embed the image with the image encoder.
2. Embed each candidate label (e.g., `"a photo of a dog"`, `"a photo of a cat"`) with the text encoder.
3. Compute cosine similarity between image embedding and each text embedding.
4. The highest-similarity label wins.

**Why "zero-shot."** You never trained CLIP on your specific labels — you just embed them at inference. Works for any open-vocabulary classification task.

---

## 🔥 The headline architecture — at a glance

| Component | What it is | Modern default | Where in the notebook |
|---|---|---|---|
| **Tokenizer** | Text → token IDs | BPE (50,257 for GPT-2; 128K for Llama 3.1) | Early cells |
| **Token embedding** | Token ID → `d_model` vector | `nn.Embedding(vocab, 768)` | Early cells |
| **Position info** | Tell model where each token sits | RoPE (Llama) / Learned (GPT-2) | Discussion + diagrams |
| **Self-attention** | Each token weighs every other token | Multi-head + causal mask | Code cells (PyTorch) |
| **FFN** | Per-token MLP, 4× expansion | Linear→GELU→Linear | Code cells |
| **LayerNorm** | Stabilise training | Pre-LN | Code cells |
| **Residuals** | Skip connections | `x = x + sublayer(x)` | Code cells |
| **Decoding** | Pick next token | Greedy / beam / sampling | GPT-2 demo |
| **Post-training** | Make it follow instructions | SFT + RLHF | Discussion |

---

## 🧮 Key numbers to memorise

- **`d_model`** = embedding dimension. GPT-2 small: 768. ViT-B: 768. Llama 3 8B: 4096.
- **`n_heads`** = attention heads. ViT-B: 12. Each head has `d_head = d_model / n_heads = 64`.
- **`n_layers`** = transformer blocks. GPT-2 XL: 48. ViT-B: 12.
- **FFN ratio** = 4 (FFN hidden dim is 4×`d_model`).
- **Scaling factor in attention** = `√d_k`. Prevents softmax saturation.
- **ViT patchify**: 224×224 image / (16×16) = 14×14 = **196 patches** (+ 1 CLS = 197 tokens).
- **Context windows** (years of progress): GPT-2 = 1,024 → GPT-3 = 2,048 → Llama 2 = 4,096 → GPT-4 = 32K–128K → Llama 3.1 = **128K** → Claude 3.5 = 200K.
- **CLIP**: trained on 400M image-text pairs.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–10** | Transformer evolution (encoder-decoder → BERT → T5 → GPT) | **Skim** — ~3 min. Get the lineage straight. |
| **11–25** | Tokenization (BPE, vocab sizes), token embeddings, position embeddings (learned + RoPE) | **Read carefully** — ~5 min. The "why RoPE" point is load-bearing. |
| **26–45** | Scaled dot-product self-attention with code; causal masking; multi-head attention | **FOCUS — this is the architectural core** — ~10 min. Run the PyTorch cells. |
| **46–60** | Full decoder block: Pre-LN + attention + residual + FFN; greedy decoding with GPT-2 + tree visualisation | **Focus** — ~6 min. The "decoder block in 5 lines of PyTorch" is the takeaway. |
| **61–70** | Pretraining → instruction tuning → RLHF stages | **Read normally** — ~3 min. Conceptual. |
| **71–84** | Vision Transformer (patch embedding via Conv2d, CLS token, position embeddings, encoder blocks); CLIP zero-shot classification demo | **FOCUS** — ~8 min. The "same architecture works on images" reveal. |

**Total target read time for the notebook itself:** ~35 min. Add this brief's ~22 min and you're at **~55–60 min** — much faster than a cold read (90+ min with jargon Googling).

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why decoder-only won** — training-inference alignment; simpler than encoder-decoder.
- [ ] **The scaled dot-product attention formula** — `softmax(QK^T / √d_k) · V` — and why each piece is there.
- [ ] **What causal masking is** — and the one line of code that implements it.
- [ ] **Why multi-head attention beats single-head** — different heads learn different relationships.
- [ ] **The two flavours of position embedding** — learned absolute vs RoPE — and why RoPE enables 128K contexts.
- [ ] **The full decoder block recipe** — Pre-LN, attention, residual, FFN with 4× expansion, residual.
- [ ] **How a Vision Transformer turns an image into tokens** — Conv2d-based patch embedding + CLS token + position embeddings.
- [ ] **What "zero-shot" means in CLIP** — embed image and labels independently, cosine similarity picks the winner.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Why is the attention formula scaled by `√d_k`? What goes wrong if you remove the scaling?
2. Write down the one line of code that turns a BERT-style attention block into a GPT-style (decoder) one.
3. ViT-B has `d_model=768` and `n_heads=12`. What is `d_head`?
4. For a 224×224 image with patch size 16, how many tokens does the ViT process per image (including the CLS token)?
5. You train a model with **learned absolute position embeddings** at a sequence length of 1024. Why can't you use it at sequence length 8192 — and what positional encoding scheme would fix this?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **Without the `√d_k` scaling, the dot products `Q·K^T` grow large in magnitude (variance ~ `d_k`).** Large logits cause `softmax` to saturate — one weight goes to 1, all others to 0. Gradients vanish. Dividing by `√d_k` keeps the logit variance ~1 regardless of head dimension.

2. **Add a causal mask before the softmax:**
   ```python
   mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
   scores = scores.masked_fill(mask, float('-inf'))
   ```
   That's the only structural difference between BERT-style (no mask, bidirectional) and GPT-style (mask, autoregressive) attention. Same QKV math otherwise.

3. **`d_head = d_model / n_heads = 768 / 12 = 64`.** Every head operates on a 64-dim slice of the 768-dim embedding. After all 12 heads run, their outputs concatenate back to 768.

4. **197 tokens.** `(224 / 16)² = 14² = 196 patches`, plus 1 prepended CLS token = **197 tokens**. Each token is a 768-dim vector after the patch embedding.

5. **The learned position embedding matrix is `(1024, d_model)` — it has no embedding for position 1024 or higher**, so the model literally can't represent positions beyond what it trained on. **RoPE fixes this** because positions aren't looked up in a table — they're applied as a rotation whose math is defined for any position. Train at 4K, infer at 128K. This is exactly how Llama 3.1 reaches 128K context.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Modern_Architectures_Jargon_Card.md)
