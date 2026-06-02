<a id="top"></a>
# Modern Architectures Jargon Card

> **Use this file like a dictionary.** Skim it once (~8 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading the notebook, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Modern_Architectures_Reading_Brief.md`](./Modern_Architectures_Reading_Brief.md) FIRST. This card is just the dictionary.

---

## A

**Attention** — The mechanism that lets a transformer look at *every* other token in the sequence and decide which ones are relevant. Each token computes a "weighted sum" over all the others. Replaces the recurrence of RNNs and the local windows of CNNs.

**Attention Is All You Need** — The 2017 paper (Vaswani et al.) that introduced the transformer architecture. The title is also the key takeaway: you don't need recurrence or convolution — pure attention is enough.

**Autoregressive** — A model that generates output **one token at a time, left to right**, with each new token conditioned on everything before it. GPT, Llama, Claude — all autoregressive. The opposite of generating the whole output at once.

## B

**Base model** — A model that has only been pre-trained on next-token prediction. Doesn't follow instructions; just continues text. Needs instruction tuning + RLHF to become useful as a chatbot.

**Beam search** — A decoding strategy that keeps the top-k most-likely *partial sequences* at each step (not just the top-1 token). Better than greedy at multi-step planning, but more expensive.

**BERT** — Bidirectional Encoder Representations from Transformers. An **encoder-only** transformer (every token sees every other token, no causal mask). Great for classification and embedding; can't generate text.

**BPE (Byte Pair Encoding)** — The tokenization algorithm used by GPT/Llama. Starts with characters and iteratively merges the most-frequent adjacent pairs into single tokens. Produces subword tokens — "tokenization" might be 1 token, "tokenisation" might be 2.

## C

**Causal masking (a.k.a. autoregressive masking)** — A triangular mask applied to the attention weights so each token can only attend to **prior** tokens, not future ones. This is what makes a transformer "decoder-only" and lets it generate left-to-right. Implemented by setting future positions to `-inf` before the softmax.

**CLIP (Contrastive Language-Image Pre-training)** — OpenAI's model trained on 400 million (image, text caption) pairs. Learns a shared embedding space where matching image and text land near each other. Powers zero-shot image classification.

**CLS token** — A learnable "classification token" prepended to a sequence (used by ViT and BERT). After all the attention layers, the CLS token's final embedding is taken as a summary of the entire input and fed to a classifier.

**Context window (a.k.a. context length)** — The maximum number of tokens a model can read at once. GPT-2: 1,024. GPT-3: 2,048. GPT-4: 32K–128K. Llama 3.1: 128,000. Claude 3.5: 200,000. Bigger window = can summarise longer docs.

**Contrastive learning** — A training trick: pull matching pairs together in embedding space, push non-matching pairs apart. The recipe behind CLIP.

**Cross-entropy loss** — The standard loss for next-token prediction. Penalises the model when the true next-token probability is low.

## D

**d_model (model dimension)** — The width of each token's embedding vector. GPT-2 small uses `d_model = 768`. Llama 3 8B uses `d_model = 4096`. All token vectors carry this dimension throughout the network.

**Decoder-only** — A transformer architecture using only the decoder half — every token sees only the past (via causal masking). GPT, Llama, Claude, Mistral — all decoder-only. Cheap to train, simple to use, and dominates modern LLMs.

**Decoding strategy** — How you pick the next token from the model's probability distribution. Options: greedy (always top-1), beam search (top-k partial sequences), temperature sampling, top-p (nucleus), top-k.

## E

**Embedding** — A vector representation of something. **Token embedding** = a learned vector for each vocab word. **Position embedding** = a learned vector for each position (or computed via RoPE). **Patch embedding** = a vector for each 16×16 chunk of an image (ViT).

**Encoder** — A transformer half that processes the input — every token attends to every other token (bidirectional). Used in BERT and the original encoder-decoder transformer. Not used standalone in modern LLMs but is central to ViT.

**Encoder-decoder** — The original 2017 transformer architecture: an encoder processes the input (e.g., English sentence), then a decoder generates the output (e.g., French sentence) attending to both prior decoded tokens *and* the encoder's outputs. Used in T5 and translation models.

## F

**Feed-forward network (FFN, MLP)** — Inside each transformer block, after attention, every token is independently passed through a small two-layer MLP: `Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model)`. The "4×" is convention. This is where most of the model's parameters live.

**Flash Attention** — An optimised attention implementation that fuses the attention computation into one GPU kernel, dramatically reducing memory traffic. Lets you train much longer sequences without OOM.

## G

**GELU (Gaussian Error Linear Unit)** — The activation function used inside the FFN. A smoother version of ReLU — small negative values aren't zeroed out abruptly. Standard in modern transformers.

**GPT (Generative Pre-trained Transformer)** — OpenAI's model family. All decoder-only. Models: GPT-2 (open-weights, used in the notebook for demos), GPT-3 (175B params, the scaling-laws moment), GPT-4, GPT-4o (multimodal).

**Greedy search / greedy decoding** — The simplest decoding strategy: at each step, pick the **most-likely** next token. Deterministic but can get stuck in repetitive loops.

## H

**Head (attention head)** — One independent "view" of attention. Multi-head attention splits `d_model` into `n_heads` parallel slices of size `d_head = d_model / n_heads`, each running its own Q/K/V projections.

## I

**Instruction tuning (a.k.a. SFT, Supervised Fine-Tuning)** — Fine-tuning a base model on (instruction, response) pairs so it learns to follow instructions instead of just continuing text. Turns "raw GPT-2" into "ChatGPT-like."

## K

**Key (K)** — One of the three projections in attention. Each token produces a Key vector. To compute "how relevant is token B to token A?", you take the dot product of A's Query and B's Key.

## L

**LayerNorm (Layer Normalization)** — A normalization operation that, for each token, rescales its embedding to have mean 0 and variance 1, then learns a scale and shift. Stabilises training in deep transformers. "Pre-LN" applies it **before** each sublayer (the modern standard).

**Llama** — Meta's open-weights LLM family. Llama 1 (Feb 2023) → Llama 2 (July 2023, 4K context) → Llama 3 (8B/70B/405B) → Llama 3.1 (**128K** context via RoPE base-frequency scaling). All decoder-only.

**Logits** — The raw, un-normalised scores the model outputs for each vocabulary token at each position. You apply softmax to logits to get probabilities.

## M

**Multi-head attention** — Attention with multiple parallel "heads" (typically 8–32 of them). Each head independently learns to attend to different relationships (e.g., one head learns subject-verb agreement, another learns coreference). Outputs are concatenated and projected back to `d_model`.

## N

**n_heads** — The number of attention heads. ViT-B uses 12 heads with `d_model=768`, so each head has `d_head = 768/12 = 64`.

**n_layers** — The number of transformer blocks stacked on top of each other. GPT-2 XL: 48. ViT-B: 12. More layers = more compositionality but more compute.

**Next-token prediction (a.k.a. causal language modeling)** — The training objective: given the first `n` tokens, predict token `n+1`. Cross-entropy loss against the true next token. The whole pre-training of GPT/Llama is just this, at huge scale.

## P

**Patch (in ViT)** — A small square chunk of an image (typically 16×16 pixels). A 224×224 image becomes a 14×14 grid of patches, i.e., **196 patches**. Each patch is flattened and projected to a `d_model`-dimensional vector — the patch embedding.

**Patch embedding** — Implemented in PyTorch as `Conv2d(in_channels=3, out_channels=d_model, kernel_size=16, stride=16)` — one trick: a strided conv slices the image into patches and projects them in one operation.

**Position embedding / positional encoding** — Information about *where* each token sits in the sequence. Without it, attention is order-blind. Two flavours: **learned absolute** (a lookup table; GPT-2 style) and **RoPE** (rotate Q/K vectors; Llama style).

**Pre-LN (Pre-Layer Norm)** — Apply LayerNorm **before** the attention/MLP sublayer (vs. "Post-LN" which applies after). Pre-LN makes deep transformers train more stably. Standard in GPT-2 onwards.

**Pretraining** — The expensive first stage of training: predict the next token over trillions of tokens of text. After pretraining you have a "base model." Cost: millions of dollars; weeks on thousands of GPUs.

## Q

**Query (Q)** — One of the three projections in attention. Each token produces a Query vector that "asks" which other tokens are relevant.

## R

**Residual connection (a.k.a. skip connection)** — `output = input + sublayer(input)`. Lets gradients flow through deep networks without vanishing. Every transformer block has two — one around attention, one around the FFN.

**RLHF (Reinforcement Learning from Human Feedback)** — A post-training step where the model is fine-tuned using human preference data plus a reward model. Aligns the model's outputs with what humans actually want. The recipe behind ChatGPT.

**RoPE (Rotary Position Embedding)** — A clever way to inject position info: rotate the Q and K vectors by an angle that depends on the token's position. The dot product `Q·K` then naturally encodes the **relative distance** between tokens. Enables length extrapolation — train at 4K, infer at 128K.

## S

**Scaled dot-product attention** — The attention formula: `softmax(Q·K^T / √d_k) · V`. Translates to "compute pairwise relevance scores, normalise them with softmax, then take a weighted sum of Values." The `√d_k` scaling stops the scores from getting too large.

**Self-attention** — Attention where Q, K, and V all come from the *same* sequence. Each token looks at every other token in its own input. Contrast with cross-attention (Q from one place, K/V from another).

**Sequence length** — How many tokens are in the input. Fixed during training (e.g., 1024 for GPT-2), can grow at inference if the model supports it.

**Softmax** — Converts a row of real numbers into a probability distribution (positive, sums to 1). `softmax(x_i) = exp(x_i) / sum(exp(x_j))`. Used to turn attention scores into weights, and logits into next-token probabilities.

## T

**T5** — Google's encoder-decoder transformer. Reframes every NLP task as "text in → text out." Encoder reads the input; decoder generates the output.

**Token** — The atomic unit a transformer reads/writes. Roughly 1 token ≈ 4 characters of English. Each token has an integer ID looked up in the model's vocabulary.

**Tokenization** — Splitting raw text into tokens (and joining tokens back into text). Done by the model's tokenizer (e.g., GPT-2 uses BPE with 50,257 tokens).

**Transformer** — The neural network architecture used by every modern LLM. Made of stacked **blocks**, each containing self-attention + feed-forward + LayerNorm + residual connections.

**Transformer block (a.k.a. decoder block / encoder block)** — One layer of a transformer: `x = x + Attention(LN(x))` then `x = x + MLP(LN(x))`. Stacks of these are how you build GPT, Llama, ViT.

## V

**Value (V)** — One of the three projections in attention. Each token produces a Value vector — the actual content that gets summed up after the attention weights pick winners.

**Vision Transformer (ViT)** — A transformer applied to images. Splits the image into 16×16 patches, embeds each patch as a vector, adds a CLS token and position embeddings, and runs the whole thing through transformer encoder blocks. The CLS token's final embedding feeds a classifier. Matches or beats CNNs at scale.

**Vocabulary (vocab)** — The set of all distinct tokens a model can read or write. GPT-2: 50,257. Llama 3.1: 128,000 (bigger vocab = fewer tokens needed for the same text = better cross-lingual coverage).

## Z

**Zero-shot classification** — Classifying an image (or text) into categories the model was **never explicitly trained on**, by leveraging CLIP-style embeddings. You embed the image and a list of candidate labels (e.g., "a photo of a dog", "a photo of a cat"); the highest cosine similarity wins.

---

[🔝 Back to top](#top)
