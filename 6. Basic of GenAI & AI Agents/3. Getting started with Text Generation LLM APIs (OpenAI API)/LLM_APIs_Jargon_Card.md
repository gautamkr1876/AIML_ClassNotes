<a id="top"></a>
# LLM APIs & Decoding — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading the notebook, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`LLM_APIs_Reading_Brief.md`](./LLM_APIs_Reading_Brief.md) FIRST. This card is just the dictionary.

---

## A

**API key** — A secret string that authenticates you to a cloud API (OpenAI, Anthropic, HuggingFace). Loaded via `os.environ` or `getpass.getpass()` — never hardcoded.

**Anthropic** — The company behind the Claude family of models. One of the three big "private" providers alongside OpenAI and Google.

**`apply_chat_template()`** — A HuggingFace tokenizer method that takes a list of `{role, content}` messages and formats them using **that specific model's** special-token rules. Encapsulates Llama's, Phi's, Gemma's, and Qwen's formats so you don't have to remember them.

**Argmax** — The function "give me the index of the largest value." Used in greedy decoding: pick the token whose logit is highest.

**`AutoModelForCausalLM`** — HuggingFace class that auto-loads any decoder-only language model (GPT-2, Llama, Phi, Qwen, etc.) from a model name. The "causal" means autoregressive.

**`AutoTokenizer`** — HuggingFace's auto-loader for tokenizers. Pairs with the model — you must use the exact tokenizer the model was trained with.

## B

**Base model** — A model that has only been pretrained on next-token prediction. It will **continue text**, not **answer questions**. Example: prompt `"What is the capital of France?"` → base GPT-2 might output `"What is the capital of England? What is the capital of..."`.

**Beam search** — A decoding strategy that keeps the top-`k` most-likely *partial sequences* at each step (not just the top-1 token). Tries to find higher-probability *complete* sequences. More expensive than greedy.

**BOS token** — Beginning Of Sequence special token. Some tokenizers add it automatically; some require you to set it.

**BPE** — Byte Pair Encoding. The tokenization algorithm used by GPT/Llama. See the next jargon card for details.

## C

**Causal language model** — A model that predicts the next token from the previous tokens only (left-to-right). All modern chat LLMs are causal. The opposite is BERT-style "masked language modeling."

**Chat completions API** — OpenAI's main endpoint: `client.chat.completions.create(...)`. Takes a list of `{role, content}` messages and returns the next assistant message.

**Chat model** — An instruct model further specialised for **multi-turn dialogue**. Maintains conversation state via the messages list. Examples: ChatGPT, Claude, Llama-3-Instruct.

**Chat template** — The formatting rules a specific model family uses to combine system/user/assistant messages into one token stream. Llama-3 uses `<|begin_of_text|><|start_header_id|>...<|end_header_id|>`. Phi uses `<|system|>...<|user|>`. Each family is different; `apply_chat_template` hides this.

**ChatML** — A specific chat formatting standard (used by OpenAI, Qwen, and others). Uses tags like `<|im_start|>user`.

**Claude** — Anthropic's chat model family (Claude 3 Haiku, Sonnet, Opus, 3.5 Sonnet). Known for long context (200K) and careful reasoning.

**Context window** — The maximum number of tokens a model can read in one call. Phi-3-mini-128k → 128,000 tokens. GPT-4o → 128K. Claude 3.5 → 200K.

## D

**DeepSeek** — A Chinese AI lab whose open-weights models (DeepSeek-V3, V3.1) are competitive with frontier closed models. Use `<think>...</think>` blocks for visible reasoning.

**Determinism** — Same input → same output. In LLMs, set `temperature=0` (or `do_sample=False`) to be deterministic. Useful for testing and for tasks like classification where you don't want randomness.

**Decoder-only** — The architecture used by all chat LLMs (GPT, Llama, Claude, Mistral). Causal attention; left-to-right generation.

**Decoding strategy** — How you pick the next token from the model's probability distribution. Options: **greedy**, **beam search**, **top-k sampling**, **top-p (nucleus) sampling**.

**`device_map`** — A HuggingFace argument: `"auto"` puts the model on whatever hardware is available (GPU first, CPU fallback). Saves you from manual `.to("cuda")` calls.

**`do_sample`** — A HuggingFace generation argument. `True` = sample from the distribution (use temperature/top_k/top_p). `False` = deterministic (greedy or beam).

## E

**Entropy** — A measure of "how spread out" a probability distribution is. Uniform = max entropy (every token equally likely). One-hot = zero entropy (one token has all the probability). High temperature increases entropy.

**EOS token** — End Of Sequence special token. Generation stops when the model emits this.

**Exploration vs exploitation** — A sampling trade-off. **Exploit** = pick the most-likely token (greedy → boring but coherent). **Explore** = sample from the full distribution (creative but risky).

## F

**Finish reason** — A field in the API response that tells you *why* generation stopped:
- `stop` — the model emitted an EOS token (natural end). Good.
- `length` — hit `max_tokens` (response was truncated). Increase `max_tokens` to fix.
- `tool_calls` — the model wants to call a function (agent flow).

**Float16 (fp16)** — Half-precision floating-point. Half the memory of fp32 with minimal accuracy loss. Standard for inference on modern GPUs.

## G

**Gemma** — Google's open-weights model family (Gemma 2 9B, 27B).

**Generation parameters** — Hyperparameters controlling output: `max_tokens` (length cap), `temperature` (randomness), `top_k`, `top_p`, `num_beams`, `do_sample`.

**Generation model** — A model that **produces text** as output (vs. a reasoning model that thinks first). GPT-4o-mini, Llama-3-Instruct — these are generation models.

**Greedy decoding (greedy search)** — The simplest decoding strategy: at each step, pick the token with the highest probability (`argmax(logits)`). Deterministic. Can get stuck in repetitive loops.

**`getpass.getpass()`** — A Python function that prompts the user for input without echoing it on screen. Used for API keys.

## H

**HuggingFace (HF)** — The biggest open-source AI hub. Hosts 500K+ models, the `transformers` library, datasets, and Spaces. Free to download and run locally.

**Hyperparameter** — A model/generation setting (vs. a learned weight). `temperature`, `top_k`, `top_p`, `max_tokens` are hyperparameters.

## I

**Instruct model** — A base model fine-tuned on `(instruction, response)` pairs. Now follows instructions: prompt `"What is the capital of France?"` → `"The capital of France is Paris."` Examples: Llama-3-8B-Instruct, Phi-3-mini-instruct.

**Instruction tuning (also: SFT, Supervised Fine-Tuning)** — The training step that turns a base model into an instruct model.

## L

**Llama** — Meta's open-weights LLM family (Llama 3.1 8B / 70B / 405B Instruct).

**Logits** — The raw, un-normalised scores the model outputs at each position. Apply softmax to logits → probability distribution over the vocabulary.

## M

**`max_tokens` / `max_new_tokens`** — Cap on the response length. OpenAI uses `max_tokens` (total budget). HuggingFace uses `max_new_tokens` (newly generated, doesn't count the prompt). Hitting this triggers `finish_reason="length"`.

**Message** — A `{role, content}` dict. Roles: `"system"`, `"user"`, `"assistant"`, `"tool"`. The conversation is a list of these.

## N

**Nucleus sampling** — See **Top-p sampling**.

## O

**o1-mini / o1** — OpenAI's **reasoning models**. They run an internal "chain-of-thought" automatically before producing the final answer. You don't need to prompt with "think step by step" — they do it themselves.

**OpenAI** — The company. Their `openai` Python SDK provides `client.chat.completions.create(...)`.

## P

**Pad token** — Special token added to short sequences in a batch so all sequences are the same length. Models ignore positions that are padding.

**Phi** — Microsoft's small open-weights models (Phi-3-mini, Phi-3.5-mini, Phi-4). Optimised for "small but capable" — Phi-3-mini is 3.8B params.

**Prompt** — The user's input string fed to the model. After tokenization it becomes a sequence of token IDs.

**Prompt engineering** — Crafting prompts to elicit good outputs. The cheapest lever before fine-tuning.

## Q

**Qwen** — Alibaba's open-weights model family (Qwen 2.5 7B / 72B Instruct, Qwen 3 8B / 235B).

## R

**Reasoning model** — A model that performs explicit internal reasoning before answering. o1-mini, o1, Claude with extended thinking. Better for math, planning, complex logic. Slower and more expensive than a generation model.

**Role** — The "speaker" in a chat message. Standard roles: `system`, `user`, `assistant`, `tool`.

## S

**Sampling** — Drawing a token from a probability distribution randomly (vs. deterministically picking the top one). Triggered by `do_sample=True` in HF or `temperature > 0` in OpenAI.

**Seed** — An integer that initialises a random number generator. Same seed + same prompt + same model = same sampled output. Used for reproducibility.

**SFT** — Supervised Fine-Tuning. See **Instruction tuning**.

**Softmax** — Converts logits into a probability distribution. `softmax(x_i) = exp(x_i / T) / Σ exp(x_j / T)` where `T` is temperature.

**System message** — The first message, with role `"system"` — sets the model's persona and constraints. Example: `"You are a helpful assistant."`

## T

**Temperature** — A scalar that controls the **sharpness** of the next-token probability distribution. Applied as `softmax(logits / temperature)`:
- `T → 0` → distribution becomes one-hot (greedy, deterministic).
- `T = 1` → no change (original distribution).
- `T > 1` → distribution flattens (more random/creative).
- `T → ∞` → uniform distribution (every token equally likely).

**Think block / `<think>` tags** — Some models (DeepSeek V3.1, Qwen 3) put their reasoning inside `<think>...</think>` tags before producing the user-facing answer. Acts like a "reasoning model" but visible.

**TinyLlama** — A 1.1B parameter open chat model. Tiny enough to run on a laptop, big enough to demo instruction-following.

**Token** — The atomic unit a model reads/writes. ~4 characters of English per token.

**Tokenization** — Converting raw text into token IDs (and back). Done by the tokenizer that came with the model.

**Top-k sampling** — A decoding strategy. Only consider the `k` most-likely next tokens; zero out the rest; renormalise; sample. Common `k`: 40–100.

**Top-p sampling (nucleus sampling)** — A decoding strategy. Sort tokens by probability; take the smallest set whose cumulative probability is at least `p`; sample from that set. Common `p`: 0.9–0.95. Adaptive — uses fewer tokens when the distribution is peaked, more when it's flat.

**Tool / Tool calling (a.k.a. function calling)** — A feature where the model emits a structured request like `{"tool": "get_weather", "args": {...}}`, your code runs the function, and you feed the result back. Powers AI agents.

**`torch.float16`** — PyTorch's fp16 data type.

**Transformers (library)** — HuggingFace's main library. Provides `AutoTokenizer`, `AutoModelForCausalLM`, `pipeline`, `Trainer`, and `apply_chat_template`.

## U

**User message** — A `{role: "user", content: ...}` message — the actual question or instruction.

## V

**Vocabulary (vocab)** — The set of all distinct tokens a model can output. Each token has a unique integer ID. GPT-2 vocab: 50,257. Llama 3.1 vocab: 128,000.

---

[🔝 Back to top](#top)
