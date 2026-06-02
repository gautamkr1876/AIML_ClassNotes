<a id="top"></a>
# AI Engineering Jargon Card

> **Use this file like a dictionary.** Skim it once (~8 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading the notebook, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`AI_Engineering_Reading_Brief.md`](./AI_Engineering_Reading_Brief.md) FIRST. This card is just the dictionary.
>
> AI Engineering is famously acronym-heavy — there are ~60 terms here. Don't try to memorise them; just know where to look them up.

---

## A

**Accelerate (library)** — A Hugging Face library that automatically figures out the best hardware setup for running a model (GPU, CPU, multi-GPU). Used in the notebook with `device_map="auto"` to make TinyLlama "just work" on whatever machine you're on.

**Agent (AI Agent)** — An LLM-powered system that doesn't just answer — it **decides what to do**, **uses tools**, and **loops** until it finishes a task. The notebook builds a small math agent that picks between `add`, `multiply`, `subtract` tools. Unlike a chatbot, an agent takes actions.

**AI Engineering** — The discipline of **using** pre-trained foundation models (LLMs) to build production applications. Different from **ML Engineering**, which is about training models from scratch. The key skills are prompt design, orchestration (RAG, agents), evaluation, and deployment — not gradient descent.

**API key** — A secret string that authenticates you to a cloud API (OpenAI, Anthropic, etc.). In the notebook it's loaded via `getpass.getpass()` and stored in an environment variable so it's never hardcoded into the file.

**AutoGen** — A Microsoft framework for building **multi-agent** systems where several LLM agents talk to each other to solve a task. Listed in the LLM Frameworks section of the notebook.

**AWQ** — Activation-aware Weight Quantization. A specific way of shrinking a model's numeric weights (e.g., from 16-bit to 4-bit) so it uses less memory. One of several quantization formats (alongside **GPTQ**, **GGUF**, **SmoothQuant**).

**Axolotl** — A finetuning framework optimised for ease of use; lets you train LLMs on your own data with a YAML config. Listed alongside **TRL** and **Unsloth**.

## B

**Batching (continuous batching)** — Grouping multiple user requests into one GPU call. *Continuous* batching (vLLM's trick) lets new requests join an in-flight batch instead of waiting for the whole batch to finish — keeps the GPU 100% busy.

## C

**Chain of Thought (CoT)** — A prompting technique where you ask the model to **"think step by step"** before giving the final answer. Dramatically improves performance on math and multi-step reasoning. The notebook uses it for the apple-pricing math problem.

**Chat completion** — OpenAI's main API endpoint. You send a list of `{role, content}` messages (system, user, assistant) and the model returns the next assistant message. The whole notebook's OpenAI usage centres on `client.chat.completions.create(...)`.

**Chroma** — A simple, local vector database. One option among Pinecone, Milvus, FAISS, Weaviate. The notebook uses FAISS instead.

**Comet ML** — An experiment-tracking platform. Logs your hyperparameters, metrics, and outputs to a dashboard so you can compare runs. The notebook uses it to track a tiny `distilgpt2` finetuning run.

**Cosine similarity** — A distance metric for embeddings. `cos(θ) = A·B / (‖A‖‖B‖)`. Output ranges from -1 (opposite) to 1 (identical). The standard choice for text embeddings.

**CoT** — See **Chain of Thought**.

**CrewAI** — A multi-agent framework where agents have **roles** (researcher, writer, reviewer) and collaborate on a goal. Listed alongside LangGraph, AutoGen, smolagents.

## D

**Deterministic (output)** — Same input always produces the same output. In LLMs, setting `temperature=0` makes the response (near-)deterministic.

**device_map** — A Hugging Face argument that decides where to load model weights (`"auto"` = pick GPU if available, else CPU; `"cuda:0"` = a specific GPU).

**Diffusion model** — A type of foundation model that **generates images** (Stable Diffusion, DALL·E). Listed in the AI Engineering definition alongside LLMs and multimodal models.

**distilgpt2** — A small (82M parameter) version of GPT-2, distilled for speed. The notebook finetunes it on 5 Q&A examples to demonstrate the Trainer API end-to-end.

**Dot product** — Another distance metric: `Σ(aᵢ × bᵢ)`. Used when the **magnitude** of the vector also matters (not just direction).

**DPO (Direct Preference Optimization)** — A finetuning method that teaches a model human preferences directly (without a separate reward model). One of the post-training options alongside **SFT** and **RLHF**.

**DSPy** — A framework that treats prompts as **programmable, optimisable** objects — you write code, DSPy automatically tunes the prompts/weights. Listed under "Programmatic LLMs."

## E

**Embedding** — A list of numbers (e.g., 1536 floats) that represents the **meaning** of a piece of text. Similar texts have similar embeddings (close in vector space). The foundation of RAG and semantic search. The notebook produces them with `text-embedding-3-small`.

**Edge deployment** — Running an LLM on a phone, browser, or other resource-constrained device. Tools: MLC LLM, mnn-llm.

**Euclidean distance (L2)** — Another distance metric: `√Σ(aᵢ − bᵢ)²`. Standard "straight-line" distance in space. Used in the notebook's FAISS index (`IndexFlatL2`).

**Evaluation (LLM evaluation)** — Measuring how good an LLM's responses are. Hard because outputs are free-form text, not labels. Tools: **Ragas**, **DeepEval**, **LM Evaluation Harness**.

## F

**FAISS** — Facebook AI Similarity Search. A library for fast nearest-neighbour search over millions of embeddings. The notebook uses `IndexFlatL2` — a basic flat index using L2 (Euclidean) distance.

**Faithfulness** — A Ragas metric: "Are the facts in the LLM's answer actually supported by the retrieved context?" High = no hallucination. Low = the model made things up.

**Few-shot prompting** — Prompting where you **include examples** in the prompt before the real question. Also called **in-context learning**. The model imitates the pattern. Better than zero-shot for custom formats or edge cases.

**Finetuning** — Continuing to train a pre-trained model on your **own data** so it specialises. Different from prompt engineering (which doesn't change the model). The notebook does a tiny finetune on `distilgpt2`.

**Foundation model** — A large, general-purpose pre-trained model that can be adapted to many tasks (LLMs, diffusion models, multimodal models). The thing AI Engineers *use* rather than train.

**FP8 / INT8** — Numeric formats with 8 bits per weight (vs. the usual 16 or 32). Smaller = less memory + faster, with a small accuracy cost. A form of quantization.

**Function calling** — Same idea as **tool calling**. The LLM emits a structured request like `{"tool": "multiply", "args": {"a":6, "b":7}}`, your code executes it, then sends the result back to the LLM.

## G

**garak** — A vulnerability-testing tool for LLMs. Probes for prompt injection, jailbreaks, data leakage. Listed under "Security."

**GGUF** — A specific quantized model file format used by `llama.cpp`. The standard for running quantized models on local hardware.

**GPT-4o / GPT-4o-mini** — OpenAI's chat models. `gpt-4o-mini` is the cheaper, smaller version — the notebook uses it for all OpenAI examples.

**GPTQ** — Another quantization method (similar to AWQ). Different math, similar goal: shrink the model.

**Gradio** — A Python library that turns a function into a web UI in 3 lines of code. Used to build quick LLM demos.

**GRPO** — Group Relative Policy Optimization — a newer RL-based finetuning method. Listed alongside SFT, DPO, RLHF.

## H

**HF (Hugging Face)** — The dominant open-source AI hub. Hosts 500k+ models, datasets, the `transformers` library, and Spaces for hosting demos. The OpenAI-vs-HF split is the central comparison in the notebook's LLM Providers section.

**HF Spaces** — Hugging Face's free demo-hosting service. Drop in a Gradio/Streamlit app, get a public URL.

## I

**In-context learning** — Same as **few-shot prompting**. The model "learns" the pattern from examples in the prompt — no weights change.

**Inference** — Running a trained model to produce outputs (as opposed to training it). The notebook's vLLM section is entirely about making inference fast.

## J

**Jailbreak** — A prompt designed to make the model bypass its safety guidelines. Tested against by tools like garak.

## K

**KV cache** — During LLM inference, the model caches each token's "key" and "value" tensors so it doesn't recompute them for every new token. Memory-hungry — managing it well (e.g., **PagedAttention**) is a big throughput win.

## L

**LangChain** — The most popular LLM-orchestration framework. Chains together prompts, models, retrievers, output parsers. The notebook uses its **LCEL** (LangChain Expression Language) — the `prompt | llm | parser` pipe syntax.

**LangGraph** — LangChain's framework for **agents** specifically. Models the agent's reasoning as a graph (state machine). Used in the notebook's math agent example.

**Latency** — How long a single request takes to respond. Lower = better user experience. Different from **throughput** (total requests per second).

**LCEL (LangChain Expression Language)** — LangChain's pipe-style API for composing chains: `chain = prompt | llm | StrOutputParser()`. Reads left-to-right like a Unix pipeline.

**llama.cpp** — A C++ inference engine for running LLaMA-family models on CPUs and consumer hardware. Uses the **GGUF** format.

**LlamaIndex** — A LangChain-alternative focused specifically on **RAG** — indexing and querying documents.

**LLM (Large Language Model)** — A neural network trained on huge amounts of text that can generate, classify, summarise, reason. GPT-4o, Claude, Llama-3, etc. The "brain" of all AI Engineering applications.

**LM Evaluation Harness** — An open-source benchmarking library that runs LLMs against standard test sets (MMLU, HellaSwag, etc.).

**LM Studio** — A desktop GUI for running LLMs locally — like iTunes for models.

**LMQL** — A query language that **constrains** LLM output to follow a specific grammar/schema. Alternative to **Outlines**.

## M

**max_tokens / max_new_tokens** — A cap on how long the model's response can be. OpenAI uses `max_tokens`; Hugging Face uses `max_new_tokens` (the *new* tokens only, not counting the prompt).

**MCP (Model Context Protocol)** — An emerging standard (Anthropic-led) for connecting LLMs to external tools and data sources. Treats tools like a USB-C standard for AI.

**Messages (chat format)** — A list of `{role, content}` dicts the chat completion API expects. Roles: `"system"` (instructions), `"user"` (the question), `"assistant"` (the model's reply, when continuing a conversation).

**Milvus** — A scalable open-source vector database. One of the production-grade alternatives to FAISS.

**MLC LLM** — A framework for compiling LLMs to run efficiently on **edge devices** (phones, browsers).

**MLflow** — An experiment-tracking platform similar to Comet and W&B. Logs runs, models, and metrics.

**MoE (Mixture of Experts)** — A model architecture where each token only activates a few "expert" sub-networks instead of the whole model. Mixtral-8x7B is a famous example: 8 experts, only 2 active per token.

**Multimodal model** — A foundation model that handles **multiple input types** — text + images + audio. Examples: GPT-4o, Gemini, Qwen-VL.

## O

**Ollama** — A local-LLM runner. CLI-driven, simpler than LM Studio. `ollama run llama3` and you're chatting.

**OpenAI SDK** — The official Python client (`pip install openai`). Used throughout the notebook for chat completions and embeddings.

**Outlines** — A library that forces an LLM to produce output matching a JSON schema, regex, or grammar. Used when you need strict structured output.

## P

**PagedAttention** — vLLM's KV-cache trick: stores cache values in fixed-size blocks (like operating-system memory pages) instead of one big block. Saves memory, enables sharing across requests. ~24× throughput vs. naive HuggingFace inference.

**Pinecone** — A managed (paid) vector database. Often the production choice for scale.

**Pipeline (Hugging Face)** — A high-level helper that wraps tokenization + model + decoding into one call: `pipeline("text-generation", model=...)`. Easiest way to use an HF model.

**Prefix caching** — vLLM technique: if the same prompt prefix (e.g., a system message) appears in many requests, cache its KV values once and reuse them. Big win for chat apps.

**Prompt engineering** — The art of writing prompts that get good outputs. Includes zero-shot, few-shot, CoT, role prompting, etc. Cheap (no training); often the first lever you pull when results are bad.

**Prompt injection** — An attack where user input ("ignore previous instructions and...") hijacks the system prompt. The main security concern in LLM apps. Tested by **garak**.

## Q

**Quantization** — Shrinking a model's numeric weights from 16-bit floats to 8-bit, 4-bit, etc. Trade-off: 2–4× memory savings and faster inference for a small accuracy drop. Formats: AWQ, GPTQ, GGUF, SmoothQuant, FP8, INT8.

## R

**RAG (Retrieval-Augmented Generation)** — The 5-step pipeline that grounds an LLM in your private documents: (1) embed docs, (2) store in vector DB, (3) embed the user's question, (4) retrieve top-k similar docs, (5) generate an answer using those docs as context. The notebook builds one end-to-end.

**Ragas** — An LLM evaluation library purpose-built for RAG pipelines. Provides metrics like **Faithfulness** and **Response Relevancy**.

**Ray Serve** — A production-grade serving framework for ML models. Listed alongside vLLM and TGI.

**ReAct** — "**Re**ason + **Act**" — the agent pattern the notebook uses: **think** about what to do → **call a tool** → **observe** the result → **repeat** until done. The dominant agent architecture.

**Response Relevancy** — A Ragas metric: "Does the answer actually address the question?"

**Retrieval (top-k)** — Pulling the `k` most similar documents from a vector DB. The notebook uses `k=2`.

**RLHF (Reinforcement Learning from Human Feedback)** — A finetuning technique using human preference data plus a reward model. The original recipe behind ChatGPT.

**Role (in chat messages)** — One of `"system"`, `"user"`, `"assistant"`, or `"tool"`. Tells the model who's "speaking" at each turn.

## S

**Semantic search** — Search by **meaning**, not exact keywords. "Affordable laptops" should match "budget notebooks." Built on embeddings + vector DBs.

**SFT (Supervised Fine-Tuning)** — Standard finetuning: feed the model (prompt, ideal answer) pairs and minimise the next-token loss. The simplest post-training method.

**SkyPilot** — A cloud-orchestration tool: pick the cheapest GPU across AWS/GCP/Azure and run your job there.

**Smolagents** — A lightweight (Hugging Face) agents library. Simpler than LangGraph or CrewAI.

**SmoothQuant** — Another quantization method. Smooths out activation outliers before quantizing.

**Speculative decoding** — An inference trick: a small "draft" model proposes the next few tokens, the big model only verifies them. ~2× speedup when the draft is good.

**Stream (streaming)** — Set `stream=True` to receive the model's response token-by-token instead of waiting for the full reply. Makes UIs feel snappy.

**Streamlit** — A Python library for rapid web UIs, similar to Gradio. Popular for LLM demos.

**System message** — The first `{role: "system", content: ...}` message — sets the model's persona, rules, and task context. Example from the notebook: `"You are a helpful assistant."`

## T

**Temperature** — A knob between 0 and 2 controlling output randomness. `0` = deterministic, picks the most-likely token every time. `0.7` (the notebook's default) = a good balance. `>1` = creative / chaotic.

**Tensor parallelism** — Splitting a large model across multiple GPUs so each holds a slice of every layer. Lets you run 70B+ models on consumer hardware.

**TGI (Text Generation Inference)** — Hugging Face's production inference server. Alternative to vLLM.

**TextGrad** — Same idea as DSPy: automatic prompt optimisation, gradient-style.

**TinyLlama** — A tiny (1.1B parameter) chat model trained for cheap demos. Used in the notebook's open-source example.

**Tokens** — The atomic units an LLM reads/writes. Roughly 1 token ≈ 4 characters of English. You pay per token (OpenAI) and budget per token (`max_tokens`).

**Tokenizer** — The function that splits text into tokens and back. Each model family has its own.

**Tool / Tool calling** — A function the agent can invoke. The agent's LLM outputs a structured call (`{"tool": "multiply", "args": {...}}`); your code executes it and feeds the result back. Same idea as **function calling**.

**Top-k** — In retrieval, the `k` most-similar items returned. In sampling, the `k` most-likely next tokens to consider.

**Transformers (library)** — Hugging Face's main library — provides `AutoTokenizer`, `AutoModelForCausalLM`, `pipeline`, `Trainer`. The notebook uses all of these.

**Trainer / TrainingArguments** — The Hugging Face classes that handle a finetuning loop (epochs, batch size, logging). Used in the `distilgpt2` finetune.

**TRL** — Hugging Face's Transformer Reinforcement Learning library — implements SFT, DPO, RLHF in a few lines.

**Throughput** — Requests served per second (across all users). Different from **latency** (one user's wait time). vLLM optimises throughput.

## U

**Unsloth** — A finetuning library that's 2–5× faster than vanilla `transformers` with the same code.

## V

**Vector database** — A database that stores embeddings and supports **similarity search**. Traditional DBs do exact match; vector DBs do "nearest neighbour." Examples: FAISS, Chroma, Pinecone, Milvus, Weaviate.

**vLLM** — A high-throughput LLM inference engine. Key features: **PagedAttention**, continuous batching, quantization, tensor parallelism, speculative decoding, prefix caching. ~24× faster than naive HuggingFace inference.

## W

**Weaviate** — Another vector database. Open-source, supports hybrid (keyword + vector) search.

**Weights & Biases (W&B)** — Experiment tracking, like Comet ML and MLflow.

## Z

**Zero-shot prompting** — Just asking the model directly with no examples. Relies entirely on what the model already knows. Works for simple, well-defined tasks (the notebook's sentiment-classification example).

---

[🔝 Back to top](#top)
