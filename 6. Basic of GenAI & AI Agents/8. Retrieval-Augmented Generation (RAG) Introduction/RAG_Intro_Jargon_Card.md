<a id="top"></a>
# RAG (Introduction) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`RAG_Intro_Reading_Brief.md`](./RAG_Intro_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `L8 (Jun 9th)_ Retrieval_Augmented_Generation_(RAG)_Introduction.ipynb` in this folder.

---

## A

**ANN (Approximate Nearest Neighbor) search** — Finding vectors that are *close enough* to a query vector very fast, instead of finding the exact closest ones slowly. The trade you make: give up a tiny bit of accuracy to go from "seconds per query" to "milliseconds." The notebook shows why you need it — 10M docs × 768 dims = 7.68 billion operations for one exact (brute-force) query. The popular algorithm is **HNSW**.

**Augment** — The "A" in RAG: after retrieving relevant chunks, you *augment* (enrich) the prompt by pasting them in as context before the user's question. "Retrieve → **Augment** → Generate."

## B

**BAAI/bge-base-en-v1.5** — The specific **bi-encoder** embedding model the notebook uses for retrieval. Produces 768-dimensional vectors; a strong general-purpose English retriever.

**Bi-encoder** — An embedding model that encodes the query and each document **separately** into vectors, then compares them by distance. Because documents can be encoded once and stored ahead of time, search is fast and scales to millions — perfect for **Stage 1** retrieval. The trade-off: it never sees query and document together, so it can miss subtle relevance. Contrast with **cross-encoder**.

**Brute-force search** — Comparing the query vector against *every* stored vector, then sorting. Exact but unscalably slow (the notebook's 7.68-billion-ops example). The problem **ANN/HNSW** solves.

## C

**Chonkie** — The Python chunking library the notebook uses (`pip install chonkie[viz]`). Provides `TokenChunker`, `SemanticChunker`, `RecursiveChunker`, `LateChunker`, `SlumberChunker` and a `Visualizer` that color-codes chunks so you can *see* where splits land.

**Chunk** — A small segment a document is split into before embedding. Retrieval works over chunks, not whole documents. Chunk quality drives RAG quality: too big = noisy "averaged" embedding; too small = no standalone meaning; too long = the LLM's "lost in the middle" problem.

**Chunking** — The preprocessing step of breaking documents into chunks. The notebook teaches **6 strategies** (fixed, overlap, semantic, recursive, late, LLM-based). The recurring lesson: "when RAG fails, it's often the chunks, not the retriever."

**ChromaDB** — The open-source **vector database** the notebook uses (`pip install chromadb`). Stores each chunk's text + embedding (+ optional metadata) and runs semantic search over them, using **HNSW** indexing under the hood.

**ColBERT v2** — A state-of-the-art cross-encoder/late-interaction model mentioned as especially good for reranking. (The notebook's actual reranker is `ms-marco-MiniLM-L-6-v2`.)

**Context window** — The maximum text (in **tokens**) an LLM can read at once. Finite, so you can't paste a whole knowledge base into the prompt — you retrieve only the top few chunks. Also why chunk size matters.

**Cosine similarity** — A measure of how aligned two vectors are by **angle** (1 = same direction, 0 = unrelated). The default similarity used in semantic search; "distance" in ChromaDB is its inverse (lower distance = more similar).

**Cross-encoder** — A model that feeds the query and a document **together** (`[CLS] query [SEP] doc [SEP]`) and outputs one relevance score, computing attention between every query and document token. Far more accurate than a bi-encoder but can't pre-compute, so it's slow — used only to **rerank** a small candidate set in **Stage 2**. The notebook uses `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## E

**Embedding** — A fixed-length list of numbers (a vector) representing a piece of text's *meaning*, such that similar meanings produce nearby vectors. RAG embeds both chunks and the query so it can find chunks "close" to the query. The notebook's models output 384-dim (`all-MiniLM-L6-v2`) or 768-dim (`BAAI/bge-base-en-v1.5`) vectors.

**Embedding model** — The model that turns text into embeddings. The notebook uses `all-MiniLM-L6-v2` (fast, 384-dim) early on and `BAAI/bge-base-en-v1.5` (768-dim) for the production retriever.

## F

**Fixed-size chunking** — Strategy 1: split at an exact token count (`chunk_size=128`, `chunk_overlap=0`). Predictable and fast, but blindly cuts mid-sentence. The baseline everything else is compared to.

## G

**Generate** — The "G" in RAG: the LLM produces the final answer grounded in the retrieved+augmented context. The notebook uses `gpt-4o-mini` at `temperature=0` with a strict "use ONLY the documents" prompt.

**Grounding** — Forcing the LLM to answer *only* from the retrieved context, not its training memory. Enforced by prompt rules ("every fact MUST come from the documents; if they don't answer it, say so"). The core anti-hallucination mechanism of RAG.

**gpt-4o-mini** — The small, cheap OpenAI model used throughout for generation, HyDE, and LLM-based chunking.

## H

**Hallucination** — When an LLM produces confident but false content, often by filling a knowledge gap. One of the four problems RAG exists to fix (alongside knowledge cutoff, private data, and context limits).

**HNSW (Hierarchical Navigable Small World)** — The most popular **ANN** algorithm, used by ChromaDB, Pinecone, and Weaviate. Builds a multi-layer graph of vectors so search hops toward the answer in roughly logarithmic time instead of checking everything. How vector DBs stay fast at millions of vectors.

**HyDE (Hypothetical Document Embeddings)** — A trick to close the **semantic gap**: instead of embedding the user's short query, ask an LLM to *write a hypothetical answer*, then embed and search with *that*. Because the generated text looks like real documents (long, technical), its vector lands closer to real document vectors. Costs ~1 extra LLM call.

## K

**Knowledge cutoff** — The date an LLM's training data ends; it knows nothing after it. One of the four limitations RAG addresses by fetching fresh external info at query time.

**k / top-k** — How many results you ask for. The notebook uses `k_initial=20` (bi-encoder candidates) and `k_final=3–5` (after reranking) — the classic "retrieve many, keep few" funnel.

## L

**LangChain** — A framework that wraps embedding models, vector stores, and retrievers behind a common interface. The notebook uses `Chroma.from_documents(...)` and `vectorstore.as_retriever(...)` to build the production retriever.

**Late chunking** — Strategy 5: embed the **whole document first**, *then* split, so each chunk's embedding carries full-document context. Best for long, cross-referencing docs (books, contracts); computationally expensive.

**LLM-based (agentic) chunking** — Strategy 6: an LLM (`gpt-4o-mini` via Chonkie's `SlumberChunker`) decides where to split. Highest quality (each chunk is a complete thought), slowest and most expensive.

**Lost in the middle** — The tendency of LLMs to overlook information buried in the middle of a long context. A reason not to stuff huge chunks into the prompt.

## M

**Metadata** — Structured tags attached to each chunk (source, section, timestamp, `chunk_index`). Enables **filtered search** — "find similar chunks, but only where `chunk_index >= 50`."

## O

**Overlap (chunk_overlap)** — Strategy 2: let adjacent chunks share some tokens (`chunk_overlap=32`, ~25%) so context isn't lost at chunk boundaries. The notebook calls it "almost always worth it." Costs extra storage (repeated text).

## R

**RAG (Retrieval-Augmented Generation)** — A framework (Lewis et al., 2020) that fetches relevant external info and feeds it to an LLM so the answer is grounded in real documents, not just the model's memory. Three steps: **Retrieve → Augment → Generate**. Analogy: an open-book exam.

**Recursive chunking** — Strategy 4: split along a document's **structure** (headers → sections → subsections), preserving hierarchy. Best for Markdown, PDFs, and documentation.

**Reranking** — **Stage 2** of retrieval: take the bi-encoder's candidate list and re-score it with a slower, more accurate **cross-encoder**, keeping only the best few. Improves precision before sending context to the LLM.

**Retrieve** — The "R" in RAG: fetch the most relevant chunks for a query from the vector database.

**Retriever** — The component that does retrieval. In LangChain, `vectorstore.as_retriever(search_kwargs={"k": 20})` — a bi-encoder + HNSW search returning the top-20 candidates.

## S

**Semantic chunking** — Strategy 3: use an embedding model to split at **topic changes** rather than fixed counts, so each chunk is one coherent idea. The notebook uses `SemanticChunker` (`minishlab/potion-base-32M`, `threshold=0.7`). Best for unstructured prose (essays, blogs).

**Semantic gap** — The mismatch between short, casual queries ("load a pretrained model") and long, formal documents. Their vectors end up less similar than they should be, hurting retrieval. **HyDE** is the notebook's fix.

**Semantic search** — Finding documents by *meaning* (vector closeness) rather than keyword matching. The thing a vector database is built to do.

## T

**Token** — The unit text is split into for models (~¾ of a word). Chunk sizes and context windows are measured in tokens; the notebook uses GPT-2's tokenizer as a reference.

**Two-stage retrieval** — The production pattern: **Stage 1** bi-encoder + HNSW retrieves ~20 candidates fast; **Stage 2** cross-encoder reranks them to the best 3–5. Combines the bi-encoder's speed with the cross-encoder's precision. Notebook timings: ~10–50 ms + ~100–200 ms ≈ 150–250 ms total.

## V

**Vector database** — A database built to store high-dimensional vectors and find the ones most similar to a query vector fast (using **ANN/HNSW**). Unlike SQL (exact matches on structured fields), it does semantic similarity. The notebook uses **ChromaDB**.

[🔝 Back to top](#top)
