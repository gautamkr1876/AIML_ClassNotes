<a id="top"></a>
# LlamaIndex & Advanced RAG — Jargon Card

> **How to use.** Skim once (~6 min) before the notebook, then keep open as a side reference.
>
> **Companion:** read [`LlamaIndex_RAG_Reading_Brief.md`](./LlamaIndex_RAG_Reading_Brief.md) first — it carries the punchline and the concept primers.
>
> **Notebook:** `Llamaindex_class.ipynb` — 82 cells. Needs `GROQ_API_KEY` in Colab Secrets (and, for the last section, a Google API key).

---

## A

**Agno** — A separate, lightweight agent framework used in the notebook's final section. Its selling point here is **persistent memory**: give an `Agent` a `SqliteDb` and `update_memory_on_run=True`, and it extracts durable facts about the user and stores them to disk.

## B

**BGE (`BAAI/bge-small-en-v1.5`)** — The embedding model used throughout. "Small" and English-only, downloaded from Hugging Face and run **locally on the Colab CPU** — so embedding costs nothing and sends no text to an API. Good default for a class demo; a larger model would score better on hard retrieval.

## C

**Chat engine** — `index.as_chat_engine()`. Like a query engine, but it **remembers the conversation**, so follow-ups such as "and what about the second one?" resolve correctly. Query engine = one-shot Q&A; chat engine = conversation.

**`chunk_overlap`** — How many characters of the previous chunk are repeated at the start of the next (50 here). Overlap stops a sentence that straddles a boundary from being lost to both chunks.

**`chunk_size`** — The target size of each chunk in characters (512 here). Small chunks retrieve precisely but lose context; large chunks carry context but dilute the embedding, so the match gets fuzzier.

**Chunking** — Splitting documents into retrievable pieces before embedding. **The most consequential knob in any RAG system** — it decides what can ever be retrieved as a unit.

## D

**Document** — LlamaIndex's container for one loaded source file: its text plus metadata. Produced by a reader, consumed by an index.

## E

**Embedding** — Text converted into a list of numbers such that similar meanings land close together. Both your document chunks and the user's question become embeddings; retrieval is finding the chunks whose vectors sit nearest the question's.

**Empty Response** — What a query engine returns when retrieval found nothing to work with. Seen in the notebook when the keyword index is asked about "reward points" — a real, visible failure worth understanding rather than skipping.

## G

**Graph store** — Where extracted entity-relationship triplets live. `SimpleGraphStore()` is the in-memory default; production alternatives are Neo4j and friends.

**Groq** — The inference provider serving `openai/gpt-oss-120b` here. Set once via `Settings.llm = Groq(model="openai/gpt-oss-120b")` and every query engine in the notebook uses it.

## I

**Index** — A searchable structure built over your documents. LlamaIndex offers several *kinds*, and the notebook builds four: **VectorStoreIndex** (similarity search — the default), **TreeIndex** (hierarchical summaries — for "summarise everything"), **SimpleKeywordTableIndex** (keyword lookup — exact terms), **KnowledgeGraphIndex** (entities and relationships — for "how is X connected to Y?").

**`include_embeddings=True`** — On a `KnowledgeGraphIndex`, also compute embeddings so graph retrieval can fall back on semantic similarity, not just exact entity matching.

## K

**KnowledgeGraphIndex** — Builds a graph by having the **LLM read every chunk and extract triplets**. Slow and expensive (one LLM call per chunk) but it captures relationships a vector index cannot. The notebook prints its own warning: *"This took longer because the LLM had to extract entities and relations from each chunk."*

## L

**LlamaIndex** — An open-source framework for connecting LLMs to your own data. It supplies the whole RAG pipeline as composable parts: readers → documents → chunking → embedding → index → query engine.

## M

**`max_triplets_per_chunk`** — Cap on how many relationships the LLM may extract per chunk (5 here). Higher = richer graph, more LLM cost, more noise.

## N

**Node** — A chunk after splitting, with its own embedding, metadata, and links back to its source document. Documents are what you *load*; nodes are what you *retrieve*.

**`node_parser`** — The component that turns documents into nodes. Set globally: `Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)`.

**`num_children`** — On a `TreeIndex`, how many child nodes each parent summarises (3 here). It sets the branching factor of the summary tree.

## Q

**Query engine** — `index.as_query_engine()`. The interface you ask questions through. Its flow: question → retrieve relevant chunks → paste them into a prompt → LLM → answer.

**QueryEngineTool** — A query engine wrapped so an agent can call it as a tool, with a `name` and a `description`. The description is what a planner reads to decide *which knowledge source* answers which part of a question — it's the bridge from RAG to agentic RAG. **Good tool descriptions are the main lever on tool-selection quality.**

## R

**RAG (Retrieval-Augmented Generation)** — Retrieve relevant text first, then generate an answer grounded in it. Fixes two LLM weaknesses at once: stale knowledge, and no access to your private documents.

**`rel_map`** — `graph_store.get_rel_map()` — the dictionary of extracted relationships, for inspecting what the graph actually captured. The notebook's sample: `Andy jassy --[Succeeded]--> Jeff bezos`.

## S

**`Settings`** — LlamaIndex's global configuration object. Set `Settings.llm`, `Settings.embed_model`, `Settings.node_parser` once and every index and engine created *afterwards* picks them up. ⚠️ It applies at construction time — changing it later does not retroactively alter an index you already built.

**SimpleDirectoryReader** — Loads every file in a folder into `Document` objects. One line, and it's where 90% of RAG demos begin.

**SimpleKeywordTableIndex** — Extracts keywords per chunk and looks up by exact term. Fast and cheap, but brittle: ask about "reward points" when the document says "loyalty points" and you get nothing.

**SimpleVectorStore** — LlamaIndex's default, in-memory vector store. **Not** Pinecone, Chroma, FAISS or Weaviate. It lives in the Colab runtime's RAM, which means: restart the runtime and your index is gone.

**`source_nodes`** — The chunks a query engine actually retrieved, attached to the response. Reading them is how you debug RAG: a wrong answer from good sources is a generation problem; a wrong answer from irrelevant sources is a retrieval problem.

**SqliteDb** — Agno's on-disk store (`agent_memory.db`). The difference between memory that survives a restart and memory that doesn't.

**StorageContext** — The bundle telling an index *where* to put its data — vector store, graph store, doc store.

## T

**TreeIndex** — Builds a hierarchy by having the LLM summarise chunks bottom-up, so parents summarise children. Strong at "give me an executive summary of everything", weak at pinpoint factual lookup (the detail is summarised away).

**Triplet** — A relationship as (subject, relation, object), e.g. *(Andy Jassy, Succeeded, Jeff Bezos)*. The atom of a knowledge graph.

## U

**`update_memory_on_run=True`** — Agno's switch that makes the agent extract durable facts from each turn and persist them. In the notebook's run it turned one sentence into three separate memories, each tagged with topics.

**UserMemory** — Agno's stored-fact record: the memory text, an id, topics, `user_id`, the original input, and timestamps. `user_id` is what keeps different users' memories apart.

[🔝 Back to top](#top)
