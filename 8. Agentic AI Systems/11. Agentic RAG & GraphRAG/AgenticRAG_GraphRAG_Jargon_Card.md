<a id="top"></a>
# Agentic RAG & GraphRAG — Jargon Card

> **How to use.** Skim once (~6 min) before the notebook, then keep open as a side reference.
>
> **Companion:** read [`AgenticRAG_GraphRAG_Reading_Brief.md`](./AgenticRAG_GraphRAG_Reading_Brief.md) first.
>
> **Notebook:** `classroom_agentic_rag_and_graph_rag.ipynb` — 37 cells. Every credential is **optional**: no Groq key → deterministic offline fallbacks; no Neo4j → an in-memory NetworkX graph. The notebook runs end to end either way.

---

## A

**Agentic RAG** — Retrieval where an **LLM plans the retrieval itself** instead of running one fixed search. The architecture shift the notebook names explicitly:

```
plain RAG:    Question → Vector Search → Documents → Answer
agentic RAG:  Question → Agent → Plan → Multiple Tools → Evidence → Reflect → Answer
```

**`all-MiniLM-L6-v2`** — The sentence-embedding model used here (via `sentence-transformers`). Small, fast, runs locally on CPU, 384 dimensions. Loaded **lazily** — only downloaded the first time an embedding is actually needed.

**Aura (Neo4j Aura)** — Neo4j's hosted cloud service. AuraDB Free is the zero-cost tier. ⚠️ Its password is displayed **exactly once** at creation; the username is often the instance id (e.g. `df1b8ee7`), *not* the literal `neo4j`.

## C

**Cosine similarity** — How aligned two vectors are. Because `embed_texts` passes `normalize_embeddings=True`, every vector has length 1, so cosine similarity reduces to a plain **dot product** — which is why the code is just `matrix @ query_vec`. A nice trick: normalise once, then similarity is a single matrix multiply.

**Cypher** — Neo4j's query language for graphs. Where SQL says `JOIN`, Cypher draws the shape you want: `MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee)`.

## D

**Deterministic fallback** — The notebook's offline mode. Without a Groq key, `LLM.complete()` returns pre-written canned JSON (`_FALLBACK_PLAN`, `_FALLBACK_SQL_*`, `_FALLBACK_ANSWER`) so the pipeline still runs and every output is reproducible. A genuinely good teaching-notebook pattern — and a reminder that a canned plan is not a *generated* plan.

**Decomposition** — Breaking one hard question into ordered subquestions, each answerable by a single tool. The planner's entire job.

## E

**Edge** — A connection between two nodes, carrying a relationship type. This notebook's edge types: `REPORTS_TO`, `PART_OF`, `HAS_SKILL`, `OWNED_BY`, `WORKS_ON`, `ABOUT`.

**Evidence buffer** — The accumulating dictionary of results from executed tools. Each tool reads what earlier tools deposited (`evidence.get("projects", [])`) and adds its own. It's the shared state of the retrieval loop.

**Entity** — A thing in the graph: an employee, department, skill, project, document. Entities are nodes; relationships are edges.

## G

**GraphRAG** — RAG where retrieval walks a **graph of entities and relationships** instead of (or alongside) ranking text chunks by similarity. It exists for questions whose answer is a *path*, not a passage.

**Graph traversal** — Following edges from node to node: *Project → (WORKS_ON) → Employee → (REPORTS_TO) → Manager → filter by department*. A vector index cannot do this; there is no edge to follow in an embedding.

## I

**In-memory graph** — The NetworkX-backed `InMemoryOrgGraph`, used whenever Neo4j isn't configured. Same conceptual model — nodes, typed edges, node attributes — without a server. Perfect for learning; no persistence, no concurrency, no Cypher.

## L

**Lazy loading** — Deferring an expensive import or download until first actual use. `_get_embed_model()` caches the model in a module-level global and only constructs it on the first call.

## M

**Multi-hop question** — A question needing several linked steps to answer. The notebook's running example is a **four-hop** question: *Which engineering managers have team members working on projects involving LLM security?* → Document → Project → Employee → Manager → Department.

## N

**NetworkX** — A Python library for building and traversing graphs in memory. `nx` here holds the whole fictional org.

**Neo4j** — A graph database: nodes, typed relationships, properties, queried with Cypher. Optional in this notebook.

**Node** — A single entity in the graph, with a type and attributes (e.g. an employee node carries `name`, `title`, `department`, `is_engineering`).

## P

**Planner** — The LLM role that reads the question and emits an ordered list of subquestions, each tagged with a tool (`sql`, `vector`, or `graph`). Crucially, **the planner does not execute anything** — at plan time those tool names are just labels.

**Property graph** — A graph model where both nodes and edges can carry arbitrary key-value properties. `is_engineering=1` on an employee node is a property, and it's what the final filter tests.

## R

**Recursive CTE** — A SQL `WITH RECURSIVE` query that walks a self-referencing table repeatedly — the SQL way to traverse a hierarchy (e.g. everyone under the CTO). Notable because it shows SQL *can* do traversal, just far more awkwardly than a graph query.

**Reflection** — The step after executing the plan where the system asks "is this evidence sufficient to answer?" If not, it loops or expands. In this run it reported `sufficient = True` on the first pass.

**Route → Execute → Reflect → Synthesize** — The four-stage loop that runs *after* planning. Route: pick the tool for this subquestion. Execute: run it, deposit results in the evidence buffer. Reflect: is this enough? Synthesize: hand everything to the LLM to write the final answer.

## S

**Semantic search** — Embedding-based retrieval: embed the question, embed the documents, return the closest. Excellent at *"find text about X"*; structurally incapable of *"join, filter, traverse, aggregate"*.

**Synthesis** — The final LLM call that turns the collected evidence into prose. Note what it is **not**: it is not retrieval, and it can only be as good as the evidence handed to it.

**Subquestion** — One step of the plan: `{"id": "q1", "tool": "vector", "text": "Identify projects whose documents describe LLM security work."}`.

## T

**Text2SQL** — Having the LLM write a SQL query for a natural-language question, then executing it. The notebook keeps fallback SQL for three specific question shapes (CTO hierarchy, budget threshold, reports-to-a-named-manager) plus a safe default.

**Tool** — One retrieval capability the planner may select. Three here: **`sql`** (structured filtering and aggregation), **`vector`** (semantic text search), **`graph`** (relationship traversal). Each answers a different *kind* of question — that's the whole point of having three.

**Top-k** — Keep only the k best-scoring results. `cosine_topk(query_vec, matrix, k=3)` returns the three nearest documents with their scores.

**Traditional / naive RAG** — One embed, one top-k lookup, one generation. The notebook's diagnosis: *"Traditional RAG works when the answer is already written somewhere in the documents. It starts struggling when the answer has to be calculated by joining information, filtering it, traversing relationships, and aggregating results."*

[🔝 Back to top](#top)
