<a id="top"></a>
# Hybrid Search & Advanced Retrieval — Jargon Card

> **How to use.** Skim once (~6 min) before the notebook, then keep open as a side reference.
>
> **Companion:** read [`Hybrid_Search_Reading_Brief.md`](./Hybrid_Search_Reading_Brief.md) first.
>
> **Notebook:** `L11_and_L12_Hybrid_Search_&_Advanced_Retrieval.ipynb` — 64 cells, direct sequel to the Agentic RAG lecture. Needs a Groq key; corpus is 20 fictional company documents.

---

## A

**Abstention** — Deliberately returning "I don't know" instead of a confident wrong answer. The notebook's framing: Stack Overflow's *"no results found"* is a UX decision, not a bug. Implemented here as a score threshold below which the retriever reports `low_confidence`.

**`additionalProperties: false`** — Required on every object in a Groq strict-JSON schema. Omit it and the request is rejected.

**Andon cord** — Toyota's assembly-line pull-cord: any worker can stop the line when they spot a defect, *before* it propagates downstream. The notebook's analogy for the verifier — it halts the pipeline before the LLM synthesises an answer from garbage.

## B

**BM25 (Best Matching 25)** — The classic keyword-ranking algorithm. It scores a document by how many query terms it contains, weighted by how **rare** each term is across the corpus (IDF) and dampened by document length. Strength: exact identifiers and acronyms. Weakness: it has no idea "jailbreak" and "tricking the chatbot" mean the same thing.

**Bi-encoder** — Embeds the query and each document **independently**, then compares by cosine. Cheap and scalable — document embeddings are computed once and stored — but the model never sees query and document together. This is what "dense retrieval" means in this notebook.

## C

**Calibration** — Fitting a score threshold using labelled relevant/irrelevant pairs, so raw model scores become an actionable decision. Here: score 5 known-good and 5 known-bad pairs, then set the threshold midway between the worst positive and the best negative.

**Cross-encoder** — Feeds query **and** document into one model as a single input and outputs one relevance score. Far more accurate than a bi-encoder because the model can attend to interactions between the two — and unusable as a first-stage retriever, because scoring N documents costs N full forward passes.

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** — The reranker used here. Small (~90 MB), trained on MS MARCO, runs on CPU.

## D

**Dense retrieval** — Embedding-based semantic search. Synonym for bi-encoder retrieval in this context.

## H

**Hit@3** — The fraction of evaluation queries where the correct document appeared in the top 3. A blunt but honest yes/no-per-query metric.

**Hybrid search** — Running keyword (BM25) and dense retrieval, then fusing the two ranked lists. Motivation: the two fail in **complementary** ways, so their union fails less catastrophically on any single query type.

## I

**IDF (Inverse Document Frequency)** — "How rare is this term?" A term appearing in few documents gets a high weight; a term in every document gets ~0. This is why `INC-2847` — present in exactly one of 20 documents — dominates a BM25 score.

**Iterative retrieval** — Retrieve → verify → if insufficient, reformulate the query and retry, up to a hard cap. The notebook's reference: Perplexity issuing follow-up searches when initial results are weak.

## L

**Logit** — A raw, uncalibrated model score. Cross-encoder outputs are logits, not probabilities — comparable *within* one query's candidates, and only thresholdable across queries after calibration.

## M

**MRR (Mean Reciprocal Rank)** — Average of 1/rank of the correct document across queries. Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25, a miss scores 0. More sensitive than Hit@3 because it rewards *how high* the right answer landed.

**Multi-query retrieval** — Retrieve once per query variant, then fuse all the ranked lists with RRF.

## Q

**Query expansion / reformulation** — Asking the LLM to rewrite the user's query into several alternative phrasings with different vocabulary, retrieving for each, and fusing. A **vocabulary bridge**: the user says "chatbot tricks", the corpus says "prompt injection". Costs an LLM call plus N retrievals, so production systems gate it on low confidence rather than running it blindly.

## R

**`rank-bm25` / `BM25Okapi`** — The Python library and class implementing BM25 here. Requires you to tokenise documents and queries **the same way**.

**Reranking** — Re-scoring a shortlist with a heavier, more accurate model. Never applied to a full corpus — always to the 20–50 candidates a cheap retriever produced.

**RRF (Reciprocal Rank Fusion)** — Combining ranked lists using only **positions**, never scores: `RRF(d) = Σ 1/(k + rank_r(d))` with `k = 60` by convention. Scale-invariant by construction, which is the point: BM25 scores are unbounded and query-dependent while cosine sits in [−1, 1], so averaging them directly is fragile. Ranks are on the same scale by definition.

## S

**Self-correction RAG** — A verifier sitting **between retrieval and generation** that judges whether the retrieved evidence can actually answer the question. Distinct from the reflection in the previous lecture, which happened *after* evidence gathering and just before synthesis — this one is a pre-flight check on the retrieval itself.

**Shortlist** — The candidate set a cheap retriever hands to an expensive reranker. Typical production size 20–50. Bigger shortlist = better recall = more latency, roughly linear.

**Structured verdict** — The verifier's output shape: `{"verdict": "sufficient" | "insufficient", "reason": "..."}`. JSON, not prose, so the pipeline can branch on it reliably.

## T

**Tokenisation (BM25)** — Splitting text into terms. Here: `re.findall(r"[a-z0-9\-]+", text.lower())` — lowercase, keeping hyphens so `INC-2847` survives as **one** token. Tokenise documents and queries differently and BM25 silently stops matching.

## V

**Verifier** — An LLM given the question and the retrieved snippets, asked to rule `sufficient` or `insufficient`. Prompted to be **conservative**: topically related but not directly answering ⇒ insufficient. Costs roughly 50–200 ms; cheap insurance against confident nonsense.

**Vocabulary mismatch** — When the user's words and the corpus's words differ for the same concept ("clever prompts" vs "prompt injection"). The failure mode BM25 is worst at and query expansion exists to fix.

[🔝 Back to top](#top)
