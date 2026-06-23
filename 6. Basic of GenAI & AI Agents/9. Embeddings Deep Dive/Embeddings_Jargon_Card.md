<a id="top"></a>
# Embeddings Deep Dive — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Embeddings_Reading_Brief.md`](./Embeddings_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `L9_(Jun 11th) - Embeddings Deep Dive.ipynb` in this folder.
> **Sibling:** builds on `../8. Retrieval-Augmented Generation (RAG) Introduction/` — embeddings are the backbone of RAG retrieval.

---

## A

**all-MiniLM-L6-v2** — The small, fast **sentence transformer** used as the default bi-encoder. Produces **384-dimensional** embeddings. The notebook's workhorse for demos and the pre-trained baseline in the fine-tuning experiments.

**all-mpnet-base-v2** — A higher-quality (but slower) sentence transformer producing **768-dim** embeddings. Mentioned as the "use this for better quality" alternative.

**Anchor** — In a **triplet**, the reference item (the query). Training pulls the **positive** closer to the anchor and pushes the **negative** away. Triplet = (anchor, positive, negative).

## B

**BCE Loss (Binary Cross-Entropy)** — The loss used to fine-tune the **cross-encoder**. It treats relevance as a yes/no classification: input a (query, document) pair, output a probability 0–1, and penalize the gap to the true label (1 = relevant, 0 = not). Contrast with **triplet loss** (used for the bi-encoder).

**BERT** — The 2018 transformer that revolutionized NLP but was built for *token-level* tasks, not sentence embeddings. Comparing two sentences required feeding them in together (slow); averaging its token vectors gave poor embeddings. The limitation that motivated **SBERT**.

**Bi-encoder (dual encoder)** — A model that encodes query and document **independently** into vectors, compared by cosine/dot. Fast and scalable (documents pre-encoded) → used for **Stage 1** retrieval. The notebook fine-tunes one with **triplet loss**. Contrast with **cross-encoder**.

## C

**Contrastive loss** — A loss for positive/negative *pairs* (binary), pulling positives together and pushing negatives apart. Listed as an alternative to triplet loss.

**CosineSimilarityLoss** — A loss that regresses embeddings toward a target similarity score. Listed as an alternative bi-encoder loss.

**Cosine similarity** — Similarity by the **angle** between two vectors, ignoring length (1 = same direction, 0 = perpendicular/unrelated). The default metric for sentence-transformer embeddings. Pitfall: assumes vectors are normalized; if not, magnitude leaks in.

**Cross-encoder** — A model that feeds query+document **together** (`[CLS] query [SEP] doc [SEP]`) and outputs one relevance score with full token-level attention. Accurate but slow (no pre-computation) → used only to **rerank** a small candidate set. The notebook fine-tunes `ms-marco-MiniLM-L-6-v2` with **BCE loss**.

## D

**Dimension** — The length of an embedding vector (e.g., 384, 768, 1536, 3072). More dimensions can capture more nuance but cost more storage/compute. Each dimension is one "aspect of meaning."

**Distance metric** — The rule for deciding which vectors count as "close." The notebook compares four: **cosine** (angle), **dot product** (angle + magnitude), **Euclidean/L2** (straight-line), **Manhattan/L1** (city-block). Your choice changes which documents get retrieved.

**Dot product** — Multiply matching dimensions and sum. Captures **direction *and* magnitude**, so longer vectors "vote" harder. Unbounded (−∞ to +∞), so raw scores are hard to interpret. Equals cosine similarity *when vectors are normalized*.

## E

**Embedding** — A fixed-length vector of numbers representing the *meaning* of text, such that similar meanings sit close together in vector space. The foundation of semantic search, clustering, and RAG retrieval.

**Euclidean distance (L2)** — Straight-line ("ruler") distance between two points; 0 = identical, larger = farther. Sensitive to both direction and magnitude. Pitfall: in high dimensions everything starts looking equidistant (**curse of dimensionality**).

## F

**FAISS** — Facebook's library for fast vector similarity search. The notebook builds a `IndexFlatIP` (inner-product) index over normalized embeddings — inner product on normalized vectors = cosine similarity.

**Fine-tuning (embeddings)** — Continuing to train a pre-trained embedding model on **your** domain's data so the vector space reflects your actual semantic relationships. The notebook's Experiment B: pre-trained → fine-tuned, measured by retrieval metrics. Fixes the **domain mismatch** problem.

## H

**Hard negative** — A document that *looks* related to a query (shares keywords/topic) but is actually wrong — the challenging case that teaches a reranker fine-grained distinctions. Example: for "Where was Apple founded?", the **Fuji apple** cultivar doc is a hard negative. Contrast with **soft (easy) negative** (obviously unrelated).

**Hits@K (Success@K)** — Did *at least one* relevant document appear in the top K? Binary per query (1/0), averaged across queries. Good when one good result is enough (Q&A). Ignores how many relevant docs and their rank.

## I

**InputExample** — The sentence-transformers data container. For the bi-encoder it holds a triplet `texts=[anchor, positive, negative]`; for the cross-encoder it holds `texts=[query, doc]` plus a `label`.

## L

**L1 / L2** — Shorthand for **Manhattan** distance (L1, sum of absolute differences) and **Euclidean** distance (L2, straight-line). The numbers refer to the norm used.

## M

**Magnitude** — A vector's length (distance from origin). **Cosine** ignores it; **dot product** and **Euclidean** use it. Whether magnitude carries meaning (e.g., confidence) depends on your model.

**Manhattan distance (L1)** — "Taxicab" distance: sum of absolute differences per dimension, like walking city blocks. Fast (no squaring) and robust to outliers, but ignores diagonal relationships. Less common in modern embedding models.

**MAP@K (Mean Average Precision)** — Rewards placing relevant docs *higher* in the ranking, averaging precision at each relevant position. The standard reranking metric on the MTEB benchmark. Treats relevance as binary.

**Mean pooling** — Averaging all of a model's token vectors to get one sentence vector. Fast, but on vanilla BERT it gives poor embeddings (worse than 2014-era GloVe) — which is why **SBERT** was needed.

**MRR@K (Mean Reciprocal Rank)** — How quickly the *first* relevant result appears: 1/rank of the first hit (pos 1 → 1.0, pos 2 → 0.5…), averaged over queries. Good for "find the one right page." Ignores later relevant docs.

**ms-marco-MiniLM-L-6-v2** — The pre-trained **cross-encoder** the notebook uses for reranking and then fine-tunes with BCE loss.

**MultipleNegativesRankingLoss** — A bi-encoder loss that uses other items in the batch as negatives; efficient for large batches. Listed as a triplet-loss alternative.

## N

**NDCG@K (Normalized Discounted Cumulative Gain)** — Ranking quality when relevance comes in **grades** (highly/somewhat/not relevant), with a logarithmic discount for lower positions; 1.0 = perfect order. The standard graded-relevance metric on MTEB.

**Negative** — A document that should score *low* for a query. **Soft** = obviously unrelated; **hard** = deceptively similar. Quality of negatives drives reranker training.

**NLI (Natural Language Inference)** — The classic SBERT training data: sentence pairs (premise, hypothesis) labeled **entailment / contradiction / neutral**. Datasets: **SNLI** (~570K pairs), **MultiNLI** (~430K pairs).

**Normalization** — Scaling a vector to unit length (magnitude 1). Key fact from the notebook: **after normalization, cosine similarity and dot product are identical.** Many embedding models normalize by default.

## O

**OpenAI embeddings** — Hosted embedding API. Models: `text-embedding-3-small` (1536-dim, cost-effective) and `text-embedding-3-large` (3072-dim, highest quality). High quality, no local compute, but pay-per-token and *not* fine-tunable.

## P

**Positive** — In a triplet/pair, the document that *correctly* matches the query (the right answer). Triplet training pulls it toward the **anchor**.

**Precision@K** — Of the K results you returned, what fraction were relevant? ("Of what I returned, how much was useful?") Use when irrelevant results are costly. Ignores how many relevant docs exist overall.

## R

**Recall@K** — Of all the relevant documents that exist, what fraction did you retrieve in the top K? ("Of what I should have found, how much did I find?") Use when *missing* a relevant doc is costly (legal, medical, RAG). You could get perfect recall by returning everything.

**Reranking** — Stage 2 of two-stage retrieval: re-scoring the bi-encoder's candidates with a **cross-encoder** for precision. The notebook evaluates `rerank_recall@k` / `rerank_mrr` to show reranking's gain.

## S

**SBERT (Sentence-BERT)** — Reimers & Gurevych's 2019 (EMNLP, UKP Lab) fix for BERT's sentence-embedding weakness. It uses a **Siamese** architecture to make BERT output meaningful, *independent* sentence vectors you can compare with cosine similarity. Turned "65 hours to find the most similar pair among 10K sentences" into seconds. The basis of the `sentence-transformers` library.

**Semantic search** — Retrieval by meaning (vector closeness) rather than keyword matching. The notebook demos it with cosine similarity over a small document set.

**Sentence transformer** — A model (from the `sentence-transformers` library) that outputs sentence-level embeddings directly. Local, free, fine-tunable; the basis of bi-encoders. E.g., `all-MiniLM-L6-v2`.

**Siamese architecture** — Two (or more) copies of the *same* network with *shared weights*, each encoding one input, whose outputs are then compared. How SBERT trains BERT to produce comparable sentence embeddings. (Same idea as the CV Siamese networks in Module 5.)

**Soft (easy) negative** — A document obviously unrelated to the query — trivial for the model to reject, so it teaches little. Contrast with **hard negative**.

## T

**text-embedding-3-small / -large** — OpenAI's current embedding models (1536-dim / 3072-dim). See **OpenAI embeddings**.

**Triplet loss** — The loss used to fine-tune the **bi-encoder**. Given (anchor, positive, negative), it pushes the positive closer to the anchor than the negative by at least a **margin** (the notebook uses cosine distance, `margin=0.5`). Teaches *relative ordering*. Contrast with **BCE loss** (cross-encoder, absolute scoring).

**Two-stage retrieval** — Bi-encoder retrieves many candidates fast → cross-encoder reranks to the precise few. The production RAG pattern; the notebook fine-tunes *both* stages.

## V

**Vector arithmetic** — The property that semantic relationships are encoded as directions in embedding space, so `king − man + woman ≈ queen`. Evidence that embeddings capture meaning geometrically.

[🔝 Back to top](#top)
