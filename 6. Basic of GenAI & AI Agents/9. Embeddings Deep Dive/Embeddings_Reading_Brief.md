<a id="top"></a>
# Embeddings Deep Dive — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~25 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Embeddings_Jargon_Card.md`](./Embeddings_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebook:** `L9_(Jun 11th) - Embeddings Deep Dive.ipynb` (~162 cells — large, which is why this pre-read exists).
> **Sibling:** this is the deep dive behind the retrieval used in `../8. Retrieval-Augmented Generation (RAG) Introduction/`.

---

## 🎯 30-second TL;DR

An **embedding** turns text into a list of numbers (a vector) so that *similar meanings land close together* — which lets a computer do "find similar" by doing "find nearby." This notebook goes deep on the whole embedding stack:

> **What embeddings are → how to measure closeness (4 distance metrics) → where they come from (SBERT, providers) → bi-encoder vs cross-encoder → fine-tune both on your domain → measure with retrieval metrics**

The headline experiment: take pre-trained retrievers, **fine-tune them on a domain dataset** (human-rights documents), and watch retrieval quality improve — measured properly with Recall@K and MRR, tracked in Opik. The recurring truth: **general-purpose embeddings are a starting point, not the finish line** — real systems fine-tune to their own data.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **What embeddings are** — text → vectors; similar meaning → similar vector.
2. **Similarity & visualization** — cosine similarity, heatmaps, 2D/3D PCA plots, a tiny semantic search.
3. **Word-embedding intuition** — `king − man + woman ≈ queen` (meaning has geometry).
4. **Distance metrics** — cosine, dot product, Euclidean (L2), Manhattan (L1); the role of **normalization**.
5. **Ways to generate embeddings** — local sentence transformers vs hosted APIs (OpenAI, Cohere, Voyage…).
6. **Bi-encoder vs cross-encoder** — the two retrieval architectures; the two-stage pipeline.
7. **Why fine-tuning matters** — domain mismatch and task mismatch.
8. **SBERT history & training** — why BERT alone failed; Siamese architecture; NLI data; loss functions.
9. **Experiments** — evaluate pre-trained bi/cross-encoders, then fine-tune (triplet loss + BCE loss).
10. **Retrieval metrics primer** — Precision/Recall/Hits/MRR/MAP/NDCG @K.
11. **Hard-negative mining** — why the *quality* of negatives drives reranker training.
12. **Final RAG** — fine-tuned bi-encoder + cross-encoder + GPT-4o-mini.

---

## 🧠 The big idea — meaning becomes geometry

Computers can't compare *meaning* directly, but they're great at comparing *numbers*. An embedding model is trained so that the **distance between two vectors mirrors the difference in meaning between two texts.** Once meaning is geometry, every hard language task becomes an easy geometry task: "most similar document" = "nearest vector," "cluster these" = "group nearby points," "is this relevant?" = "are these vectors close?"

**The transferable analogy: a map of meaning.** On a real map, cities with similar locations have nearby coordinates, and you can do arithmetic on positions ("go 3 km north"). Embeddings are a map where *concepts* are the cities: "cat" and "kitten" sit close, "cat" and "thermodynamics" sit far. The famous proof that this map is real is **vector arithmetic** — `king − man + woman ≈ queen` — the *direction* from "man" to "woman" is the same direction as "king" to "queen," because the model encoded the gender relationship as a consistent geometric shift.

Two consequences run through the whole notebook. First, **how you measure distance on this map matters** (angle vs straight-line vs city-block give different "nearest" results). Second, **a generic world map is too coarse for your neighborhood** — a model trained on Wikipedia doesn't know that in *your* bank "SWIFT" means payment messaging, not speed — so you **fine-tune** to redraw the map around your domain.

---

## 📖 Core concept primers

Seven primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, a concrete detail from the notebook, and why it matters here.

### 1. What an embedding is

> **🪜 Mental model:** a GPS coordinate for meaning — similar sentences get nearby coordinates.

An **embedding** is a fixed-length vector (the notebook's models output 384, 768, 1536, or 3072 numbers) where each dimension captures some aspect of meaning, and *closeness = similarity*. You generate them with a model (`model.encode(texts)`), then compare with a distance metric. **Why it matters here:** everything downstream — search, clustering, RAG retrieval, reranking — is just operations on these vectors. The notebook opens by embedding a handful of sentences and showing, via a cosine-similarity heatmap and a 2D/3D PCA plot, that semantically related sentences really do cluster together.

### 2. The four distance metrics (and normalization)

> **🪜 Mental model:** different rulers. Cosine measures the *angle* between arrows; Euclidean measures the *straight-line* gap between their tips.

The metric decides which vectors count as "close," and that changes what your system retrieves. The notebook compares four using toy vectors (Strawberry `[4,0,1]`, Blueberry `[3,0,1]`, Broccoli `[8,2,9]`):

| Metric | Measures | Higher/Lower = closer | Watch out for |
|---|---|---|---|
| **Cosine similarity** | Angle (direction only) | Higher (→1) | Assumes normalized vectors |
| **Dot product** | Direction + magnitude | Higher | Unbounded; longer vectors win |
| **Euclidean (L2)** | Straight-line distance | Lower | Curse of dimensionality |
| **Manhattan (L1)** | City-block distance | Lower | Ignores diagonal relationships |

**Why it matters here:** sentence-transformer models are trained for **cosine** similarity, so that's usually the right pick. The key insight the notebook proves with code: **once you normalize vectors to unit length, cosine similarity and dot product become identical** — magnitude stops mattering. That's why so many models normalize by default.

### 3. Where embeddings come from (local vs hosted)

> **🪜 Mental model:** cook at home vs order delivery — local models are free and customizable but need setup; APIs are instant and high-quality but cost per token and can't be tuned.

Two routes: **local sentence transformers** (the `sentence-transformers` library — `all-MiniLM-L6-v2` at 384-dim, `all-mpnet-base-v2` at 768-dim — free, private, **fine-tunable**) and **hosted APIs** (OpenAI `text-embedding-3-small` at 1536-dim / `-large` at 3072-dim, plus Cohere, Voyage, Google, Mistral — top quality, zero setup, but pay-per-token and *not* fine-tunable). **Why it matters here:** the notebook builds its whole fine-tuning story on local sentence transformers precisely *because* they can be fine-tuned; it uses OpenAI only for final generation.

### 4. Bi-encoder vs Cross-encoder

> **🪜 Mental model:** a recruiter. The bi-encoder skims thousands of résumés fast; the cross-encoder interviews the top few in depth. You use both.

A **bi-encoder** encodes query and document **separately** into vectors compared by cosine/dot — documents are pre-encoded, so search is fast and scales to millions (**Stage 1**), but it never sees the pair together and can miss nuance. A **cross-encoder** feeds query+document **together** (`[CLS] query [SEP] doc [SEP]`) with full token-level attention and outputs one relevance score — far more accurate but too slow to run on everything, so it only **reranks** a small candidate set (**Stage 2**). **Why it matters here:** this is the same two-stage funnel from the RAG lecture — but here you'll *train both halves* and measure each one's contribution.

### 5. Why fine-tuning matters

> **🪜 Mental model:** a world atlas vs a city street map — the atlas knows continents but not that "5th & Main" is next to your office.

Pre-trained models learn **general** semantics (Wikipedia, web crawls) — broad but shallow. In a specialized domain they miss the relationships that matter: that "SWIFT" means payment messaging, or that a support ticket's *problem description* should match a *solution pattern*, not just shared keywords. This is **domain mismatch** (vocabulary) and **task mismatch** (the model's idea of "similar" ≠ your idea of "relevant"). **Fine-tuning** continues training on your data, reshaping the vector space so your relevant pairs land close. **Why it matters here:** it's the notebook's central experiment — Experiment A evaluates pre-trained models, Experiment B fine-tunes them and shows the retrieval-metric lift.

### 6. SBERT — how sentence embeddings became practical

> **🪜 Mental model:** BERT could *grade* a sentence pair but not *file* sentences for fast lookup; SBERT taught it to file them.

When **BERT** (2018) arrived it was state-of-the-art but built for *token-level* tasks. To compare two sentences you had to feed them in **together** — accurate but catastrophically slow: finding the most similar pair among 10,000 sentences meant ~50 million BERT runs, about **65 hours**. The fast alternative — averaging BERT's token vectors (**mean pooling**) — produced embeddings worse than 2014-era GloVe. In 2019 **Reimers & Gurevych** (UKP Lab) published **SBERT**, using a **Siamese** architecture (two weight-shared BERT copies) so BERT learns to output meaningful *independent* sentence vectors comparable by plain cosine similarity — turning 65 hours into seconds. **Why it matters here:** every bi-encoder in the notebook (and in RAG) is an SBERT-style model; this explains *why* they exist.

### 7. How SBERT is trained (and the two loss functions)

> **🪜 Mental model:** teach by examples of "these mean the same, those don't" — then nudge the map until the geometry agrees.

SBERT starts from a pre-trained transformer (BERT/RoBERTa/MPNet) and fine-tunes it on **NLI (Natural Language Inference)** data — sentence pairs labeled **entailment / contradiction / neutral** (SNLI ~570K pairs, MultiNLI ~430K). The notebook then trains *its own* models with two different losses, one per architecture:

- **Triplet loss (bi-encoder):** given `(anchor, positive, negative)`, push the positive closer to the anchor than the negative by a **margin** (cosine distance, `margin=0.5`). Teaches *relative ordering*.
- **BCE loss (cross-encoder):** treat `(query, document)` as a yes/no classification, output a 0–1 relevance probability, penalize the gap to the true label. Teaches *absolute scoring*.

**Why it matters here:** these are the actual `losses.TripletLoss` and `BinaryCrossEntropyLoss` calls in Experiment B — knowing which loss fits which architecture is a common interview question.

---

## 🔥 The experiments & metrics — at a glance

**Experiments** (dataset: `sdiazlor/rag-human-rights-from-files`; vectors indexed in **FAISS**, tracked in **Opik**):

| | Bi-encoder | Cross-encoder |
|---|---|---|
| **Model** | `all-MiniLM-L6-v2` (384-dim) | `ms-marco-MiniLM-L-6-v2` |
| **Role** | Stage 1 — fast retrieval | Stage 2 — precise reranking |
| **Fine-tune loss** | Triplet loss (margin 0.5, cosine) | BCE loss |
| **Training** | 10 epochs, batch 16 | 10 epochs, batch 16 |
| **Measured by** | recall@1/5/10, MRR | rerank_recall@1/5/10, rerank_MRR |

The notebook compares **default vs fine-tuned** for each and plots the lift. (Exact result numbers are computed at runtime, so confirm them when you run it — the point is the *direction*: domain fine-tuning improves retrieval.)

**Retrieval metrics primer** — all "@K" (only the top K results count):

| Metric | Answers | Use when |
|---|---|---|
| **Precision@K** | Of what I returned, how much was relevant? | Irrelevant results are costly |
| **Recall@K** | Of all relevant docs, how many did I find? | *Missing* a doc is costly (RAG, legal, medical) |
| **Hits@K** | Did *any* relevant doc appear in top K? | One good result is enough (Q&A) |
| **MRR@K** | How high is the *first* relevant result? | Position of first hit matters |
| **MAP@K** | Are relevant docs ranked *high*? | Whole ranked list matters (reranking) |
| **NDCG@K** | Ranking quality with *graded* relevance | Some results are "more relevant" than others |

---

## 🧮 The one identity to internalize

The notebook proves this with code, and it's a classic interview point:

```
If vectors are normalized to unit length (‖v‖ = 1):
    cosine_similarity(a, b)  ==  dot_product(a, b)
```

**Word-by-word translation:** "cosine similarity is the dot product divided by both vectors' lengths; if both lengths are already 1, dividing by 1 changes nothing — so the two metrics give identical scores (and identical rankings)."

**Worked reading:** normalize Strawberry `[4,0,1]` and Blueberry `[3,0,1]` to unit length, and their cosine similarity and dot product match to many decimal places. **The takeaway:** if your model outputs normalized embeddings (most do), cosine vs dot product is a non-decision — they rank identically. The distinction only bites with **un-normalized** vectors, where dot product favors longer ones.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–25 | What embeddings are; cosine similarity; heatmap; 2D/3D PCA; tiny semantic search | **Focus.** The foundation. |
| 28 | `king − man + woman ≈ queen` intuition | **Read.** The "meaning = geometry" proof. |
| 29–56 | Four distance metrics + normalization (with toy-vector code) | **Focus.** Build the comparison table in your head. |
| 58–65 | Ways to generate embeddings (local vs OpenAI vs others) | **Read.** Note the dimension/cost trade-offs. |
| 66–70 | Bi-encoder vs cross-encoder; two-stage pipeline | **Focus.** Core architecture. |
| 72–86 | Why fine-tuning matters; SBERT history & training | **Focus.** The "why these models exist" story. |
| 87–99 | Setup; the human-rights dataset; data prep for both model types | **Skim.** Note triplet vs (query,doc,label) formats. |
| 100–117 | Experiment A: evaluate pre-trained bi & cross encoders + metrics primer | **Focus.** Read the metrics primer carefully. |
| 118–141 | Experiment B: fine-tune bi (triplet) + cross (BCE); compare to default | **Focus + slow down.** The loss functions are key. |
| 142–145 | Hard-negative mining primer | **Read.** The Apple/Fuji example. |
| 146–158 | Two-stage RAG with fine-tuned models + GPT-4o-mini; summary | **Read.** Ties it back to RAG. |

---

## ✅ Walk-away checklist

After the notebook, you should be able to say, in your own words:

- [ ] What an embedding is and why "meaning becomes geometry."
- [ ] The four distance metrics and when each is appropriate.
- [ ] Why normalization makes cosine similarity and dot product identical.
- [ ] Bi-encoder vs cross-encoder, and which loss trains each (triplet vs BCE).
- [ ] Why pre-trained embeddings need fine-tuning for a real domain.
- [ ] Why SBERT was needed (BERT's 65-hour problem) and what the Siamese setup fixed.
- [ ] What Precision@K, Recall@K, and MRR each measure, and when to prefer each.
- [ ] The difference between a soft and a hard negative, and why hard negatives train better rerankers.

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. Your embedding model outputs normalized vectors. Does it matter whether you rank results by cosine similarity or dot product? Why?
2. You're fine-tuning a **bi-encoder** and a **cross-encoder**. Which loss goes with which, and what does each teach?
3. In one sentence, why couldn't vanilla BERT be used for fast semantic search over 10,000 sentences, and what did SBERT change?
4. A legal-search RAG system *cannot* afford to miss a relevant statute. Which retrieval metric should you optimize, and why that one over Precision@K?
5. For the query "Where was Apple founded?", classify each as a soft or hard negative: (a) a document about a bridge in Arkansas, (b) a document about the Fuji apple cultivar. Why does the distinction matter for training?

<details>
<summary>Answers</summary>

1. **No, it doesn't matter** — for unit-length (normalized) vectors, cosine similarity equals the dot product, so they produce identical scores and identical rankings. (It only matters for un-normalized vectors, where dot product favors longer ones.)
2. **Bi-encoder → triplet loss** (anchor/positive/negative; teaches *relative ordering* — positive closer than negative by a margin). **Cross-encoder → BCE loss** (query/document/label; teaches *absolute* 0–1 relevance scoring).
3. Comparing sentences with BERT required feeding each pair in *together*, so finding the most similar pair among 10K sentences meant ~50M runs (~65 hours); **SBERT** used a **Siamese** architecture to make BERT produce meaningful *independent* sentence vectors you can pre-compute and compare with cosine similarity in seconds.
4. **Recall@K** — it measures the fraction of *all* relevant documents you actually retrieved, so it directly penalizes *missing* a relevant statute. Precision@K only measures how clean your returned list is, and would reward a tidy list even if it omitted critical documents.
5. (a) bridge in Arkansas = **soft (easy) negative** — obviously unrelated, trivial to reject. (b) Fuji apple cultivar = **hard negative** — shares the surface word "Apple" but is the wrong meaning. Hard negatives matter because they force the reranker to learn fine-grained distinctions; training only on easy negatives produces a model that fails on tricky, look-alike cases.

</details>

[🔝 Back to top](#top)
